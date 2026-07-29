"""Factored discrete receding-horizon models.

``build_factored_rollout`` keeps persistent state in named categorical blocks
while using ordinary CFFG primitives for every coupling.  A transition can be
selected by the action alone or by a declared Cartesian product of parent
blocks and the action.  Observations likewise read one block or a declared
product.  Cartesian coordinates use NumPy/C order: the rightmost component is
the fastest-moving index.

No product factor is registered.  The builder lowers each Cartesian product
to deterministic ``transition_mixture`` nodes and lowers every fan-out to a
chain of ternary equality nodes.  Learned transition blobs therefore retain
their factored shapes even though product-valued message edges may exist.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np

from cffg.core.ir import (
    AgentSpec,
    BeliefBinding,
    BlobRef,
    BoundaryFactorization,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    LearnableSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)


@dataclass(frozen=True)
class CouplingSpec:
    """A named Cartesian product of state blocks, in C-order."""

    name: str
    blocks: tuple[str, ...]


@dataclass(frozen=True)
class StateBlockSpec:
    """One persistent categorical state component.

    ``transition`` is ``[U, K', K]`` for an action-only block.  When
    ``selector`` names a coupling over parent cards ``P1, ..., Pn``, its
    public shape is ``[P1, ..., Pn, U, K', K]``; the builder flattens only
    those selector axes for the derived ``transition_mixture`` blob.
    """

    name: str
    initial: np.ndarray
    transition: np.ndarray
    selector: str | None = None
    initial_learnable: str | None = None
    transition_learnable: str | None = None


@dataclass(frozen=True)
class ObservationSpec:
    """One categorical sensor over a block or declared coupling."""

    name: str
    source: str
    likelihood: np.ndarray
    preference: np.ndarray
    learnable: str | None = None


@dataclass(frozen=True)
class _FactorDraft:
    name: str
    type: str
    edges: dict[str, str]  # role -> logical net
    param_blobs: dict[str, str]


@dataclass(frozen=True)
class _PortDraft:
    name: str
    net: str
    role: PortRole


class _Graph:
    """A tiny private netlist lowered to degree-two FFG edges.

    A logical net may have any number of factor/port endpoints.  Materializing
    it inserts the minimum left-deep chain of ternary equality nodes.  This is
    deliberately builder-local; the reusable public fan-out primitive remains
    tracked by cff-mjku.
    """

    def __init__(self, name: str):
        self.template = Template(name)
        self.arrays: dict[str, np.ndarray] = {}
        self._cards: dict[str, int] = {}
        self._factors: list[_FactorDraft] = []
        self._ports: list[_PortDraft] = []

    def net(self, name: str, card: int) -> str:
        if card < 1:
            raise ValueError(f"factored rollout: net {name!r} has cardinality {card}")
        previous = self._cards.setdefault(name, card)
        if previous != card:
            raise ValueError(
                f"factored rollout: net {name!r} has conflicting cardinalities "
                f"{previous} and {card}"
            )
        return name

    def blob(self, name: str, value: np.ndarray) -> str:
        value = np.asarray(value)
        if name in self.arrays:
            if (
                self.arrays[name].shape != value.shape
                or not np.array_equal(self.arrays[name], value)
            ):
                raise ValueError(
                    f"factored rollout: generated blob {name!r} has conflicting values"
                )
            return name
        self.arrays[name] = value
        self.template.blobs[name] = BlobRef(name, f"{name}.npy")
        return name

    def factor(
        self,
        name: str,
        type: str,
        edges: dict[str, tuple[str, int]],
        *,
        param_blobs: dict[str, str],
    ) -> str:
        if any(draft.name == name for draft in self._factors):
            raise ValueError(f"factored rollout: duplicate factor name {name!r}")
        logical = {
            role: self.net(net, card) for role, (net, card) in edges.items()
        }
        self._factors.append(
            _FactorDraft(name, type, logical, dict(param_blobs))
        )
        return name

    def port(self, name: str, net: str, card: int, role: PortRole) -> str:
        if any(port.name == name for port in self._ports):
            raise ValueError(f"factored rollout: duplicate port name {name!r}")
        self._ports.append(_PortDraft(name, self.net(net, card), role))
        return name

    def materialize(
        self,
    ) -> tuple[Template, dict[tuple[str, str, str], str]]:
        endpoints: dict[str, list[tuple[str, str, str]]] = {
            net: [] for net in self._cards
        }
        for factor in self._factors:
            for role, net in factor.edges.items():
                endpoints[net].append(("factor", factor.name, role))
        for port in self._ports:
            endpoints[port.net].append(("port", port.name, ""))

        variables: dict[str, VariableSpec] = {}
        endpoint_var: dict[tuple[str, str, str], str] = {}
        equalities: list[FactorSpec] = []

        for net, net_endpoints in endpoints.items():
            card = self._cards[net]
            count = len(net_endpoints)
            if count < 2:
                raise ValueError(
                    f"factored rollout: logical net {net!r} has {count} endpoint; "
                    "terminate it with a factor or port"
                )
            if count == 2:
                variables[net] = VariableSpec(net, card)
                for endpoint in net_endpoints:
                    endpoint_var[endpoint] = net
                continue

            leaves = [f"{net}/branch/{index}" for index in range(count)]
            for leaf, endpoint in zip(leaves, net_endpoints, strict=True):
                variables[leaf] = VariableSpec(leaf, card)
                endpoint_var[endpoint] = leaf

            carry: str | None = None
            for index in range(count - 2):
                last = index == count - 3
                if index == 0:
                    a, b = leaves[0], leaves[1]
                    next_leaf = 2
                else:
                    assert carry is not None
                    a, b = carry, leaves[index + 1]
                    next_leaf = index + 2
                if last:
                    c = leaves[next_leaf]
                else:
                    c = f"{net}/carry/{index}"
                    variables[c] = VariableSpec(c, card)
                equalities.append(
                    FactorSpec(
                        f"branch/{net}/{index}",
                        "equality",
                        edges={"a": a, "b": b, "c": c},
                    )
                )
                carry = c

        factors: dict[str, FactorSpec] = {}
        for draft in self._factors:
            factors[draft.name] = FactorSpec(
                draft.name,
                draft.type,
                edges={
                    role: endpoint_var[("factor", draft.name, role)]
                    for role in draft.edges
                },
                param_blobs=draft.param_blobs,
            )
        factors.update({factor.name: factor for factor in equalities})
        ports = {
            port.name: PortSpec(
                port.name,
                endpoint_var[("port", port.name, "")],
                port.role,
            )
            for port in self._ports
        }
        self.template.variables = variables
        self.template.factors = factors
        self.template.ports = ports
        return self.template, endpoint_var


_SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def _check_name(kind: str, name: str) -> None:
    if not _SAFE_NAME.fullmatch(name):
        raise ValueError(
            f"factored rollout: {kind} name {name!r} must match "
            "[A-Za-z][A-Za-z0-9_-]*"
        )


def _as_probability(value: np.ndarray, label: str, *, axis: int) -> np.ndarray:
    """Validate a normalized probability object without changing zero support."""
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(float)
    if array.ndim <= axis:
        raise ValueError(
            f"factored rollout: {label} needs axis {axis}, got shape {array.shape}"
        )
    if array.size == 0 or not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"factored rollout: {label} must be finite and non-negative")
    sums = array.sum(axis=axis)
    dtype = array.dtype
    eps = np.finfo(dtype).eps
    width = array.shape[axis]
    tolerance = eps * max(32, 8 * width)
    if np.any(np.abs(sums - 1) > tolerance):
        raise ValueError(
            f"factored rollout: {label} must normalize along axis {axis}; "
            f"sums are {sums}"
        )
    return array


def _deterministic_product(
    left_card: int, right_card: int, dtype: np.dtype
) -> np.ndarray:
    """``out = left * right_card + right`` as a TM blob [right,out,left]."""
    table = np.zeros(
        (right_card, left_card * right_card, left_card), dtype=dtype
    )
    left = np.arange(left_card)[:, None]
    right = np.arange(right_card)[None, :]
    table[
        np.broadcast_to(right, (left_card, right_card)),
        left * right_card + right,
        np.broadcast_to(left, (left_card, right_card)),
    ] = 1
    return table


def _unique_specs(
    blocks: tuple[StateBlockSpec, ...],
    couplings: tuple[CouplingSpec, ...],
) -> tuple[dict[str, StateBlockSpec], dict[str, CouplingSpec]]:
    block_map: dict[str, StateBlockSpec] = {}
    for block in blocks:
        _check_name("state block", block.name)
        if block.name in block_map:
            raise ValueError(
                f"factored rollout: duplicate state block {block.name!r}"
            )
        block_map[block.name] = block
    if not block_map:
        raise ValueError("factored rollout: at least one state block is required")

    coupling_map: dict[str, CouplingSpec] = {}
    for coupling in couplings:
        _check_name("coupling", coupling.name)
        if coupling.name in block_map or coupling.name in coupling_map:
            raise ValueError(
                f"factored rollout: coupling name {coupling.name!r} is not unique"
            )
        if not coupling.blocks:
            raise ValueError(
                f"factored rollout: coupling {coupling.name!r} has no blocks"
            )
        if len(set(coupling.blocks)) != len(coupling.blocks):
            raise ValueError(
                f"factored rollout: coupling {coupling.name!r} repeats a block"
            )
        missing = sorted(set(coupling.blocks) - set(block_map))
        if missing:
            raise ValueError(
                f"factored rollout: coupling {coupling.name!r} references "
                f"unknown blocks {missing}"
            )
        coupling_map[coupling.name] = coupling
    return block_map, coupling_map


def build_factored_rollout(
    name: str,
    *,
    steps: int,
    blocks: Iterable[StateBlockSpec],
    actions: np.ndarray,
    observation: ObservationSpec,
    couplings: Iterable[CouplingSpec] = (),
    past: int = 0,
    gfe: bool = True,
    learn: tuple[str, ...] = (),
    learned_a: bool = False,
    learned_c: bool = False,
    epistemic_b: bool = False,
) -> tuple[Template, dict[str, np.ndarray]]:
    """Build a factored planning/perception window and its parameter arrays.

    ``past`` follows :func:`cffg.models.rollout.build_rollout`: past slices
    expose action/observation DATA ports; future slices carry an action prior
    and observation preference (with P-substitution when ``gfe`` is true).

    The three boolean compatibility arguments are accepted because
    :class:`cffg.agent.RollingAgent` supplies them to every rollout builder.
    Their parameter-uncertainty composites remain separate tickets and are
    rejected when enabled here.
    """
    if steps < 1:
        raise ValueError("factored rollout: steps must be >= 1")
    if not 0 <= past <= steps:
        raise ValueError(
            f"factored rollout: past must be between 0 and steps, got {past}"
        )
    if learned_a or learned_c or epistemic_b:
        requested = [
            label
            for label, enabled in (
                ("learned_a", learned_a),
                ("learned_c", learned_c),
                ("epistemic_b", epistemic_b),
            )
            if enabled
        ]
        raise ValueError(
            "factored rollout: parameter-uncertainty composites are not part "
            f"of this builder yet (requested {requested})"
        )

    block_tuple = tuple(blocks)
    coupling_tuple = tuple(couplings)
    block_map, coupling_map = _unique_specs(block_tuple, coupling_tuple)
    _check_name("observation", observation.name)

    action_prior = _as_probability(actions, "action prior", axis=0)
    if action_prior.ndim != 1:
        raise ValueError(
            f"factored rollout: action prior must be 1-D, got {action_prior.shape}"
        )
    n_actions = action_prior.shape[0]

    cards: dict[str, int] = {}
    initials: dict[str, np.ndarray] = {}
    transitions: dict[str, np.ndarray] = {}
    for block in block_tuple:
        initial = _as_probability(
            block.initial, f"initial belief for {block.name!r}", axis=0
        )
        if initial.ndim != 1:
            raise ValueError(
                f"factored rollout: initial belief for {block.name!r} must be "
                f"1-D, got {initial.shape}"
            )
        card = initial.shape[0]
        cards[block.name] = card
        initials[block.name] = initial

    for block in block_tuple:
        parents: tuple[str, ...] = ()
        if block.selector is not None:
            coupling = coupling_map.get(block.selector)
            if coupling is None:
                raise ValueError(
                    f"factored rollout: block {block.name!r} selects on unknown "
                    f"coupling {block.selector!r}"
                )
            if block.name in coupling.blocks:
                raise ValueError(
                    f"factored rollout: block {block.name!r} is already the "
                    "transition input and must not repeat itself in its selector"
                )
            parents = coupling.blocks
        expected = (
            *(cards[parent] for parent in parents),
            n_actions,
            cards[block.name],
            cards[block.name],
        )
        transition = np.asarray(block.transition)
        if transition.shape != expected:
            raise ValueError(
                f"factored rollout: transition for {block.name!r} has shape "
                f"{transition.shape}, expected {expected}"
            )
        flat = transition.reshape(
            (int(np.prod(expected[:-2])), *expected[-2:])
        )
        transitions[block.name] = _as_probability(
            flat, f"transition for {block.name!r}", axis=1
        )

    source_blocks: tuple[str, ...]
    if observation.source in block_map:
        source_blocks = (observation.source,)
    elif observation.source in coupling_map:
        source_blocks = coupling_map[observation.source].blocks
    else:
        raise ValueError(
            f"factored rollout: observation source {observation.source!r} is "
            "neither a state block nor a coupling"
        )
    source_card = int(np.prod([cards[block] for block in source_blocks]))
    likelihood = np.asarray(observation.likelihood)
    if likelihood.ndim != 2 or likelihood.shape[1] != source_card:
        raise ValueError(
            f"factored rollout: observation likelihood has shape "
            f"{likelihood.shape}, expected [n_obs, {source_card}]"
        )
    likelihood = _as_probability(
        likelihood, f"observation {observation.name!r}", axis=0
    )
    preference = _as_probability(
        observation.preference,
        f"preference for observation {observation.name!r}",
        axis=0,
    )
    if preference.ndim != 1 or preference.shape[0] != likelihood.shape[0]:
        raise ValueError(
            f"factored rollout: preference has shape {preference.shape}, "
            f"expected ({likelihood.shape[0]},)"
        )

    declared_learnables = {
        candidate
        for block in block_tuple
        for candidate in (
            block.initial_learnable,
            block.transition_learnable,
        )
        if candidate is not None
    }
    if observation.learnable is not None:
        declared_learnables.add(observation.learnable)
    duplicates = [
        candidate
        for candidate in declared_learnables
        if sum(
            candidate == value
            for block in block_tuple
            for value in (
                block.initial_learnable,
                block.transition_learnable,
            )
        )
        + int(candidate == observation.learnable)
        > 1
    ]
    if duplicates:
        raise ValueError(
            f"factored rollout: learnable names must be unique, got "
            f"{sorted(duplicates)}"
        )
    unknown_learn = sorted(set(learn) - declared_learnables)
    if unknown_learn:
        raise ValueError(
            f"factored rollout: unknown learnables {unknown_learn}; declared "
            f"{sorted(declared_learnables)}"
        )

    dtype = np.result_type(
        action_prior.dtype,
        likelihood.dtype,
        preference.dtype,
        *(value.dtype for value in initials.values()),
        *(value.dtype for value in transitions.values()),
    )
    if not np.issubdtype(dtype, np.floating):
        dtype = np.dtype(float)

    graph = _Graph(name)
    action_blob = graph.blob("action/prior", action_prior)
    observation_blob = graph.blob(
        f"observation/{observation.name}/likelihood", likelihood
    )
    preference_blob = graph.blob(
        f"observation/{observation.name}/preference", preference
    )
    initial_blobs = {
        block.name: graph.blob(
            f"block/{block.name}/initial", initials[block.name]
        )
        for block in block_tuple
    }
    transition_blobs = {
        block.name: graph.blob(
            f"block/{block.name}/transition", transitions[block.name]
        )
        for block in block_tuple
    }

    state_nets = {
        (block.name, index): (f"state/{block.name}/{index}", cards[block.name])
        for block in block_tuple
        for index in range(steps + 1)
    }
    for block in block_tuple:
        graph.factor(
            f"prior/{block.name}",
            "categorical",
            {"out": state_nets[(block.name, 0)]},
            param_blobs={"statistic": initial_blobs[block.name]},
        )

    product_cache: dict[
        tuple[tuple[str, int], ...], tuple[str, int]
    ] = {}

    def product(
        label: str, components: tuple[tuple[str, int], ...]
    ) -> tuple[str, int]:
        if len(components) == 1:
            return components[0]
        if components in product_cache:
            return product_cache[components]
        left_net, left_card = components[0]
        for index, (right_net, right_card) in enumerate(components[1:]):
            out_card = left_card * right_card
            out_net = f"product/{label}/{index}"
            blob = graph.blob(
                f"deterministic/product/{left_card}x{right_card}",
                _deterministic_product(left_card, right_card, dtype),
            )
            graph.factor(
                f"product/{label}/{index}",
                "transition_mixture",
                {
                    "out": (out_net, out_card),
                    "in": (left_net, left_card),
                    "selector": (right_net, right_card),
                },
                param_blobs={"matrices": blob},
            )
            left_net, left_card = out_net, out_card
        product_cache[components] = (left_net, left_card)
        return left_net, left_card

    transition_factors: dict[str, list[str]] = {
        block.name: [] for block in block_tuple
    }
    observation_factors: list[str] = []
    action_factor_names: dict[int, str] = {}

    for index in range(1, steps + 1):
        action_net = (f"action/{index}", n_actions)
        if index <= past:
            graph.port(
                f"action{index}",
                action_net[0],
                action_net[1],
                PortRole.DATA,
            )
        else:
            factor_name = f"action_prior/{index}"
            graph.factor(
                factor_name,
                "categorical",
                {"out": action_net},
                param_blobs={"statistic": action_blob},
            )
            action_factor_names[index] = factor_name

        for block in block_tuple:
            selector_components = tuple(
                state_nets[(parent, index - 1)]
                for parent in (
                    coupling_map[block.selector].blocks
                    if block.selector is not None
                    else ()
                )
            ) + (action_net,)
            selector = product(
                f"selector/{block.name}/{index}", selector_components
            )
            factor_name = f"transition/{block.name}/{index}"
            graph.factor(
                factor_name,
                "transition_mixture",
                {
                    "out": state_nets[(block.name, index)],
                    "in": state_nets[(block.name, index - 1)],
                    "selector": selector,
                },
                param_blobs={"matrices": transition_blobs[block.name]},
            )
            transition_factors[block.name].append(factor_name)

        observation_source = product(
            f"observation/{observation.name}/{index}",
            tuple(state_nets[(block, index)] for block in source_blocks),
        )
        outcome_net = (f"observation/{observation.name}/out/{index}", likelihood.shape[0])
        factor_name = f"observation/{observation.name}/{index}"
        graph.factor(
            factor_name,
            "matrix_cat",
            {"out": outcome_net, "in": observation_source},
            param_blobs={"matrix": observation_blob},
        )
        observation_factors.append(factor_name)
        if index <= past:
            graph.port(
                f"observation{index}",
                outcome_net[0],
                outcome_net[1],
                PortRole.DATA,
            )
        else:
            graph.factor(
                f"preference/{observation.name}/{index}",
                "categorical",
                {"out": outcome_net},
                param_blobs={"statistic": preference_blob},
            )

    for block in block_tuple:
        final_net, final_card = state_nets[(block.name, steps)]
        graph.port(
            f"horizon/{block.name}", final_net, final_card, PortRole.GLUE
        )

    boundary_index = past if past else 0
    boundary_markers: dict[str, str] = {}
    if len(block_tuple) > 1:
        for block in block_tuple:
            marker = f"boundary/{block.name}/{boundary_index}"
            marker_out = (f"{marker}/unit", 1)
            matrix_blob = graph.blob(
                f"boundary/marker/{cards[block.name]}",
                np.ones((1, cards[block.name]), dtype=dtype),
            )
            unit_blob = graph.blob(
                "boundary/unit", np.ones(1, dtype=dtype)
            )
            graph.factor(
                marker,
                "matrix_cat",
                {
                    "out": marker_out,
                    "in": state_nets[(block.name, boundary_index)],
                },
                param_blobs={"matrix": matrix_blob},
            )
            graph.factor(
                f"{marker}/unit",
                "categorical",
                {"out": marker_out},
                param_blobs={"statistic": unit_blob},
            )
            boundary_markers[block.name] = marker

    template, endpoint_var = graph.materialize()

    for index, factor_name in enumerate(observation_factors, start=1):
        if index > past and gfe:
            template.constraints.append(
                ConstraintSpec(
                    ConstraintKind.P_SUBSTITUTION,
                    factor_name,
                    (("substituted_role", "out"),),
                )
            )

    beliefs: list[BeliefBinding] = []
    active_learn = set(learn)
    for block in block_tuple:
        if len(block_tuple) > 1:
            marker = boundary_markers[block.name]
            template.constraints.append(
                ConstraintSpec(ConstraintKind.MEAN_FIELD, marker)
            )
            posterior = endpoint_var[("factor", marker, "in")]
            factorization = BoundaryFactorization.MEAN_FIELD
            constraints = (marker,)
        elif boundary_index == 0:
            posterior = endpoint_var[
                ("factor", f"prior/{block.name}", "out")
            ]
            factorization = BoundaryFactorization.JOINT
            constraints = ()
        else:
            posterior = endpoint_var[
                (
                    "factor",
                    f"transition/{block.name}/{boundary_index}",
                    "out",
                )
            ]
            factorization = BoundaryFactorization.JOINT
            constraints = ()
        beliefs.append(
            BeliefBinding(
                name=block.name,
                blocks=(block.name,),
                prior=f"prior/{block.name}",
                posterior=posterior,
                factorization=factorization,
                constraints=constraints,
                episode_learnable=(
                    block.initial_learnable
                    if block.initial_learnable in active_learn
                    else None
                ),
            )
        )

    if past:
        action_readout = template.ports["action1"].variable
        action_port = "action1"
        observation_port = "observation1"
    else:
        action_readout = endpoint_var[
            ("factor", action_factor_names[1], "out")
        ]
        action_port = None
        observation_port = None
    template.agent = AgentSpec(
        action=action_readout,
        action_port=action_port,
        observation_port=observation_port,
        # AgentSpec serializes beliefs as a canonical name-keyed mapping.
        # Emit the same order here so an in-memory builder result equals its
        # JSON/.cffgz round trip, independent of block declaration order.
        beliefs=tuple(sorted(beliefs, key=lambda binding: binding.name)),
    )

    for block in block_tuple:
        if (
            block.initial_learnable is not None
            and block.initial_learnable in active_learn
        ):
            sources = (
                (f"prior/{block.name}",) if past else ()
            )
            template.learnables[block.initial_learnable] = LearnableSpec(
                block.initial_learnable,
                "dirichlet",
                (initial_blobs[block.name],),
                sources,
            )
        if (
            block.transition_learnable is not None
            and block.transition_learnable in active_learn
        ):
            sources = tuple(
                transition_factors[block.name][:past]
            )
            template.learnables[block.transition_learnable] = LearnableSpec(
                block.transition_learnable,
                "dirichlet",
                (transition_blobs[block.name],),
                sources,
            )
    if observation.learnable is not None and observation.learnable in active_learn:
        template.learnables[observation.learnable] = LearnableSpec(
            observation.learnable,
            "dirichlet",
            (observation_blob,),
            tuple(observation_factors[:past]),
        )

    return template, graph.arrays


def _corridor_position_transition(
    length: int, modes: int, dtype: np.dtype
) -> np.ndarray:
    """Mode parity reverses left/right; stay is shared by every mode."""
    n_actions = 3
    transition = np.zeros(
        (modes, n_actions, length, length), dtype=dtype
    )
    moves = (-1, 0, 1)
    for mode in range(modes):
        direction = -1 if mode % 2 else 1
        for action, move in enumerate(moves):
            for position in range(length):
                target = min(
                    max(position + direction * move, 0), length - 1
                )
                transition[mode, action, target, position] = 1
    return transition


def _corridor_mode_transition(
    modes: int, dtype: np.dtype
) -> np.ndarray:
    return np.stack([np.eye(modes, dtype=dtype)] * 3)


def build_factored_corridor(
    past: int = 0,
    steps: int = 3,
    *,
    length: int = 6,
    modes: int = 2,
    position_prior: np.ndarray | None = None,
    mode_prior: np.ndarray | None = None,
    position_transition: np.ndarray | None = None,
    mode_transition: np.ndarray | None = None,
    observation_matrix: np.ndarray | None = None,
    observation_source: str = "state",
    preference: np.ndarray | None = None,
    goal_utility: float = 3.0,
    actions: np.ndarray | None = None,
    dtype: np.dtype = np.float64,
    gfe: bool = True,
    learn: tuple[str, ...] = (),
    learned_a: bool = False,
    learned_c: bool = False,
    epistemic_b: bool = False,
) -> tuple[Template, dict[str, np.ndarray]]:
    """The canonical two-block corridor: position × persistent mode.

    Modes reverse the meaning of left/right by parity, making the position
    transition genuinely product-selected.  The sensor can identify the full
    product (default), observe position alone, or use any caller-supplied
    normalized likelihood over either source.
    """
    if length < 2:
        raise ValueError("factored corridor: length must be >= 2")
    if modes < 1:
        raise ValueError("factored corridor: modes must be >= 1")
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.floating):
        raise ValueError(f"factored corridor: dtype must be floating, got {dtype}")

    if position_prior is None:
        position_prior = np.zeros(length, dtype=dtype)
        position_prior[0] = 1
    if mode_prior is None:
        mode_prior = np.zeros(modes, dtype=dtype)
        mode_prior[0] = 1
    if position_transition is None:
        position_transition = _corridor_position_transition(
            length, modes, dtype
        )
    if mode_transition is None:
        mode_transition = _corridor_mode_transition(modes, dtype)
    if actions is None:
        actions = np.full(3, 1 / 3, dtype=dtype)

    couplings = [
        CouplingSpec("mode_action", ("mode",)),
        CouplingSpec("state", ("position", "mode")),
    ]
    source_cards = {
        "position": length,
        "mode": modes,
        "state": length * modes,
    }
    if observation_source not in source_cards:
        raise ValueError(
            "factored corridor: observation_source must be 'position', "
            f"'mode', or 'state', got {observation_source!r}"
        )
    source_card = source_cards[observation_source]
    if observation_matrix is None:
        observation_matrix = np.eye(source_card, dtype=dtype)
    likelihood = np.asarray(observation_matrix)
    if preference is None:
        if likelihood.ndim != 2 or likelihood.shape[1] != source_card:
            raise ValueError(
                f"factored corridor: observation matrix shape {likelihood.shape} "
                f"does not match source cardinality {source_card}"
            )
        if observation_source == "position":
            target_columns = (length - 1,)
        elif observation_source == "mode":
            target_columns = tuple(range(modes))
        else:
            target_columns = tuple(
                (length - 1) * modes + mode for mode in range(modes)
            )
        target_mass = likelihood[:, target_columns].mean(axis=1)
        utility = goal_utility * target_mass
        weights = np.exp(utility - utility.max())
        preference = weights / weights.sum()

    return build_factored_rollout(
        f"factored-corridor-{length}x{modes}[{past}|{steps}]",
        steps=steps,
        past=past,
        blocks=(
            StateBlockSpec(
                "position",
                np.asarray(position_prior, dtype=dtype),
                np.asarray(position_transition, dtype=dtype),
                selector="mode_action",
                transition_learnable="position_transition",
            ),
            StateBlockSpec(
                "mode",
                np.asarray(mode_prior, dtype=dtype),
                np.asarray(mode_transition, dtype=dtype),
                transition_learnable="mode_transition",
            ),
        ),
        couplings=tuple(couplings),
        actions=np.asarray(actions, dtype=dtype),
        observation=ObservationSpec(
            "sensor",
            observation_source,
            np.asarray(likelihood, dtype=dtype),
            np.asarray(preference, dtype=dtype),
        ),
        gfe=gfe,
        learn=learn,
        learned_a=learned_a,
        learned_c=learned_c,
        epistemic_b=epistemic_b,
    )

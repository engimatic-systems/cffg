"""Strict topology recognition for production D10 rSLDS components."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from cffg.core.ir import PortRole, Template


@dataclass(frozen=True)
class StructuredHybridTopology:
    """One whole-template recurrent switching chain owned by D10."""

    recurrence_members: tuple[str, ...]
    dynamics_members: tuple[str, ...]
    emission_members: tuple[str, ...]
    initial_regime_member: str
    initial_state_member: str
    regime_edges: tuple[tuple[str, ...], ...]
    state_edges: tuple[tuple[str, ...], ...]
    observation_edges: tuple[str, ...]
    owned_factors: frozenset[str]
    owned_edges: frozenset[str]
    cardinality: int
    dimension: int


@dataclass(frozen=True)
class StructuredHybridClassification:
    topology: StructuredHybridTopology | None
    reason: str


class _DisjointSet:
    def __init__(self, names):
        self.parent = {name: name for name in names}

    def find(self, name: str) -> str:
        parent = self.parent[name]
        if parent != name:
            self.parent[name] = self.find(parent)
        return self.parent[name]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _closures(template: Template) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    sets = _DisjointSet(template.variables)
    for factor in template.factors.values():
        if factor.type != "equality":
            continue
        variables = tuple(factor.edges.values())
        for variable in variables[1:]:
            sets.union(variables[0], variable)
    root_of = {name: sets.find(name) for name in template.variables}
    members: dict[str, list[str]] = defaultdict(list)
    for name, root in root_of.items():
        members[root].append(name)
    return root_of, {
        root: tuple(sorted(names)) for root, names in members.items()
    }


def _equality_closures_are_trees(
    template: Template,
    root_of: dict[str, str],
    closure_members: dict[str, tuple[str, ...]],
) -> bool:
    """Prove each variable/equality incidence component is a tree."""
    factors_by_root: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for factor in template.factors.values():
        if factor.type != "equality":
            continue
        variables = tuple(factor.edges.values())
        if len(set(variables)) != len(variables):
            return False
        factors_by_root[root_of[variables[0]]].append(variables)
    for root, variables in closure_members.items():
        factors = factors_by_root[root]
        incidences = sum(len(edges) for edges in factors)
        if incidences != len(variables) + len(factors) - 1:
            return False
    return True


def _same_blobs(template: Template, names: tuple[str, ...]) -> bool:
    return bool(names) and all(
        template.factors[name].param_blobs
        == template.factors[names[0]].param_blobs
        for name in names[1:]
    )


def _fail(reason: str) -> StructuredHybridClassification:
    return StructuredHybridClassification(None, reason)


def classify_structured_hybrid(template: Template) -> StructuredHybridClassification:
    """Recognize only the tied-boundary whole-template rSLDS derivation.

    Equality nodes are treated as aliases.  Any extra attachment, objective
    constraint, untied boundary law, or disconnected factor retains D3/D9.
    """
    recurrence = tuple(
        sorted(name for name, factor in template.factors.items() if factor.type == "recurrence_link")
    )
    if not recurrence:
        return _fail("no recurrence_link component")
    if template.constraints:
        return _fail("component constraints are outside ordinary D10 BFE")
    if template.learnables:
        return _fail(
            "declared recurrence learning retains D3/D9 until its D10 migration"
        )

    root_of, closure_members = _closures(template)
    if not _equality_closures_are_trees(template, root_of, closure_members):
        return _fail("equality closure is cyclic or repeats a variable")
    source_of = {
        name: root_of[template.factors[name].edges["in"]] for name in recurrence
    }
    destination_of = {
        name: root_of[template.factors[name].edges["out"]] for name in recurrence
    }
    drive_of = {
        name: root_of[template.factors[name].edges["drive"]] for name in recurrence
    }
    if any(source_of[name] == destination_of[name] for name in recurrence):
        return _fail("categorical recurrence self-loop")

    outgoing: dict[str, list[str]] = defaultdict(list)
    incoming: dict[str, list[str]] = defaultdict(list)
    for name in recurrence:
        outgoing[source_of[name]].append(name)
        incoming[destination_of[name]].append(name)
    if any(len(names) > 1 for names in (*outgoing.values(), *incoming.values())):
        return _fail("parallel or ambiguous categorical recurrence ownership")
    roots = set(source_of.values()) | set(destination_of.values())
    starts = sorted(root for root in roots if root not in incoming)
    ends = sorted(root for root in roots if root not in outgoing)
    if len(starts) != 1 or len(ends) != 1:
        return _fail("categorical recurrence component is cyclic or disconnected")

    ordered: list[str] = []
    regime_roots = [starts[0]]
    current = starts[0]
    while current in outgoing:
        name = outgoing[current][0]
        if name in ordered:
            return _fail("categorical recurrence component contains a cycle")
        ordered.append(name)
        current = destination_of[name]
        regime_roots.append(current)
    if set(ordered) != set(recurrence):
        return _fail("categorical recurrence component is disconnected")

    cardinalities = {
        template.variables[template.factors[name].edges["out"]].cardinality
        for name in ordered
    } | {
        template.variables[template.factors[name].edges["in"]].cardinality
        for name in ordered
    }
    dimensions = {
        template.variables[template.factors[name].edges["drive"]].dim
        for name in ordered
    }
    if len(cardinalities) != 1 or min(cardinalities) < 2:
        return _fail("recurrence cardinalities disagree or K is below two")
    if len(dimensions) != 1 or min(dimensions) < 1:
        return _fail("recurrence drive dimensions disagree")

    mixture_names = tuple(
        sorted(
            name
            for name, factor in template.factors.items()
            if factor.type == "gaussian_mixture_linear"
        )
    )
    dynamics: list[str] = []
    state_roots = [drive_of[ordered[0]]]
    for index, recurrence_name in enumerate(ordered):
        candidates = [
            name
            for name in mixture_names
            if root_of[template.factors[name].edges["selector"]]
            == destination_of[recurrence_name]
            and root_of[template.factors[name].edges["in"]]
            == drive_of[recurrence_name]
        ]
        if len(candidates) != 1:
            return _fail("each recurrence needs one matching destination-selected dynamics node")
        dynamics_name = candidates[0]
        dynamics.append(dynamics_name)
        next_state = root_of[template.factors[dynamics_name].edges["out"]]
        if next_state == drive_of[recurrence_name]:
            return _fail("Gaussian dynamics input and output share one closure")
        if index + 1 < len(ordered) and next_state != drive_of[ordered[index + 1]]:
            return _fail("Gaussian dynamics and recurrence time order disagree")
        state_roots.append(next_state)
    if set(dynamics) != set(mixture_names):
        return _fail("unowned switching dynamics node")
    if len(set(state_roots)) != len(state_roots):
        return _fail("Gaussian trajectory reuses a state closure")

    attachments: dict[str, list[str]] = defaultdict(list)
    for name, factor in template.factors.items():
        if factor.type == "equality":
            continue
        for variable in factor.edges.values():
            attachments[root_of[variable]].append(name)

    initial_regime = [
        name
        for name in attachments[regime_roots[0]]
        if template.factors[name].type == "categorical"
    ]
    initial_state = [
        name
        for name in attachments[state_roots[0]]
        if template.factors[name].type == "gaussian"
    ]
    if len(initial_regime) != 1 or len(initial_state) != 1:
        return _fail("D10 requires one tied categorical and Gaussian initial law")

    emissions: list[str] = []
    observations: list[str] = []
    ports_by_variable = defaultdict(list)
    for port in template.ports.values():
        ports_by_variable[port.variable].append(port)
    for state_root in state_roots:
        candidates = []
        for name in attachments[state_root]:
            factor = template.factors[name]
            if factor.type != "gaussian_linear" or root_of[factor.edges["in"]] != state_root:
                continue
            out_variable = factor.edges["out"]
            ports = ports_by_variable[out_variable]
            if len(ports) == 1 and ports[0].role is PortRole.DATA:
                candidates.append((name, out_variable))
        if len(candidates) != 1:
            return _fail("each Gaussian state needs one shared conditioned emission")
        emission, observation = candidates[0]
        emissions.append(emission)
        observations.append(observation)
    if len(set(emissions)) != len(emissions):
        return _fail("Gaussian trajectory reuses an emission owner")
    if len(set(observations)) != len(observations):
        return _fail("Gaussian trajectory reuses an observation boundary")

    ordered_recurrence = tuple(ordered)
    ordered_dynamics = tuple(dynamics)
    ordered_emissions = tuple(emissions)
    if not _same_blobs(template, ordered_recurrence):
        return _fail("recurrence parameters are not tied across time")
    if not _same_blobs(template, ordered_dynamics):
        return _fail("dynamics parameters are not tied across time")
    if not _same_blobs(template, ordered_emissions):
        return _fail("emission parameters are not tied across time")

    equality = {
        name for name, factor in template.factors.items() if factor.type == "equality"
    }
    owned = equality | set(ordered_recurrence) | set(ordered_dynamics) | set(ordered_emissions) | {
        initial_regime[0], initial_state[0]
    }
    if owned != set(template.factors):
        return _fail("extra factor attachment is outside the tied D10 boundary")
    owned_edges = set().union(
        *(set(closure_members[root]) for root in (*regime_roots, *state_roots))
    ) | set(observations)
    if owned_edges != set(template.variables):
        return _fail("extra or disconnected variable is outside the D10 component")
    if set(template.ports) != {
        ports_by_variable[name][0].name for name in observations
    }:
        return _fail("only the shared emission DATA boundary is derived")

    return StructuredHybridClassification(
        StructuredHybridTopology(
            recurrence_members=ordered_recurrence,
            dynamics_members=ordered_dynamics,
            emission_members=ordered_emissions,
            initial_regime_member=initial_regime[0],
            initial_state_member=initial_state[0],
            regime_edges=tuple(closure_members[root] for root in regime_roots),
            state_edges=tuple(closure_members[root] for root in state_roots),
            observation_edges=tuple(observations),
            owned_factors=frozenset(owned),
            owned_edges=frozenset(owned_edges),
            cardinality=next(iter(cardinalities)),
            dimension=next(iter(dimensions)),
        ),
        "eligible tied-boundary recurrent switching chain",
    )


__all__ = [
    "StructuredHybridClassification",
    "StructuredHybridTopology",
    "classify_structured_hybrid",
]

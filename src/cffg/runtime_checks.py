"""Host runtime finiteness and exact-support checks over current State data."""

from dataclasses import dataclass, field

import jax
import numpy as np

from cffg._immutable import FrozenArray
from cffg.compiled import NEG, Arena
from cffg.execution import parameters as _parameters
from cffg.kernel_runtime import (
    discrete_potential as _disc_potential,
)
from cffg.kernel_runtime import (
    parameters as _kernel_params,
)
from cffg.runtime_types import State


def _np_log_support(values: np.ndarray) -> np.ndarray:
    """Exact support of log-space values under the reserved NEG sentinel.

    Discrete messages are clamped to NEG after every update, and compiled
    exact-zero factor entries are NEG. Every other finite value is supported,
    regardless of magnitude: exponentiation underflows and a numeric
    threshold would merely move that same false-zero boundary elsewhere.
    """
    values = np.asarray(values)
    return values != np.asarray(NEG, dtype=values.dtype)


def state_is_finite(state: State) -> bool:
    """All numeric runtime leaves must remain finite (NEG is finite)."""
    return all(
        bool(np.isfinite(np.asarray(leaf)).all())
        for leaf in jax.tree_util.tree_leaves(state)
    )


@dataclass(frozen=True)
class StateChecks:
    """Immutable support metadata; numerical parameters are read from State."""

    arena: Arena
    edge_groups: tuple[tuple[int, FrozenArray], ...] = field(init=False, repr=False)

    def __post_init__(self):
        by_card: dict[int, list[list[int]]] = {}
        for edge in self.arena.edges.values():
            if not edge.dim and not edge.ef and not edge.param_shape:
                by_card.setdefault(edge.card, []).append([edge.slot0, edge.slot1])
        object.__setattr__(
            self,
            "edge_groups",
            tuple(
                (card, FrozenArray(np.asarray(pairs, dtype=np.int64).reshape(-1, 2)))
                for card, pairs in sorted(by_card.items())
            ),
        )

    def has_zero_support(self, state: State) -> bool:
        """Whether an edge or ordinary factor belief has empty exact support.

        Support is tested in log space by exact comparison with the reserved
        NEG sentinel, never by exponentiating or applying a magnitude
        threshold. exp() underflows for legitimate small potentials, while a
        threshold only creates another false-zero boundary. For an edge, the
        two facing messages must each support a state; summing first would
        lose which operand (if either) carried NEG.

        Vectorized over each edge-cardinality bucket and each group's member
        stack: the check runs on every exact-path run, so a
        per-member Python loop would put an O(T) interpreter cost back into
        the O(E) schedule."""
        messages = np.asarray(state.messages)
        for card, pairs in self.edge_groups:
            support = _np_log_support(
                messages[self.arena.message_indices(pairs[:, 0], card)]
            ) & _np_log_support(messages[self.arena.message_indices(pairs[:, 1], card)])
            if not np.all(np.any(support, axis=-1)):
                return True

        for group in self.arena.groups:
            supports = [  # per role [M, card]: exact support of the incoming
                _np_log_support(
                    messages[self.arena.message_indices(group.gather[:, r], card)]
                )
                for r, card in enumerate(group.cards)
            ]
            n = len(group.members)
            if group.type == "equality":
                joint = np.logical_and.reduce(supports)  # [M, k]
                if not np.all(np.any(joint, axis=-1)):
                    return True
                continue
            factor_support = _np_log_support(
                _disc_potential(group, _kernel_params(group, state.params))
            )
            for role, card in enumerate(group.cards):
                shape = [n] + [1] * len(group.cards)
                shape[1 + role] = card
                factor_support = factor_support & supports[role].reshape(shape)
            if not np.all(np.any(factor_support.reshape(n, -1), axis=-1)):
                return True

        for constraint in self.arena.constraint_groups:
            parameters = _parameters.constraint_parameters(constraint, state.params)
            supports = [
                np.flatnonzero(
                    _np_log_support(
                        messages[self.arena.message_indices(constraint.gather[i], card)]
                    )
                ).tolist()
                for i, card in enumerate(constraint.cards)
            ]
            if any(not support for support in supports):
                return True
            if constraint.type == "exactly_k":
                possible = {0}
                for support in supports:
                    possible = {
                        total + value for total in possible for value in support
                    }
                feasible = int(np.asarray(parameters.threshold)) in possible
            elif constraint.type == "at_most_k":
                feasible = sum(min(support) for support in supports) <= int(
                    np.asarray(parameters.threshold)
                )
            elif constraint.type == "table":
                feasible = any(
                    all(int(row[i]) in supports[i] for i in range(len(supports)))
                    for row in np.asarray(parameters.table)
                )
            else:  # alldifferent, n/card <= 10 by compiler validation

                def match(i: int, used: set[int], *, supports=supports) -> bool:
                    if i == len(supports):
                        return True
                    return any(
                        value not in used and match(i + 1, used | {value})
                        for value in supports[i]
                    )

                feasible = match(0, set())
            if not feasible:
                return True
        return False

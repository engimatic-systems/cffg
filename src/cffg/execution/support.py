"""The exact log-zero predicate shared by local rules and support checks."""

import jax
import jax.numpy as jnp

from cffg.compiled import NEG, ConstraintPlan, GroupPlan
from cffg.execution.parameters import ConstraintParameters
from cffg.kernel_runtime import discrete_potential


def log_support(values: jax.Array) -> jax.Array:
    """Entries other than the reserved sentinel; finite checks are separate."""
    return values != jnp.asarray(NEG, dtype=values.dtype)


def discrete_zero_support(
    plan: GroupPlan,
    incoming: tuple[jax.Array, ...],
    params: dict[str, jax.Array],
) -> jax.Array:
    """Whether any local factor belief has empty exact support."""
    if plan.type == "equality":
        joint = incoming[0]
        for support in incoming[1:]:
            joint = joint & support
        return ~jnp.all(jnp.any(joint, axis=-1))
    n = len(plan.members)
    support = log_support(discrete_potential(plan, params))
    for role, card in enumerate(plan.cards):
        shape = [n] + [1] * len(plan.cards)
        shape[1 + role] = card
        support = support & incoming[role].reshape(shape)
    return ~jnp.all(jnp.any(support.reshape(n, -1), axis=-1))


def constraint_zero_support(
    plan: ConstraintPlan,
    incoming: tuple[jax.Array, ...],
    params: ConstraintParameters,
) -> jax.Array:
    """Whether no legal tuple has an exactly supported value on every port."""
    if plan.type in ("exactly_k", "at_most_k"):
        # Binary choices give a contiguous interval of feasible counts.
        rows = jnp.stack(incoming)
        lower = jnp.sum(~rows[:, 0])
        enough = lower <= params.threshold
        if plan.type == "exactly_k":
            enough &= params.threshold <= jnp.sum(rows[:, 1])
        return ~enough | ~jnp.any(rows, axis=1).all()

    if plan.type == "table":
        allowed = jnp.ones(params.table.shape[0], dtype=bool)
        for role, support in enumerate(incoming):
            values = params.table[:, role]
            allowed &= (values >= 0) & (values < plan.cards[role]) & support[values]
        return ~allowed.any()

    # Compiler admission bounds the shared cardinality to ten. Reachability
    # tracks precisely which subsets can be assigned to successive ports.
    card = plan.cards[0]
    subsets = jnp.arange(1 << card, dtype=jnp.int32)
    bits = 1 << jnp.arange(card, dtype=jnp.int32)
    contains = (bits[:, None] & subsets) != 0
    previous = bits[:, None] ^ subsets

    def include(reachable, support):
        assigned = contains & reachable[previous] & support[:, None]
        return assigned.any(axis=0), None

    empty = subsets == 0
    reachable, _ = jax.lax.scan(include, empty, jnp.stack(incoming))
    return ~reachable.any()

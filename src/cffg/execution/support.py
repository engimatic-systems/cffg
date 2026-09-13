"""The exact log-zero predicate shared by local rules and support checks."""

import jax
import jax.numpy as jnp

from cffg.compiled import NEG, GroupPlan
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

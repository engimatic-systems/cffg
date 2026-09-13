"""The exact log-zero predicate shared by local rules and support checks."""

import jax
import jax.numpy as jnp

from cffg.compiled import NEG


def log_support(values: jax.Array) -> jax.Array:
    """Entries other than the reserved sentinel; finite checks are separate."""
    return values != jnp.asarray(NEG, dtype=values.dtype)

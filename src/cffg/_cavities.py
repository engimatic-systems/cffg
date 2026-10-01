"""Additive message products that never include the recipient's own input."""

import jax.numpy as jnp


def sum_other_legs(values):
    """Exclusive sums for arrays shaped [members, legs, ...], in linear space.

    Total-minus-self can erase weak information beside a strong recipient.
    Prefix and suffix sums exclude that recipient before any addition.
    """
    values = jnp.asarray(values)
    zero = jnp.zeros_like(values[:, :1])
    prefix = jnp.cumsum(values[:, :-1], axis=1, dtype=values.dtype)
    suffix = jnp.cumsum(values[:, :0:-1], axis=1, dtype=values.dtype)[:, ::-1]
    return (jnp.concatenate((zero, prefix), axis=1)
            + jnp.concatenate((suffix, zero), axis=1))

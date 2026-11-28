"""One committed change-or-retain transition event, projected on its declared simplex face."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln


class ChangeUpdate(NamedTuple):
    concentration: jax.Array
    valid: jax.Array


def _log_rising(a, c):
    """log Γ(a+c)/Γ(a), avoiding subtraction of log Γ at saturated counts."""
    # Subtract Stirling expansions algebraically before either log Gamma is
    # rounded. At a >= 16 the first omitted Bernoulli term is < 1e-17.
    def remainder(x):
        inv = 1 / x
        z = inv * inv
        polynomial = -691 / 360360
        for coefficient in (1/1188, -1/1680, 1/1260, -1/360, 1/12):
            polynomial = coefficient + z * polynomial
        return inv * polynomial

    x = jnp.maximum(a, 16)
    large = (c*jnp.log(x) + ((x + c - .5)*jnp.log1p(c/x) - c)
             + (remainder(x+c) - remainder(x)))
    ratio = jnp.where(a >= 16, large, gammaln(a+c) - gammaln(a))
    return jnp.where(c == 0, 0, jnp.where(c == 1, jnp.log(a), ratio))


def _trace(mean):
    # The complement of the dominant outcome can round out of 1 - mean.
    dominant = jnp.arange(mean.size) == jnp.argmax(mean)
    other = jnp.where(dominant, 0, mean)
    return jnp.sum(other * (1 - other)) + jnp.sum(other) * jnp.sum(jnp.where(dominant, mean, 0))


def update_column(prior, carry, counts, allowed, hazard):
    """Return one projected column. Invalid input/output returns the original carry.

    The last dimension of a vmapped call is outcomes. Counts may be fractional
    expected statistics, pooled once across all readers of the same parameter.
    With zero counts the result is the predicted law *before* the next event.
    """
    prior, carry, counts, allowed = map(jnp.asarray, (prior, carry, counts, allowed))
    if carry.ndim != 1 or carry.size == 0 or any(x.shape != carry.shape for x in (prior, counts, allowed)):
        raise ValueError("transition columns require nonempty vectors of one shape")
    if (carry.dtype not in (jnp.float32, jnp.float64) or prior.dtype != carry.dtype
            or counts.dtype != carry.dtype or allowed.dtype != jnp.bool_):
        raise TypeError("concentrations and counts need a shared real dtype and a boolean support mask")
    h = jnp.asarray(hazard, dtype=carry.dtype)
    if h.shape != ():
        raise ValueError("the transition-change hazard must be scalar")

    branches = jnp.stack((carry, prior))
    valid = (jnp.any(allowed) & jnp.all(jnp.isfinite(branches))
             & jnp.all(jnp.where(allowed, branches > 0, branches == 0))
             & jnp.all(jnp.isfinite(counts)) & jnp.all(jnp.where(allowed, counts >= 0, counts == 0))
             & (h >= 0) & (h <= 1))
    updated = branches + counts
    branch_totals = jnp.sum(branches, axis=1)
    totals = jnp.sum(updated, axis=1)
    valid &= jnp.all(jnp.isfinite(totals))
    means = updated / totals[:, None]
    log_z = (jnp.sum(_log_rising(jnp.where(allowed, branches, 1), counts), axis=1)
             - _log_rising(branch_totals, jnp.sum(counts)))
    log_w = jnp.stack((jnp.log1p(-h), jnp.log(h))) + log_z
    weights = jax.nn.softmax(log_w)
    mean = weights @ means
    dominant = jnp.arange(carry.size) == jnp.argmax(mean)
    traces = jax.vmap(_trace)(means)
    contrast = means[1] - means[0]
    contrast = jnp.where(dominant, -jnp.sum(jnp.where(dominant, 0, contrast)), contrast)
    within = traces / (1 + totals)
    variance = weights @ within + jnp.prod(weights) * jnp.sum(jnp.square(contrast))
    scale = (weights @ (traces * (totals / (1 + totals)))) / variance
    result = mean * scale
    # At the hazard endpoints or identical updated laws, no projection is needed.
    result = jnp.where(jnp.all(carry == prior), updated[0], result)
    result = jnp.where(jnp.sum(allowed) == 1, prior, result)
    result = jnp.where(h == 0, updated[0], jnp.where(h == 1, updated[1], result))
    valid &= jnp.all(jnp.isfinite(result)) & jnp.all(jnp.where(allowed, result > 0, result == 0))
    return ChangeUpdate(jnp.where(valid, result, carry), valid)

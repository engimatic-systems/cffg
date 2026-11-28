"""One committed change-or-retain transition event, projected on its declared simplex face."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import special as jax_special
from scipy import special


class ChangeUpdate(NamedTuple):
    concentration: jax.Array
    valid: jax.Array


def _log_rising(a, c, xp, functions):
    """log Γ(a+c)/Γ(a), avoiding subtraction of log Γ at saturated counts."""

    # Subtract Stirling expansions algebraically before either log Gamma is
    # rounded. At a >= 16 the first omitted Bernoulli term is < 1e-17.
    def remainder(x):
        inv = 1 / x
        z = inv * inv
        polynomial = -691 / 360360
        for coefficient in (1 / 1188, -1 / 1680, 1 / 1260, -1 / 360, 1 / 12):
            polynomial = coefficient + z * polynomial
        return inv * polynomial

    x = xp.maximum(a, 16)
    large = (
        c * xp.log(x)
        + ((x + c - 0.5) * xp.log1p(c / x) - c)
        + (remainder(x + c) - remainder(x))
    )
    ratio = xp.where(a >= 16, large, functions.gammaln(a + c) - functions.gammaln(a))
    return xp.where(c == 0, 0, xp.where(c == 1, xp.log(a), ratio))


def _log_ratio(reset, retain, counts, xp, functions):
    """Log Bayes factor per gamma ratio; cancel a common large count first."""
    lo = xp.minimum(reset, retain)
    delta = xp.abs(reset - retain)
    direct = _log_rising(reset, counts, xp, functions) - _log_rising(
        retain, counts, xp, functions
    )
    shifted = _log_rising(lo + counts, delta, xp, functions) - _log_rising(
        lo, delta, xp, functions
    )
    return xp.where(
        counts > xp.maximum(reset, retain), xp.sign(reset - retain) * shifted, direct
    )


def _trace(mean, xp):
    # The complement of the dominant outcome can round out of 1 - mean.
    dominant = xp.arange(mean.shape[-1]) == xp.argmax(mean, axis=-1, keepdims=True)
    other = xp.where(dominant, 0, mean)
    return xp.sum(other * (1 - other), axis=-1) + xp.max(mean, axis=-1) * xp.sum(
        other, axis=-1
    )


def _classify(values):
    """Identify bitwise zeros and positive finite normals, even with float32 FTZ."""
    dtype = np.dtype(values.dtype)
    uint = np.uint32 if dtype == np.float32 else np.uint64
    bits = values.view(uint)
    magnitude = np.asarray(np.iinfo(uint).max >> 1, dtype=uint)
    minimum = np.asarray(np.finfo(dtype).tiny, dtype=dtype).view(uint)
    maximum = np.asarray(np.finfo(dtype).max, dtype=dtype).view(uint)
    # Use unsigned comparisons for both classes. Mixing a floating comparison
    # with the zero-bit test lets CPU lowering fold it back to a flushed test.
    return (bits & magnitude) == 0, (bits >= minimum) & (bits <= maximum)


def _project(prior, carry, counts, allowed, h, xp, functions):
    tiny = xp.finfo(carry.dtype).tiny
    branches = xp.stack((carry, prior), axis=-2)
    zero, normal = _classify(branches)
    empty, positive = _classify(counts)
    zero_hazard, positive_hazard = _classify(h)
    valid = (
        xp.any(allowed, axis=-1)
        & xp.all(xp.where(allowed[..., None, :], normal, zero), axis=(-2, -1))
        & xp.all(empty | (allowed & positive), axis=-1)
        & (zero_hazard | positive_hazard)
        & (h <= 1)
    )
    updated = branches + counts[..., None, :]
    branch_totals = xp.sum(branches, axis=-1)
    totals = xp.sum(updated, axis=-1)
    valid &= xp.all(xp.isfinite(totals), axis=-1)
    valid &= xp.all(
        xp.where(
            allowed[..., None, :], branches / branch_totals[..., None] >= tiny, True
        ),
        axis=(-2, -1),
    )
    means = updated / totals[..., None]
    safe = xp.where(allowed[..., None, :], branches, 1)
    log_bf = xp.sum(
        _log_ratio(safe[..., 1, :], safe[..., 0, :], counts, xp, functions), axis=-1
    ) - _log_ratio(
        branch_totals[..., 1],
        branch_totals[..., 0],
        xp.sum(counts, axis=-1),
        xp,
        functions,
    )
    log_odds = xp.log(h) - xp.log1p(-h) + log_bf
    # Separate logistic tails preserve a rare branch even when 1 - w rounds to zero.
    weights = xp.stack((functions.expit(-log_odds), functions.expit(log_odds)), axis=-1)
    mean = xp.sum(weights[..., None] * means, axis=-2)
    traces = _trace(means, xp)
    dominant = xp.arange(carry.shape[-1]) == xp.argmax(mean, axis=-1, keepdims=True)
    contrast = means[..., 1, :] - means[..., 0, :]
    contrast = xp.where(
        dominant,
        -xp.sum(xp.where(dominant, 0, contrast), axis=-1, keepdims=True),
        contrast,
    )
    within = traces / (1 + totals)
    variance = xp.sum(weights * within, axis=-1) + xp.prod(weights, axis=-1) * xp.sum(
        xp.square(contrast), axis=-1
    )
    scale = xp.sum(weights * traces * (totals / (1 + totals)), axis=-1) / variance
    result = mean * scale[..., None]
    # At the hazard endpoints or identical updated laws, no projection is needed.
    result = xp.where(
        xp.all(carry == prior, axis=-1, keepdims=True), updated[..., 0, :], result
    )
    result = xp.where(xp.sum(allowed, axis=-1, keepdims=True) == 1, prior, result)
    result = xp.where(
        h == 0, updated[..., 0, :], xp.where(h == 1, updated[..., 1, :], result)
    )
    result_total = xp.sum(result, axis=-1, keepdims=True)
    valid &= (
        xp.all(xp.isfinite(result), axis=-1)
        & xp.isfinite(result_total[..., 0])
        & xp.all(
            xp.where(
                allowed, (result >= tiny) & (result / result_total >= tiny), result == 0
            ),
            axis=-1,
        )
    )
    return ChangeUpdate(xp.where(valid[..., None], result, carry), valid)


def host_columns(prior, carry, counts, allowed, hazard):
    """Validated learner arrays with destination outcome last; keep their host dtype."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
        return _project(
            prior,
            carry,
            counts,
            allowed,
            np.asarray(hazard, dtype=carry.dtype),
            np,
            special,
        )


def update_column(prior, carry, counts, allowed, hazard):
    """Return one projected column. Invalid input/output returns the original carry.

    The last dimension of a vmapped call is outcomes. Counts may be fractional
    expected statistics, pooled once across all readers of the same parameter.
    With zero counts the result is the predicted law *before* the next event.
    """
    prior, carry, counts, allowed = map(jnp.asarray, (prior, carry, counts, allowed))
    if (
        carry.ndim != 1
        or carry.size == 0
        or any(x.shape != carry.shape for x in (prior, counts, allowed))
    ):
        raise ValueError("transition columns require nonempty vectors of one shape")
    if (
        carry.dtype not in (jnp.float32, jnp.float64)
        or prior.dtype != carry.dtype
        or counts.dtype != carry.dtype
        or allowed.dtype != jnp.bool_
    ):
        raise TypeError(
            "concentrations and counts need a shared real dtype and a boolean support mask"
        )
    h = jnp.asarray(hazard, dtype=carry.dtype)
    if h.shape != ():
        raise ValueError("the transition-change hazard must be scalar")

    return _project(prior, carry, counts, allowed, h, jnp, jax_special)

"""Frame-local linear regression under uncertain inputs and per-output Gamma noise."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import digamma

from cffg.efrules import LOG2PI
from cffg.mixed_scoring import _array, _matrix_valid


class Regression(NamedTuple):
    """K independent transforms. Precision is for the input columns.

    Coefficients have shape [K,O,D]. Given a channel precision tau, its
    coefficient row has covariance inv(precision)/tau. Each tau has a Gamma
    prior with the supplied shape and rate [K,O].
    """

    mean: jax.Array
    precision: jax.Array
    shape: jax.Array
    rate: jax.Array


class Frame(NamedTuple):
    """Pixels [N,O] and one uncertain design vector per slot [K,D]."""

    pixels: jax.Array
    design: jax.Array
    covariance: jax.Array
    active: jax.Array


def _valid_model(model):
    return (
        jnp.all(jnp.isfinite(model.mean))
        & _matrix_valid(model.precision, strict=True)
        & jnp.all(jnp.isfinite(model.shape) & (model.shape > 0))
        & jnp.all(jnp.isfinite(model.rate) & (model.rate > 0))
    )


def _inputs(model, frame):
    if model.mean.ndim != 3 or model.mean.dtype not in (jnp.float32, jnp.float64):
        raise ValueError("mean requires [slots,outputs,inputs] in float32 or float64")
    dtype = model.mean.dtype
    if jax.dtypes.canonicalize_dtype(dtype) != dtype:
        raise ValueError("the requested model dtype is disabled in JAX")
    k, o, d = model.mean.shape
    if min(k, o, d) < 1 or frame.pixels.ndim != 2:
        raise ValueError("invalid model dimensions or pixel rank")
    _array("precision", model.precision, (k, d, d), dtype)
    _array("shape", model.shape, (k, o), dtype)
    _array("rate", model.rate, (k, o), dtype)
    _array("pixels", frame.pixels, (len(frame.pixels), o), dtype)
    _array("design", frame.design, (k, d), dtype)
    _array("covariance", frame.covariance, (k, d, d), dtype)
    _array("active", frame.active, (k,), jnp.bool_)
    return (
        _valid_model(model)
        & _matrix_valid(frame.covariance, strict=False)
        & jnp.all(jnp.isfinite(frame.design))
        & jnp.any(frame.active)
    )


def _root(matrix, *, positive):
    """Factor in unit-diagonal coordinates, preserving deterministic input axes."""
    scale = jnp.sqrt(jnp.diagonal(matrix, axis1=-2, axis2=-1))
    safe = jnp.where(scale > 0, scale, 1)
    corr = matrix / safe[:, :, None] / safe[:, None, :]
    if positive:
        return scale[:, :, None] * jnp.linalg.cholesky(corr)
    values, vectors = jnp.linalg.eigh(corr)
    return scale[:, :, None] * vectors * jnp.sqrt(jnp.maximum(values, 0))[:, None, :]


def _dot(expression, *args):
    return jnp.einsum(expression, *args, precision=jax.lax.Precision.HIGHEST)


def score_pixels(model: Regression, frame: Frame):
    """Return absolute expected log likelihoods [N,K] with -inf for unused slots."""
    valid = _inputs(model, frame) & jnp.all(jnp.isfinite(frame.pixels))
    root = _root(model.precision, positive=True)
    noise = _root(frame.covariance, positive=False)
    prediction = _dot("kod,kd->ko", model.mean, frame.design)
    response_noise = _dot("kod,kdj->koj", model.mean, noise)
    residual = frame.pixels[:, None, :] - prediction
    spread = jnp.sum(response_noise**2, axis=-1)
    design = jnp.concatenate((frame.design[:, :, None], noise), axis=-1)
    leverage = jnp.sum(solve_triangular(root, design, lower=True) ** 2, axis=(-2, -1))
    rate_ratio = model.shape / model.rate
    terms = (
        digamma(model.shape)
        - jnp.log(model.rate)
        - LOG2PI
        - rate_ratio * (residual**2 + spread)
        - leverage[:, None]
    )
    scores = 0.5 * jnp.sum(terms, axis=-1)
    valid &= jnp.all(~frame.active | jnp.isfinite(scores))
    return jnp.where(valid & frame.active, scores, -jnp.inf), valid


def fit_frame(prior: Regression, frame: Frame, weights):
    """Fit one frame from a fixed prior. All outputs of a pixel share its weights.

    Rows may have unassigned probability. A pixel with no assigned mass
    contributes nothing. Invalid data or an improper posterior preserves the
    full prior and returns False. Repeated fitting of the same frame uses the
    original prior and its latest pixel responsibilities.
    """
    valid = _inputs(prior, frame)
    k = prior.mean.shape[0]
    _array("weights", weights, (len(frame.pixels), k), prior.mean.dtype)
    tolerance = (8 + k.bit_length()) * jnp.finfo(weights.dtype).eps
    valid &= (
        jnp.all(jnp.isfinite(weights) & (weights >= 0))
        & jnp.all(frame.active | (weights == 0))
        & jnp.all(weights.sum(axis=1) <= 1 + tolerance)
    )
    used = jnp.any(weights > 0, axis=1)
    pixels = jnp.where(used[:, None], frame.pixels, 0)
    valid &= jnp.all(jnp.isfinite(pixels))

    def fit():
        mass = weights.sum(axis=0)
        cov_root = _root(frame.covariance, positive=False)
        second = frame.covariance + frame.design[:, :, None] * frame.design[:, None, :]
        precision = prior.precision + mass[:, None, None] * second
        root = _root(precision, positive=True)
        prediction = _dot("kod,kd->ko", prior.mean, frame.design)
        delta = jnp.where(weights[:, :, None] > 0, pixels[:, None, :] - prediction, 0)
        shift_sum = _dot("nk,nko->ko", weights, delta)
        rhs = shift_sum[:, :, None] * frame.design[:, None, :] - mass[:, None, None] * _dot(
            "kod,kdj->koj", prior.mean, frame.covariance
        )
        y = solve_triangular(root, rhs.swapaxes(-1, -2), lower=True)
        shift = solve_triangular(root, y, lower=True, trans="T").swapaxes(-1, -2)
        mean = prior.mean + shift
        residual = delta - _dot("kod,kd->ko", shift, frame.design)
        fitted = _dot("nk,nko->ko", weights, residual**2)
        input_noise = jnp.sum(_dot("kod,kdj->koj", mean, cov_root) ** 2, axis=-1)
        penalty = jnp.sum(
            _dot("kod,kdj->koj", shift, _root(prior.precision, positive=True)) ** 2, axis=-1
        )
        rate = prior.rate + 0.5 * (fitted + mass[:, None] * input_noise + penalty)
        updated = Regression(mean, precision, prior.shape + mass[:, None] * 0.5, rate)

        def keep_empty(old, new):
            mask = (mass > 0).reshape((k,) + (1,) * (old.ndim - 1))
            return jnp.where(mask, new, old)

        updated = jax.tree.map(keep_empty, prior, updated)
        proper = _valid_model(updated)
        return jax.tree.map(lambda a, b: jnp.where(proper, a, b), updated, prior), proper

    return jax.lax.cond(valid, fit, lambda: (prior, jnp.asarray(False)))

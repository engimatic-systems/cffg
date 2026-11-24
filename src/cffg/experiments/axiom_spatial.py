"""Spatial variational coordinates of the AXIOM object model for one image."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

from cffg.frame_regression import Frame, Regression, _dot, _root, score_pixels
from cffg.mixed_scoring import _array, _dirichlet_log, _matrix_valid


class Positions(NamedTuple):
    """One Gaussian per slot; the final bias coordinate is implicit."""

    mean: jax.Array
    cov: jax.Array


class SpatialReadout(NamedTuple):
    positions: Positions
    assignment: jax.Array
    best_log_likelihood: jax.Array
    occupancy: jax.Array
    valid: jax.Array


def appearance_frame(pixels, positions, active):
    ones = jnp.ones_like(positions.mean[:, :1])
    design = jnp.concatenate((positions.mean, ones), axis=-1)
    noise = jnp.pad(positions.cov, ((0, 0), (0, 1), (0, 1)))
    return Frame(pixels, design, noise, active)


def _inverse(precision):
    root = _root(precision, positive=True)
    eye = jnp.broadcast_to(jnp.eye(root.shape[-1], dtype=root.dtype), root.shape)
    raw = cho_solve((root, True), eye)
    return 0.5 * raw + 0.5 * raw.swapaxes(-1, -2)


def _position_terms(model, pixels, prior):
    """Fixed quadratic and linear terms around the frame prior's center."""
    k, o, d = model.mean.shape
    design_cov = _inverse(model.precision)
    inv_prior = _inverse(prior.cov)
    tau = model.shape / model.rate
    slope = model.mean[:, :, :-1]
    slope_tau = slope * tau[:, :, None]
    curvature = o * design_cov[:, :-1, :-1] + _dot("koi,koj->kij", slope_tau, slope)
    residual = pixels[:, None, :] - (_dot("kod,kd->ko", slope, prior.mean) + model.mean[:, :, -1])
    cov_derivative = o * (
        _dot("kij,kj->ki", design_cov[:, :-1, :-1], prior.mean) + design_cov[:, :-1, -1]
    )
    return inv_prior, curvature, residual, slope_tau, cov_derivative


def _infer_positions(terms, weights, prior, previous):
    inv_prior, curvature, residual, slope_tau, cov_derivative = terms
    mass = weights.sum(axis=0)
    center_error = _dot("nk,nko->ko", weights, residual)
    gradient = _dot("ko,koi->ki", center_error, slope_tau) - mass[:, None] * cov_derivative
    precision = inv_prior + mass[:, None, None] * curvature
    cov = _inverse(precision)
    mean = prior.mean + _dot("kij,kj->ki", cov, gradient)
    mean = jnp.where((mass > 0)[:, None], mean, previous.mean)
    cov = jnp.where((mass > 0)[:, None, None], cov, previous.cov)
    return Positions(mean, cov), (
        jnp.all(jnp.isfinite(mean))
        & _matrix_valid(precision, strict=True)
        & _matrix_valid(cov, strict=True)
    )


def infer_slots(
    model: Regression, pixels, prior: Positions, initial: Positions, counts, active, *, sweeps=2
) -> SpatialReadout:
    """Run a fixed number of assignment-then-position steps for one frame.

    'prior' is the same Gaussian factor on every sweep; 'initial' is the
    starting belief, for example a prediction from the previous frame.
    Counts are Dirichlet concentrations of the slot mixture. The returned
    absolute best score excludes the mixing contribution used for assignment.
    """
    k, o, d = model.mean.shape
    dtype = model.mean.dtype
    if type(sweeps) is not int or sweeps < 1 or d < 2 or len(pixels) < 1:
        raise ValueError("sweeps, token count and spatial dimension must be positive")
    for pos in (prior, initial):
        _array("mean", pos.mean, (k, d - 1), dtype)
        _array("cov", pos.cov, (k, d - 1, d - 1), dtype)
    _array("counts", counts, (k,), dtype)
    _array("active", active, (k,), jnp.bool_)
    valid = (
        _matrix_valid(prior.cov, strict=True)
        & jnp.all(jnp.isfinite(prior.mean))
        & _matrix_valid(initial.cov, strict=True)
        & jnp.all(jnp.isfinite(initial.mean))
        & jnp.any(active)
        & jnp.all(jnp.isfinite(counts) & (counts > 0))
    )
    terms = _position_terms(model, pixels, prior)
    log_prior = _dirichlet_log(counts, active)
    log_prior = jnp.where(active, log_prior, -jnp.inf)
    log_prior -= jnp.max(log_prior)

    def one(_, carry):
        pos, _, _, ok = carry
        ell, scored = score_pixels(model, appearance_frame(pixels, pos, active))
        best = jnp.max(ell, axis=1)
        w = jax.nn.softmax((ell - best[:, None]) + log_prior, axis=-1)
        pos, proper = _infer_positions(terms, w, prior, pos)
        return pos, w, best, ok & scored & proper & jnp.all(jnp.isfinite(w))

    positions, weights, best, valid = jax.lax.fori_loop(
        0,
        sweeps,
        one,
        (initial, jnp.zeros((len(pixels), k), dtype), jnp.zeros(len(pixels), dtype), valid),
    )
    occupancy = jnp.bincount(weights.argmax(axis=1), length=k)
    safe_pos = jax.tree.map(lambda x, y: jnp.where(valid, x, y), positions, initial)
    return SpatialReadout(
        safe_pos,
        jnp.where(valid, weights, 0),
        jnp.where(valid, best, -jnp.inf),
        jnp.where(valid, occupancy, 0),
        valid,
    )

"""Initialize and grow the single-layer AXIOM spatial model on RGB observations."""

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from cffg.experiments.axiom_spatial import (
    Positions,
    SpatialReadout,
    _infer_positions,
    _position_terms,
    appearance_frame,
    infer_slots,
)
from cffg.frame_regression import Regression, _valid_model, fit_frame


class SpatialModel(NamedTuple):
    appearance_prior: Regression
    appearance: Regression
    prior_counts: jax.Array
    counts: jax.Array
    position_prior: Positions


class FrameStatus(IntEnum):
    EXPLAINED = 0
    CAPACITY = 1
    GROWTH_BUDGET = 2
    INVALID = 3


class ObservedSlots(NamedTuple):
    model: SpatialModel
    readout: SpatialReadout
    # Pairs (pixel, slot); -1 in entries past the end of the trace.
    growth: jax.Array
    births: jax.Array
    status: jax.Array


def frame_tokens(rgb, dtype=jnp.float32):
    """Encode [height,width,3] uint8 RGB as five normalized row-major channels."""
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != jnp.uint8 or min(rgb.shape[:2]) < 1:
        raise ValueError("expected a nonempty uint8 RGB frame")
    if dtype not in (jnp.float32, jnp.float64) or jax.dtypes.canonicalize_dtype(dtype) != dtype:
        raise ValueError("requested token precision unavailable")
    h, w = rgb.shape[:2]
    y, x = jnp.indices((h, w), dtype=dtype)
    xy = jnp.stack((x / (w / 2) - 1, y / (h / 2) - 1), axis=-1)
    return jnp.concatenate((xy, rgb.astype(dtype) / 128 - 1), axis=-1).reshape((-1, 5))


def initial_model(key, *, slots=32, dtype=jnp.float32):
    """Initialize the released single-layer template from its appearance seed and fixed position seed."""
    if (
        type(slots) is not int
        or slots < 1
        or dtype not in (jnp.float32, jnp.float64)
        or jax.dtypes.canonicalize_dtype(dtype) != dtype
    ):
        raise ValueError("positive slot count and enabled floating dtype required")
    # The default template maps x/y through the spatial diagonal and color through bias.
    m = jnp.broadcast_to(
        jnp.concatenate(
            (jnp.eye(2, 3, dtype=dtype), jnp.tile(jnp.array([0, 0, 1], dtype), (3, 1)))
        ),
        (slots, 5, 3),
    )
    p = jnp.broadcast_to(jnp.diag(jnp.array([100.0, 100.0, 0.001], dtype)), (slots, 3, 3))
    shape = jnp.full((slots, 5), 11.0, dtype)
    rate = jnp.broadcast_to(jnp.array([0.075, 0.075, 0.75, 0.75, 0.75], dtype) ** 2, (slots, 5))
    prior = Regression(m, p, shape, rate)
    # The layer draws a mask before appearance, including for its fixed fifth template.
    _, appearance_key = jax.random.split(key)
    appearance_key, _ = jax.random.split(appearance_key)
    appearance_key, transform_key = jax.random.split(appearance_key)
    _, noise_key = jax.random.split(appearance_key)
    pos_key = jax.random.key(0)
    appearance = Regression(
        m * (1 + jax.random.uniform(transform_key, m.shape, dtype, minval=1, maxval=2)),
        jnp.tile(jnp.eye(3, dtype=dtype), (slots, 1, 1)),
        shape * jax.random.uniform(noise_key, shape.shape, dtype, minval=1, maxval=3),
        rate,
    )
    positions = Positions(
        jax.random.normal(pos_key, (slots, 2), dtype) / jnp.sqrt(jnp.asarray(2.0, dtype)),
        jnp.tile(jnp.eye(2, dtype=dtype), (slots, 1, 1)),
    )
    counts = jnp.ones(slots, dtype)
    return SpatialModel(prior, appearance, counts, counts, positions)


def slot_descriptors(model: SpatialModel, positions: Positions):
    """Return the five-channel variational forward means and covariances.

    The released tracker uses inverse expected noise precision. Uncertainty
    in the position enters the appearance fit and has already been used there.
    """
    reg = model.appearance
    means = (
        jnp.einsum(
            "koi,ki->ko", reg.mean[:, :, :-1], positions.mean, precision=jax.lax.Precision.HIGHEST
        )
        + reg.mean[:, :, -1]
    )
    covariance = jax.vmap(jnp.diag)(reg.rate / reg.shape)
    return means, covariance


def _fit(model, pixels, positions, w, new_slot=None):
    k = len(model.counts)
    available = jnp.ones(k, bool)
    candidate, valid = fit_frame(
        model.appearance_prior, appearance_frame(pixels, positions, available), w
    )
    mass = w.sum(axis=0)
    write = mass > 0
    if new_slot is not None:
        write = jnp.arange(k) == new_slot
        mass = jnp.where(write, 1000.0, jnp.minimum(mass, 100.0))
        candidate = candidate._replace(
            precision=candidate.precision.at[:, -1, -1].set(1000),
            rate=candidate.rate.at[:, :2].multiply(2),
        )

    def select(before, after):
        return jnp.where(write.reshape((k,) + (1,) * (before.ndim - 1)), after, before)

    posterior = jax.tree.map(select, model.appearance, candidate)
    counts = jnp.where(write, model.prior_counts + mass, model.counts)
    return model._replace(appearance=posterior, counts=counts), valid & _valid_model(
        posterior
    ) & jnp.all(jnp.isfinite(counts))


def _forced_fit(model, pixels, forced, new_slot=None):
    available = jnp.ones_like(model.counts, bool)
    posprior = model.position_prior
    terms = _position_terms(model.appearance, pixels, posprior)
    pos, valid = _infer_positions(terms, forced, posprior, posprior, keep_empty=False)
    model, ok = _fit(model, pixels, pos, forced, new_slot)
    valid &= ok
    # The first refinement uses uniform mixing; the next uses the fitted counts.
    result = infer_slots(
        model.appearance,
        pixels,
        posprior,
        pos,
        jnp.ones_like(model.counts),
        available,
        sweeps=1,
        keep_empty=False,
    )
    model, ok = _fit(model, pixels, result.positions, result.assignment)
    valid &= result.valid & ok
    result = infer_slots(
        model.appearance,
        pixels,
        posprior,
        result.positions,
        model.counts,
        available,
        sweeps=1,
        keep_empty=False,
    )
    model, ok = _fit(model, pixels, result.positions, result.assignment)
    return model, result._replace(valid=valid & result.valid & ok)


def initialize_frame(model, pixels):
    """Fit the first frame with every token initially assigned to slot zero."""
    if pixels.ndim != 2 or pixels.shape[1] != 5 or not len(pixels):
        raise ValueError("expected a nonempty five-channel pixel frame")
    forced = jnp.zeros((len(pixels), len(model.counts)), pixels.dtype).at[:, 0].set(1)
    learned, result = _forced_fit(model, pixels, forced)
    safe_model = jax.tree.map(lambda x, y: jnp.where(result.valid, x, y), learned, model)
    return safe_model, result


def observe_frame(model, pixels, predicted, *, threshold=5.7, max_births=20, sweeps=2):
    """Observe a subsequent frame; preserve the model atomically on a numerical failure."""
    if type(max_births) is not int or max_births < 0:
        raise ValueError("max_births must be a nonnegative integer")
    k = len(model.counts)
    available = jnp.ones(k, bool)
    readout = infer_slots(
        model.appearance,
        pixels,
        model.position_prior,
        predicted,
        model.counts,
        available,
        sweeps=sweeps,
    )
    readout = readout._replace(valid=readout.valid & jnp.isfinite(threshold))
    trace = jnp.full((max_births, 2), -1, jnp.int32)
    state = (model, readout, trace, jnp.int32(0))

    def need_birth(s):
        _, r, _, n = s
        return (
            r.valid
            & (n < max_births)
            & (jnp.min(r.best_log_likelihood) < threshold)
            & jnp.any(r.occupancy == 0)
        )

    def grow(s):
        m, r, history, n = s
        token = jnp.argmin(r.best_log_likelihood).astype(jnp.int32)
        slot = jnp.argmax(r.occupancy == 0).astype(jnp.int32)
        forced = r.assignment.at[token].set(jax.nn.one_hot(slot, k, dtype=pixels.dtype))
        m, r = _forced_fit(m, pixels, forced, slot)
        return m, r, history.at[n].set(jnp.stack((token, slot))), n + 1

    if max_births:
        state = jax.lax.while_loop(need_birth, grow, state)
    learned, result, trace, births = state
    poor = jnp.min(result.best_log_likelihood) < threshold
    status = jnp.select(
        [~result.valid, ~poor, jnp.all(result.occupancy > 0)],
        [int(FrameStatus.INVALID), int(FrameStatus.EXPLAINED), int(FrameStatus.CAPACITY)],
        default=int(FrameStatus.GROWTH_BUDGET),
    )
    safe_model = jax.tree.map(lambda x, y: jnp.where(result.valid, x, y), learned, model)
    return ObservedSlots(safe_model, result, trace, births, status)

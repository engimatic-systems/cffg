"""Temporal slot descriptors and causal learner masks for a single-layer AXIOM agent."""

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp

from cffg.experiments.axiom_motion import MotionSettings, observed_state
from cffg.mixed_scoring import _array, _matrix_valid


@dataclass(frozen=True)
class TrackingSettings:
    motion: MotionSettings = field(default_factory=MotionSettings)
    min_motion: float = 0.003
    min_use: float = 0.02
    max_unused_steps: int = 10
    min_track_steps: int = 1

    def __post_init__(self):
        if not math.isfinite(self.min_motion) or not math.isfinite(self.min_use):
            raise ValueError("tracking thresholds must be finite")
        if self.min_motion < 0 or not 0 <= self.min_use <= 1:
            raise ValueError("invalid motion or use threshold")
        if any(
            type(i) is not int or not 0 < i <= 2**31 - 1
            for i in (self.min_track_steps, self.max_unused_steps)
        ):
            raise ValueError("tracking horizons must be positive int32 counts")


class DecodedSlots(NamedTuple):
    """Normalized [x,y,r,g,b] moments, hard pixel winners and summed soft assignments."""

    mean: jax.Array  # [slots,5]
    covariance: jax.Array  # [slots,5,5]
    occupied: jax.Array  # [slots] bool, e.g. spatial_readout.occupancy > 0
    mass: jax.Array  # [slots] sum of spatial_readout.assignment over pixels
    valid: jax.Array  # scalar bool, supplied by the spatial owner

    @classmethod
    def from_readout(cls, mean, covariance, readout, *, accepted=True):
        """Join slot_descriptors(model,positions) to the hard and soft statistics of that frame."""
        accepted = jnp.asarray(accepted)
        _array("accepted", accepted, (), jnp.bool_)
        return cls(
            mean,
            covariance,
            readout.occupancy > 0,
            readout.assignment.sum(axis=0),
            readout.valid & accepted,
        )


class TrackCarry(NamedTuple):
    """Save these arrays together with the tracking settings for exact continuation."""

    scene: jax.Array  # [x,y,unused,vx,vy,vu,width,height,r,g,b]
    covariance: jax.Array  # last decoded covariance for the stored descriptor
    use: jax.Array
    moving: jax.Array
    consecutive: jax.Array
    tracked: jax.Array
    foreground: jax.Array
    initialized: jax.Array


class TrackFrame(NamedTuple):
    state: TrackCarry
    identity_eligible: jax.Array  # applies to the PREVIOUS scene's appearance
    dynamics_eligible: jax.Array  # applies to the real transition ending in state
    valid: jax.Array


def _frame(frame):
    mean = frame.mean
    if mean.ndim != 2 or mean.shape[1] != 5 or not len(mean):
        raise ValueError("expected one or more [x,y,r,g,b] descriptors")
    if (
        mean.dtype not in (jnp.float32, jnp.float64)
        or jax.dtypes.canonicalize_dtype(mean.dtype) != mean.dtype
    ):
        raise TypeError("expected a supported float32 or float64 descriptor")
    k = len(mean)
    _array("covariance", frame.covariance, (k, 5, 5), mean.dtype)
    _array("occupied", frame.occupied, (k,), jnp.bool_)
    _array("mass", frame.mass, (k,), mean.dtype)
    flag = jnp.asarray(frame.valid)
    _array("valid", flag, (), jnp.bool_)
    mass = frame.mass
    # A hard winner must have positive responsibility on at least one pixel.
    valid = flag & frame.occupied.any() & jnp.isfinite(mass).all() & (mass >= 0).all()
    return valid & (~frame.occupied | (mass > 0)).all(), mass < mass.max()


def _appearance(mean, covariance):
    # This offset is the released model's descriptor convention in normalized spatial units.
    diagonal = jnp.diagonal(covariance, axis1=-2, axis2=-1)[:, :2]
    size = 3 * jnp.sqrt(diagonal + 1e-8)
    return jnp.concatenate((size, mean[:, 2:]), axis=-1)


def initial_tracks(frame: DecodedSlots, *, settings=TrackingSettings()):
    """Initialize from a decoded first frame. Every slot supplies finite initial moments."""
    valid, foreground = _frame(frame)
    mean, cov = frame.mean, frame.covariance
    valid &= jnp.isfinite(mean).all() & _matrix_valid(cov, strict=False)
    unused = jnp.where(frame.occupied, 0, settings.motion.unused_increment).astype(mean.dtype)
    zeros = jnp.zeros_like(unused)
    off = jnp.zeros_like(frame.occupied)
    scene = jnp.column_stack((mean[:, :2], unused, jnp.zeros_like(mean[:, :3]), _appearance(mean, cov)))
    valid &= jnp.isfinite(scene).all()
    state = TrackCarry(
        jnp.where(valid, scene, 0),
        jnp.where(valid, cov, 0),
        jnp.where(valid, frame.occupied.astype(mean.dtype) * 0.01, 0),
        zeros,
        jnp.zeros(unused.shape, jnp.int32),
        off,
        valid & foreground,
        valid,
    )
    return TrackFrame(state, off, off, valid)


def track_frame(previous: TrackCarry, frame: DecodedSlots, *, update=True, settings=TrackingSettings()):
    """Step one observation. With update=False both write masks close, including on episode reset.

    Use previous.scene for iMM evidence and for the input end of a real dynamics transition.
    A missing slot retains its last appearance and covariance and ages in the motion coordinates.
    """
    valid, foreground = _frame(frame)
    k, dtype = len(frame.mean), frame.mean.dtype
    _array("previous scene", previous.scene, (k, 11), dtype)
    _array("previous covariance", previous.covariance, (k, 5, 5), dtype)
    for name in ("use", "moving"):
        _array(name, getattr(previous, name), (k,), dtype)
    _array("consecutive", previous.consecutive, (k,), jnp.int32)
    for name in ("foreground", "tracked"):
        _array(name, getattr(previous, name), (k,), jnp.bool_)
    _array("initialized", previous.initialized, (), jnp.bool_)
    update = jnp.asarray(update)
    _array("update", update, (), jnp.bool_)

    occupied = frame.occupied
    before = previous.scene
    cov = jnp.where(occupied[:, None, None], frame.covariance, previous.covariance)
    # An absent slot carries no measurement. Its upstream placeholders may be NaN.
    mean = jnp.where(occupied[:, None], frame.mean, 0)
    valid &= previous.initialized & jnp.isfinite(mean).all() & _matrix_valid(cov, strict=False)
    motion = observed_state(before[:, :6], mean[:, :2], occupied, settings=settings.motion)
    appearance = jnp.where(occupied[:, None], _appearance(mean, cov), before[:, 6:])
    scene = jnp.concatenate((motion, appearance), axis=1)
    delta = motion[:, :2] - before[:, :2]
    # The activity average includes a teleport's measured displacement, even when stored velocity is zero.
    clipped = jnp.where(jnp.abs(delta) < settings.motion.clip, 0, delta)
    speed = jnp.hypot(clipped[:, 0], clipped[:, 1])
    use = 0.99 * previous.use + 0.01 * occupied.astype(dtype)
    moving = jnp.where(occupied, 0.99 * previous.moving + 0.01 * speed, previous.moving)
    eligible = (
        (moving > settings.min_motion)
        & (use > settings.min_use)
        & (motion[:, 2] < settings.max_unused_steps * settings.motion.unused_increment)
    )
    # Keep long-running counts in range even for uninterrupted tracks.
    limit = jnp.iinfo(jnp.int32).max
    consecutive = jnp.where(eligible, jnp.minimum(previous.consecutive, limit - 1) + 1, 0)
    tracked = (consecutive >= settings.min_track_steps).at[0].set(False)
    valid &= jnp.isfinite(scene).all() & jnp.isfinite(moving).all() & jnp.isfinite(use).all()
    candidate = TrackCarry(
        scene, cov, use, moving, consecutive, tracked, foreground, previous.initialized
    )
    state = jax.tree.map(lambda new, old: jnp.where(valid, new, old), candidate, previous)
    writes = valid & update
    return TrackFrame(state, writes & (before[:, 2] == 0) & foreground, writes & tracked, valid)

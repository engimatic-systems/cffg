"""Acquire affine motion experts from causal pairs of six-coordinate slot states.

Reference: VersesTech/axiom b99f625, models/tmm.py (velocity model) and
infer.py (observation preparation and ordered switch inference). One time
unit is one observed frame. The state is (x, y, unused, vx, vy, v_unused).
All matrices are frozen after insertion.
"""

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class MotionSettings:
    variance: float = 2.
    logp_threshold: float = -1e-5
    teleport_distance: float = .2
    clip: float = .00075
    unused_increment: float = .05

    def __post_init__(self):
        if not all(map(math.isfinite, (self.variance, self.logp_threshold, self.teleport_distance,
                                        self.clip, self.unused_increment))):
            raise ValueError("motion settings must be finite")
        if min(self.variance, self.teleport_distance, self.unused_increment) <= 0 or self.clip < 0:
            raise ValueError("invalid motion variance, distance, increment or clip")


class MotionBank(NamedTuple):
    affine: jax.Array      # [capacity,6,7], bias in final column
    occupied: jax.Array    # [capacity]


class MotionStatus(IntEnum):
    SKIPPED = 0
    MATCHED = 1
    VELOCITY_BIRTH = 2
    TELEPORT_BIRTH = 3
    FULL = 4
    EMPTY_BIRTH = 5
    INVALID = 6
    ROLLED_BACK = 7


class MotionFrame(NamedTuple):
    bank: MotionBank
    previous: jax.Array
    current: jax.Array
    action: jax.Array     # real action that led from previous to current
    switch: jax.Array    # first maximum after acquisition for each tracked slot
    born: jax.Array      # allocated row or -1
    status: jax.Array
    log_likelihood: jax.Array  # [slots,capacity], scored at that slot's turn
    valid: jax.Array


def initial_motion(*, capacity=500, dtype=jnp.float32, settings=MotionSettings()):
    """Seed drift, remain unused, become unused and stop, in release index order."""
    if capacity < 4 or jnp.dtype(dtype) not in (jnp.dtype("float32"), jnp.dtype("float64")):
        raise ValueError("motion bank requires at least four slots and float32 or float64")
    indices = jnp.arange(2)
    # Every seed preserves horizontal and vertical position.
    seeds = jnp.zeros((4, 6, 7), dtype=dtype).at[:, indices, indices].set(1)
    seeds = seeds.at[0, indices, indices + 3].set(1)
    seeds = seeds.at[:2, indices + 3, indices + 3].set(1)
    seeds = seeds.at[1, 2, 2].set(1)
    seeds = seeds.at[1:3, 2, -1].set(settings.unused_increment)
    return MotionBank(jnp.zeros((capacity, 6, 7), dtype).at[:4].set(seeds),
                      jnp.arange(capacity) < 4)


def observed_state(previous, position, visible, *, settings=MotionSettings()):
    """Derive the current six coordinates from a slot's actual observation.

    Missing slots keep their last position and increment the unused counter.
    Velocity is the clipped displacement. At the teleport distance or above,
    the measured velocity resets. The unused-counter velocity is always zero.
    Leading batch axes are allowed; visibility and eligibility to train are
    separate inputs.
    """
    previous, position, visible = map(jnp.asarray, (previous, position, visible))
    if (previous.shape != (*visible.shape, 6) or position.shape != (*visible.shape, 2)
            or visible.dtype != jnp.bool_ or position.dtype != previous.dtype
            or previous.dtype not in (jnp.float32, jnp.float64)):
        raise ValueError("expected compatible six-coordinate states, xy positions and visibility")
    xy = jnp.where(visible[..., None], position, previous[..., :2])
    unused = jnp.where(visible, 0., previous[..., 2] + settings.unused_increment)
    delta = xy - previous[..., :2]
    velocity = jnp.where(jnp.abs(delta) < settings.clip, 0., delta)
    moving = jnp.linalg.norm(velocity, axis=-1) < settings.teleport_distance
    velocity = jnp.where(moving[..., None], velocity, 0.)
    return jnp.concatenate((xy, unused[..., None], velocity, jnp.zeros_like(unused[..., None])), axis=-1)


def _apply(affine, state):
    return (affine[..., :-1] @ state[..., None])[..., 0] + affine[..., -1]


def motion_step(bank, state, switch):
    """Apply an occupied expert to each state. Invalid selections yield NaNs."""
    state, switch = jnp.asarray(state), jnp.asarray(switch)
    if state.shape != (*switch.shape, 6) or not jnp.issubdtype(switch.dtype, jnp.integer):
        raise ValueError("expected a six-coordinate state per integer switch")
    index = jnp.clip(switch, 0, bank.affine.shape[0] - 1)
    prediction = _apply(bank.affine[index], state)
    valid = (index == switch) & bank.occupied[index] & jnp.isfinite(prediction).all(axis=-1)
    return jnp.where(valid[..., None], prediction, jnp.nan)


def _scores(affine, before, after, variance, normalizer):
    difference = _apply(affine, before) - after
    return normalizer - jnp.sum(difference * difference, axis=-1) / (2 * variance)


def _birth(before, after, settings):
    displacement = after[:2] - before[:2]
    teleport = jnp.linalg.norm(displacement) > settings.teleport_distance
    # For ordinary motion both predicted position and velocity receive the
    # change in velocity. A teleport maps all inputs to the observed target.
    drift = jnp.diag(jnp.array([1, 1, 0, 1, 1, 0], dtype=before.dtype))
    drift = drift.at[:2, 3:5].set(jnp.eye(2, dtype=before.dtype))
    impulse = displacement - before[3:5]
    offset = jnp.zeros(6, before.dtype).at[:2].set(impulse).at[3:5].set(impulse)
    offset = jnp.where(teleport, after.at[2].set(0).at[5].set(0), offset)
    matrix = jnp.column_stack((jnp.where(teleport, 0., drift), offset))
    return jnp.where(jnp.abs(matrix) < settings.clip, 0., matrix), teleport


def acquire_motion(bank, previous, current, action, tracked, *, update=True,
                   settings=MotionSettings()):
    """Process a real frame in slot order; a later slot sees earlier births.

    On a full bank the best existing switch remains available. A skipped slot
    has the release's dummy switch 0 and all-zero score vector. An invalid
    tracked observation or bank rolls back this entire frame and returns -1
    for every switch. Caller-owned masks identify tracked objects. Actions are
    carried to the downstream interaction learner and do not condition tMM.
    """
    affine, occupied = bank = MotionBank(*map(jnp.asarray, bank))
    previous, current, tracked, action, update = map(
        jnp.asarray, (previous, current, tracked, action, update))
    if (affine.shape != (occupied.size, 6, 7) or occupied.ndim != 1 or occupied.size == 0
            or tracked.ndim != 1 or current.shape != previous.shape
            or current.shape != (tracked.size, 6) or action.shape or update.shape):
        raise ValueError("incompatible motion bank, frame, action or update dimensions")
    if (affine.dtype not in (jnp.float32, jnp.float64) or affine.dtype != previous.dtype
            or affine.dtype != current.dtype or occupied.dtype != jnp.bool_
            or tracked.dtype != jnp.bool_ or update.dtype != jnp.bool_
            or not jnp.issubdtype(action.dtype, jnp.integer)):
        raise TypeError("expected matching floating states, boolean masks and an integer action")
    variance = jnp.asarray(settings.variance, affine.dtype)
    normalizer = -3 * jnp.log(2 * jnp.pi * variance)
    threshold = normalizer + .5 * variance * settings.logp_threshold

    def observe(table, pair):
        before, after = pair
        scores = jnp.where(table.occupied, _scores(table.affine, before, after, variance, normalizer),
                           -jnp.inf)
        miss = scores.max() < threshold
        free = ~table.occupied
        row = jnp.argmax(free).astype(jnp.int32)

        def install(_):
            candidate, teleport = _birth(before, after, settings)
            nonzero = (candidate != 0).any()
            written = MotionBank(table.affine.at[row].set(candidate),
                                 table.occupied.at[row].set(nonzero))
            new_score = _scores(candidate, before, after, variance, normalizer)
            scored = scores.at[row].set(jnp.where(nonzero, new_score, -jnp.inf))
            kind = jnp.where(teleport, MotionStatus.TELEPORT_BIRTH, MotionStatus.VELOCITY_BIRTH)
            kind = jnp.where(nonzero, kind, MotionStatus.EMPTY_BIRTH)
            return written, scored, jnp.where(nonzero, row, -1), kind

        def retain(_):
            return (table, scores, jnp.int32(-1),
                    jnp.where(miss, MotionStatus.FULL, MotionStatus.MATCHED))

        new, final_scores, born, status = jax.lax.cond(miss & free.any(), install, retain, None)
        ok = (jnp.isfinite(before).all() & jnp.isfinite(after).all()
              & jnp.all(~new.occupied | jnp.isfinite(final_scores)))
        return new, (jnp.argmax(final_scores).astype(jnp.int32), born,
                     jnp.where(ok, status, MotionStatus.INVALID), final_scores)

    def visit(table, data):
        before, after, enabled = data
        def skip(_):
            return table, (jnp.int32(0), jnp.int32(-1), jnp.asarray(MotionStatus.SKIPPED),
                           jnp.zeros(occupied.shape, affine.dtype))
        return jax.lax.cond(enabled, lambda _: observe(table, (before, after)), skip, None)

    new, (switch, born, status, scores) = jax.lax.scan(
        visit, bank, (previous, current, tracked & update))
    bank_ok = (occupied.any() & jnp.all(~occupied | (jnp.isfinite(affine).all(axis=(1, 2))
                                                      & (affine != 0).any(axis=(1, 2)))))
    inputs_ok = bank_ok & (action >= 0) & jnp.isfinite(threshold)
    valid = inputs_ok & jnp.all(status != MotionStatus.INVALID)
    final = jax.tree.map(lambda old, candidate: jnp.where(valid, candidate, old), bank, new)
    status = jnp.where(valid, status, jnp.where(inputs_ok & (status != MotionStatus.INVALID),
                                                MotionStatus.ROLLED_BACK, MotionStatus.INVALID))
    return MotionFrame(final, previous, current, action, jnp.where(valid, switch, -1),
                       jnp.where(valid, born, -1), status, jnp.where(valid, scores, jnp.nan), valid)

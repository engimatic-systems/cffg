"""Construct rMM evidence at the boundary between real and imagined transitions.

Reference: AXIOM b99f625, models/rmm.py (contact geometry and observation
construction). The recorded scene precedes the applied action. Supplied identity
labels refer to that scene. No actor reads the realized reward or next switch.
"""

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from cffg.mixed_learning import MixedBatch


@dataclass(frozen=True)
class InteractionConfig:
    actions: int
    identities: int = 32
    motions: int = 500
    fixed_radius: bool = False
    train_radius: float = .075
    predict_radius: float = .075
    static_contacts: bool = False
    exclude_background: bool = True
    relative_scale: bool = False

    def __post_init__(self):
        if min(self.actions, self.identities, self.motions) < 1:
            raise ValueError("identity, motion and action cardinalities must be positive")
        if not all(math.isfinite(r) and r >= 0 for r in (self.train_radius, self.predict_radius)):
            raise ValueError("contact radii must be nonnegative and finite")


class InteractionBatch(NamedTuple):
    records: MixedBatch
    neighbor: jax.Array     # -1 for no eligible other object
    valid: jax.Array


def _sampled_overlap(center, radii, index, grid):
    """Test contact on the released 30 x 30 grid of each candidate neighbor."""
    # Positive radii define a nonempty ellipse; collapsed axes have no included points.
    unit = jnp.where(radii[index] > 0, radii[index], 1.)
    relative = (center - center[index])[:, None, :] + radii[:, None, :] * grid
    in_actor = jnp.square(relative / unit).sum(axis=-1) <= 1
    in_other = jnp.square(grid).sum(axis=1) <= 1
    full = (radii > 0).all(axis=1)
    return (in_actor & in_other).any(axis=1) & full & full[index]


def _neighbors(scene, tracked, radius, config):
    count = tracked.size
    ids = jnp.arange(count, dtype=jnp.int32)
    center = scene[:, :2]
    size = scene[:, 6:8]
    if config.fixed_radius:
        axis = jnp.linspace(-1., 1., 30, dtype=scene.dtype)
        grid = jnp.stack(jnp.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
        radii = size * radius
        projected = center + scene[:, 3:5]

    def select(i):
        delta = center[i] - center
        distance = jnp.square(delta).sum(axis=1)
        if config.fixed_radius:
            nearby = (_sampled_overlap(center, radii, i, grid)
                      | _sampled_overlap(projected, radii, i, grid))
            nearby &= scene[:, 2] == 0
            if config.exclude_background:
                nearby &= ids != 0
        else:
            nearby = (jnp.abs(delta) < size + size[i] + radius).all(axis=1)
        nearby &= ids != i
        dynamic = nearby & tracked
        static = nearby & ~tracked
        if config.exclude_background:
            static &= ids != 0
        if config.fixed_radius:
            # For tuned geometry the static fallback is the first intersecting slot.
            static_index = jnp.argmax(static)
        else:
            # 20,000 is the reference squared-distance sentinel for a masked pair.
            dynamic &= distance < 20_000
            static &= distance < 20_000
            static_index = jnp.argmin(jnp.where(static, distance, jnp.inf))
        dynamic_index = jnp.argmin(jnp.where(dynamic, distance, jnp.inf))
        return jnp.where(dynamic.any(), dynamic_index, jnp.where(
            config.static_contacts & static.any(), static_index, -1)).astype(jnp.int32)

    # The ellipse grid is temporary for a single focal slot.
    return jax.lax.map(select, ids)


def interaction_batch(scene, identities, tracked, action, absent_distance, *, config,
                      reward=None, switches=None):
    """Return per-slot point observations for learning or contextual prediction.

    Scene columns start with (x,y,u,vx,vy,vu,width,height). Supply both actual
    outcomes for learning or omit both for prediction. absent_distance is the
    frozen two-vector used when a default-model slot has no neighbor. The
    release draws it as 1.2 + .01 * uniform during tracing; passing it explicitly
    preserves it across checkpoint, compilation and replay.
    """
    scene, identities, tracked, action, absent = map(
        jnp.asarray, (scene, identities, tracked, action, absent_distance))
    if (scene.ndim != 2 or scene.shape[1] < 8 or scene.shape[0] == 0
            or tracked.shape != (scene.shape[0],) or identities.shape != tracked.shape
            or action.shape or absent.shape != (2,)):
        raise ValueError("invalid scene, label, tracking mask, action or absent distance shape")
    if (scene.dtype not in (jnp.float32, jnp.float64) or tracked.dtype != jnp.bool_
            or not jnp.issubdtype(identities.dtype, jnp.integer)
            or not jnp.issubdtype(action.dtype, jnp.integer)):
        raise TypeError("expected floating scene, integer identities and action, boolean mask")
    training = reward is not None
    if training != (switches is not None):
        raise ValueError("provide both realized reward and motion switches, or omit both")
    radius = config.predict_radius if config.fixed_radius and not training else config.train_radius
    neighbor = _neighbors(scene, tracked, radius, config)
    valid = (jnp.isfinite(scene[:, :8]).all() & (scene[:, 6:8] >= 0).all()
             & ((identities >= 0) & (identities < config.identities)).all()
             & (action >= 0) & (action < config.actions))
    if training:
        switches, reward = jnp.asarray(switches), jnp.asarray(reward)
        if (switches.shape != tracked.shape or reward.shape
                or not jnp.issubdtype(switches.dtype, jnp.integer)):
            raise ValueError("expected an integer switch per slot and a scalar reward")
        valid &= ((switches >= 0) & (switches < config.motions)).all()
        valid &= (reward == -1) | (reward == 0) | (reward == 1)
        reward_class = (reward + 1).astype(jnp.int32)
        if config.fixed_radius:
            neighbor = jnp.where(switches == 0, -1, neighbor)
    else:
        switches = jnp.full(tracked.shape, -1, jnp.int32)
        reward_class = -1

    other = jnp.maximum(neighbor, 0)
    other_id = identities[other]
    # Training infers each pair's labels; equal labels collapse to the null category.
    # Imagined rollouts pass cached scene identities and retain equal labels.
    known = (neighbor >= 0) & (~jnp.asarray(training) | (other_id != identities))
    other_id = jnp.where(known, other_id, config.identities)
    features = scene[:, :5].at[:, 3:5].multiply(10)
    if not config.fixed_radius:
        delta = scene[:, :2] - scene[other, :2]
        if config.relative_scale:
            valid &= jnp.all((neighbor < 0) | (scene[:, 6:8] > 0).all(axis=1))
            delta = delta / jnp.where(scene[:, 6:8] > 0, scene[:, 6:8], 1.)
        features = jnp.concatenate((features, jnp.where((neighbor < 0)[:, None],
                                                        absent.astype(scene.dtype), delta)), axis=1)
    valid &= jnp.isfinite(features).all()
    n, d = features.shape
    fields = (identities, other_id, (scene[:, 2] == 0).astype(jnp.int32),
              jnp.full(n, action), jnp.full(n, reward_class), switches)
    return InteractionBatch(MixedBatch(features, jnp.zeros((n, d, d), scene.dtype),
                                       jnp.ones(n, bool), fields), neighbor, valid)

"""Roll out learned AXIOM motion and interaction tables from an observed scene.

The host freezes a compact snapshot once per real decision. Within the
simulation, object identities and learned parameters stay fixed while
positions, velocities, contact evidence and actions change each frame.
"""

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from cffg import mixed_scoring
from cffg.experiments.axiom_evidence import release_view
from cffg.experiments.axiom_interaction import interaction_batch
from cffg.experiments.axiom_prediction import ObjectDynamics, ObjectPrediction, predict_objects
from cffg.mixed_state import MixedBank


class WorldStatus(IntEnum):
    READY = 0
    NO_TRACKED_INTERACTION_SUPPORT = 1
    INVALID_MODEL = 2


class WorldSnapshot(NamedTuple):
    gaussian: MixedBank
    context_terms: tuple
    outputs: ObjectDynamics
    component_rows: jax.Array
    total_alpha: jax.Array
    identities: jax.Array
    tracked: jax.Array
    focused_slots: jax.Array
    absent_distance: jax.Array
    status: jax.Array

    @property
    def valid(self):
        return self.status == WorldStatus.READY


def _bucket(mask, *, pad):
    """Keep stable order and round up to a power of two to limit recompilations."""
    live = np.flatnonzero(mask).astype(np.int32)
    size = 1 << (max(1, len(live)) - 1).bit_length()
    return np.pad(live, (0, size - len(live)), constant_values=pad)


@jax.jit
def _prepare(posterior, rows):
    # Padding uses a real component's parameters and never gets probability.
    live = rows >= 0
    indices = jnp.where(live, rows, jnp.maximum(rows[0], 0))
    subset = jax.tree.map(lambda x: x[indices], posterior)
    # A finite dummy makes all-untracked, empty-bank rollouts well-defined.
    support = live.at[0].set(live[0] | ~live.any())
    subset = release_view(subset._replace(active=support))
    proper = mixed_scoring.validate_bank(subset)
    means = []
    terms = []
    for i, (head, low) in enumerate(zip(subset.categories, subset.carry.categories, strict=True)):
        if i < 4:
            table = mixed_scoring._dirichlet_log(head, True)
            terms.append((table, mixed_scoring._dirichlet_adjustment(head, low, True, table)))
        else:
            counts = head + low
            means.append(counts / counts.sum(axis=1, keepdims=True))
    gaussian = subset._replace(categories=(), carry=subset.carry._replace(categories=()))
    masses = posterior.alpha + posterior.carry.alpha
    proper &= (jnp.isfinite(masses) & (masses > 0)).all()
    total = masses.sum()
    return gaussian, tuple(terms), tuple(means), total, proper


def freeze_world(posterior, motion, identities, tracked, absent_distance, *, config, noise_sd=None):
    """Prepare one live model for sampled rollouts. Pass the raw rMM MixedBank.

    The six categorical heads are own/other identity, visible, action, reward
    and motion switch. All motion matrices, including unused zero maps, retain
    their released switch indices. For an empty rMM a tracked prediction fails;
    a frame with no tracked objects retains its state and scores zero.
    """
    tracked = np.asarray(tracked)
    labels = jnp.asarray(identities)
    motion_shape = motion.affine.shape
    dtype = posterior.mu.dtype
    expected = (config.identities + 1, config.identities + 1, 2, config.actions, 3, config.motions)
    if (tuple(head.shape[1] for head in posterior.categories) != expected
            or posterior.mu.shape[1] != (5 if config.fixed_radius else 7)
            or motion_shape != (config.motions, 6, 7)):
        raise ValueError("interaction heads, continuous features and motion maps must match config")
    if (tracked.ndim != 1 or not len(tracked) or tracked.dtype != bool
            or labels.shape != tracked.shape or not jnp.issubdtype(labels.dtype, jnp.integer)):
        raise ValueError("expected one integer identity and a boolean mask per scene slot")
    absent = jnp.asarray(absent_distance, dtype)
    noise = jnp.zeros(6, dtype) if noise_sd is None else jnp.asarray(noise_sd, dtype)
    if absent.shape != (2,) or noise.shape != (6,) or motion.affine.dtype != dtype:
        raise ValueError("expected a two-vector for missing contacts and six motion noise scales")

    rows = jnp.asarray(_bucket(np.asarray(posterior.active), pad=-1))
    focus = jnp.asarray(_bucket(tracked, pad=len(tracked)))
    gaussian, terms, (reward_p, motion_p), total, valid = _prepare(posterior, rows)
    outputs = ObjectDynamics(gaussian.alpha + gaussian.carry.alpha, gaussian.active,
                             motion_p, reward_p, motion.affine, noise)
    valid &= (jnp.isfinite(total) & (total > 0)
              & jnp.isfinite(absent).all() & jnp.isfinite(motion.affine).all()
              & jnp.isfinite(noise).all() & (noise >= 0).all()
              & ((labels >= 0) & (labels < config.identities)).all())
    status = jnp.where(valid, jnp.where((rows >= 0).any() | ~tracked.any(), WorldStatus.READY,
                                       WorldStatus.NO_TRACKED_INTERACTION_SUPPORT),
                       WorldStatus.INVALID_MODEL)
    return WorldSnapshot(gaussian, terms, outputs, rows, total, labels, jnp.asarray(tracked),
                         focus, absent, status)


def _context_scores(snapshot, records):
    # Only the four known categorical heads enter a hypothetical transition.
    # Center each contribution before summing so a very unlikely continuous
    # field shared across all components cannot erase categorical log odds.
    active = snapshot.gaussian.active
    def relative(x):
        supported = jnp.where(active, x, -jnp.inf)
        return supported - supported.max(axis=1, keepdims=True)

    base, low = mixed_scoring._gaussian_log(snapshot.gaussian, records)
    scores = base + low
    logits = relative(relative(base) + relative(low))
    for label, (table, correction) in zip(records.categories[:4], snapshot.context_terms, strict=True):
        for term in (table.T[label], correction.T[label]):
            scores += term
            logits = relative(logits + relative(term))
    return jnp.where(active, scores, -jnp.inf), jax.nn.softmax(logits, axis=1)


def predict_scene(snapshot, scene, action, key, *, config, match_threshold=-10.,
                  reward_threshold=.45):
    """Predict one frame. Component and switch outputs use original physical rows."""
    evidence = interaction_batch(scene, snapshot.identities, snapshot.tracked, action,
                                 snapshot.absent_distance, config=config)
    count = scene.shape[0]
    if scene.dtype != snapshot.gaussian.mu.dtype:
        raise TypeError("scene and model must share a floating dtype")
    focus = snapshot.focused_slots
    take = jnp.minimum(focus, count - 1)
    records = jax.tree.map(lambda x: x[take], evidence.records)
    scores, q = _context_scores(snapshot, records)
    pred = predict_objects(snapshot.outputs, scene[take], focus < count, scores, key,
                           responsibilities=q, total_alpha=snapshot.total_alpha,
                           match_threshold=match_threshold, reward_threshold=reward_threshold)
    scored = ((focus == count)[:, None] | ~snapshot.gaussian.active | jnp.isfinite(scores)).all()
    valid = (evidence.valid & snapshot.valid & scored
             & jnp.isfinite(pred.reward) & jnp.isfinite(pred.novelty))
    state = scene.at[focus].set(pred.state, mode="drop")
    component = snapshot.component_rows[jnp.maximum(pred.component, 0)]
    empty = jnp.full((count,), -1, jnp.int32)
    components = empty.at[focus].set(component, mode="drop")
    switches = empty.at[focus].set(pred.motion, mode="drop")
    novelty = pred.novelty * (focus.size / count)
    return ObjectPrediction(jnp.where(valid, state, scene), jnp.where(valid, components, -1),
                            jnp.where(valid, switches, -1), jnp.where(valid, pred.reward, jnp.nan),
                            jnp.where(valid, novelty, jnp.nan))


def rollout_step(snapshot, scene, action, key, *, config, match_threshold=-10.,
                 reward_threshold=.45):
    """Bind config with functools.partial and pass this callback to axiom_planning.plan."""
    result = predict_scene(snapshot, scene, action, key, config=config,
                           match_threshold=match_threshold, reward_threshold=reward_threshold)
    return result.state, (result.reward, result.novelty)

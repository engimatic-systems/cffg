"""AXIOM object predictions from frozen interaction scores and motion experts.

The caller supplies absolute component log likelihoods for the current scene
and candidate action, with unknown reward and motion evidence omitted. Learned
features and interaction geometry belong to the object model. This module uses
the released agent's reward and reciprocal-count heuristics.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class ObjectDynamics(NamedTuple):
    """K interaction components, M motion classes, D dynamic coordinates."""
    alpha: jax.Array            # [K] including dormant components
    occupied: jax.Array         # [K]
    motion_probability: jax.Array  # [K,M]
    reward_probability: jax.Array  # [K,3] for negative, neutral, positive
    affine: jax.Array           # [M,D,D+1], bias in the last column
    noise_sd: jax.Array         # [D] standard deviations in the motion model


class ObjectPrediction(NamedTuple):
    state: jax.Array
    component: jax.Array
    motion: jax.Array
    reward: jax.Array          # one signed reward from the most extreme object
    novelty: jax.Array         # mean bonus over all slots, including untracked


def _draw_rows(weights, key):
    """Inverse CDF using one shared uniform draw across the scene's objects."""
    cumulative = jnp.cumsum(weights, axis=-1)
    total = cumulative[:, -1]
    point = jax.random.uniform(key, (), dtype=weights.dtype) * total
    # Rounding the product up to the endpoint must not select a zero-mass tail.
    point = jnp.minimum(point, jnp.nextafter(total, jnp.zeros_like(total)))
    index = (cumulative <= point[:, None]).sum(axis=1, dtype=jnp.int32)
    # Rows with empty support are flagged by the caller. Keep the gather in bounds.
    return jnp.minimum(index, weights.shape[1] - 1)


def predict_objects(model, objects, tracked, log_likelihood, key, *,
                    match_threshold=-10., reward_threshold=.45,
                    responsibilities=None, total_alpha=None):
    """Draw one scene's successor and score it. Inputs remain read-only.

    Objects have shape [slots, features]. The first D features evolve and
    the first two give position in centered frame coordinates. Untracked slots
    stay fixed. A tracked slot without a usable prediction yields NaN scores.
    As in the release, object draws share random numbers within a scene. Each
    simulated time, policy and replicate receives its own key from the sampler.
    """
    alpha, occupied, motion_p, reward_p, affine, noise_sd = model
    slots, features = objects.shape
    modes, dim, columns = affine.shape
    components = alpha.shape[0]
    if not (slots and components and modes and 2 <= dim <= features and columns == dim + 1
            and alpha.shape == occupied.shape == (components,) and noise_sd.shape == (dim,)
            and motion_p.shape == (components, modes) and reward_p.shape == (components, 3)
            and tracked.shape == (slots,) and log_likelihood.shape == (slots, components)):
        raise ValueError("incompatible object, interaction or motion model dimensions")
    comp_key, motion_key, noise_key = jax.random.split(key, 3)
    logits = jnp.where(occupied, log_likelihood, -jnp.inf)
    fit = jnp.max(logits, axis=1)
    q = jax.nn.softmax(logits, axis=1) if responsibilities is None else responsibilities
    if q.shape != logits.shape:
        raise ValueError("expected one responsibility per object and component")
    component = _draw_rows(q, comp_key)
    motion_weights = jnp.where(motion_p > 1e-4, motion_p, 0)[component]
    motion = _draw_rows(motion_weights, motion_key)
    maps = affine[motion]
    positions = jnp.einsum("sij,sj->si", maps[:, :, :-1], objects[:, :dim]) + maps[:, :, -1]
    positions += noise_sd * jax.random.normal(noise_key, (dim,), dtype=positions.dtype)
    next_state = objects.at[:, :dim].set(jnp.where(tracked[:, None], positions, objects[:, :dim]))

    # Retain the released positive-reward override and unnormalized thresholded heads.
    supported_reward = jnp.where(reward_p > reward_threshold, reward_p, 0)
    signed_reward = q @ (supported_reward[:, 2] - supported_reward[:, 0])
    rewards = jnp.where(supported_reward[component, 2] > reward_threshold, 1., signed_reward)
    rewards = jnp.where(tracked, rewards, 0.)
    reward = rewards[jnp.argmax(jnp.abs(rewards))]

    # A compact snapshot can contain only occupied rows. Dormant counts still
    # contribute to the reference novelty baseline.
    total = alpha.sum() if total_alpha is None else total_alpha
    inverse_total = 1 / total
    # Divide after weighting so a dormant component contributes exactly zero
    # even when its count is too small for its reciprocal to be representable.
    bonuses = jnp.sum(q / alpha, axis=1) - inverse_total
    bonuses = jnp.where(fit < match_threshold, 1 - inverse_total, bonuses)
    outside = jnp.any(jnp.abs(positions[:, :2]) > 1.1, axis=1)
    novelty = jnp.mean(jnp.where(tracked, jnp.where(outside, -1., bonuses), 0.))

    # Tree summation of a normalized head can round at each reduction level.
    motion_tol = (8 + modes.bit_length()) * jnp.finfo(motion_p.dtype).eps
    reward_tol = 10 * jnp.finfo(reward_p.dtype).eps
    proper = (jnp.all(jnp.isfinite(alpha) & (alpha > 0))
              & jnp.all(jnp.isfinite(noise_sd) & (noise_sd >= 0))
              & jnp.all(jnp.isfinite(motion_p) & (motion_p >= 0) & (motion_p <= 1))
              & jnp.all(jnp.isfinite(reward_p) & (reward_p >= 0) & (reward_p <= 1))
              & jnp.all(jnp.abs(motion_p.sum(axis=1) - 1) <= motion_tol)
              & jnp.all(jnp.abs(reward_p.sum(axis=1) - 1) <= reward_tol))
    q_tol = (8 + components.bit_length()) * jnp.finfo(q.dtype).eps
    proper &= jnp.isfinite(total) & (total > 0)
    valid_q = (jnp.isfinite(q) & (q >= 0) & (occupied | (q == 0))).all(axis=1)
    valid_slot = (valid_q & (jnp.abs(q.sum(axis=1) - 1) <= q_tol)
                  & jnp.isfinite(fit) & (motion_weights.sum(axis=1) > 0)
                  & jnp.isfinite(positions).all(axis=1))
    valid = proper & jnp.all(~tracked | valid_slot)
    return ObjectPrediction(next_state, jnp.where(tracked, component, -1),
                            jnp.where(tracked, motion, -1), jnp.where(valid, reward, jnp.nan),
                            jnp.where(valid, novelty, jnp.nan))

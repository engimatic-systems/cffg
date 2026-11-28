"""Full-budget stochastic search with memoized point likelihoods for noiseless affine motion.

Only identical bit patterns in a single frozen decision share a calculation.
Each policy/replicate/tick uses the ordinary planner's random key and its own
two draws. Physics advances every sampled motion; the cache holds no future
return, model update or policy choice.
"""

from collections import OrderedDict
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from cffg.experiments.axiom_interaction import interaction_batch
from cffg.experiments.axiom_planning import _candidates, refit
from cffg.experiments.axiom_rollout import _context_scores
from cffg.mixed_learning import MixedBatch
from cffg.rollout_planning import trajectory_key


@partial(jax.jit, static_argnames=("settings",))
def _streams(proposal, key, *, settings):
    sample_key, sim = jax.random.split(key)
    support = jnp.ones(proposal.probabilities.shape[1], bool)
    policies, valid = _candidates(proposal, support, settings, sample_key)
    dtype = proposal.probabilities.dtype
    def draws(i, d, t):
        z, motion, _ = jax.random.split(jax.random.fold_in(trajectory_key(sim, i, d), t), 3)
        return jnp.stack((jax.random.uniform(z, (), dtype), jax.random.uniform(motion, (), dtype)))
    ticks = jnp.arange(policies.shape[1])
    one_draw = jax.vmap(draws, in_axes=(None, None, 0))
    one_policy = jax.vmap(one_draw, in_axes=(None, 0, None))
    all_draws = jax.vmap(one_policy, in_axes=(0, None, None))
    return policies, valid, all_draws(jnp.arange(settings.candidates), jnp.arange(settings.replicates), ticks)


@partial(jax.jit, static_argnames=("config",))
def _features(model, observed, states, actions, *, config):
    # In the untuned model only tracked neighbors can interact. Preserve their
    # original order for distance ties. Sizes and identity labels stay fixed.
    focus = model.focused_slots
    take = jnp.minimum(focus, observed.shape[0] - 1)
    tracked = focus < observed.shape[0]
    def record(dynamics, action):
        scene = observed[take].at[:, :6].set(dynamics)
        evidence = interaction_batch(scene, model.identities[take], tracked, action,
                                     model.absent_distance, config=config)
        return evidence.records.mean, jnp.stack(evidence.records.categories[:4], axis=-1), evidence.valid
    return jax.vmap(record)(states, actions)


@jax.jit
def _queries(model, point, labels):
    n, d = point.shape
    query = MixedBatch(point, jnp.zeros((n, d, d), point.dtype), jnp.ones(n, bool),
                       tuple(labels[:, i] for i in range(4)))
    scores, q = _context_scores(model, query)
    reward_p = jnp.where(model.outputs.reward_probability > .45, model.outputs.reward_probability, 0)
    reward = q @ (reward_p[:, 2] - reward_p[:, 0])
    inverse_total = 1 / model.total_alpha
    bonus = jnp.sum(q / model.outputs.alpha, axis=1) - inverse_total
    bonus = jnp.where(scores.max(axis=1) < -10., 1 - inverse_total, bonus)
    tolerance = (8 + q.shape[1].bit_length()) * jnp.finfo(q.dtype).eps
    valid = ((jnp.isfinite(scores) | ~model.gaussian.active).all(axis=1)
             & (jnp.isfinite(q) & (q >= 0) & (model.gaussian.active | (q == 0))).all(axis=1)
             & (jnp.abs(q.sum(axis=1) - 1) <= tolerance))
    return jnp.cumsum(q, axis=1), reward, bonus, valid


@jax.jit
def _motion_cdf(model):
    p = model.outputs.motion_probability
    r = model.outputs.reward_probability
    eps = jnp.finfo(p.dtype).eps
    valid = (((p >= 0) & (p <= 1)).all() & ((r >= 0) & (r <= 1)).all()
             & (jnp.abs(p.sum(axis=1) - 1) <= (8 + p.shape[1].bit_length()) * eps).all()
             & (jnp.abs(r.sum(axis=1) - 1) <= 10 * eps).all())
    return jnp.cumsum(jnp.where(p > 1e-4, p, 0), axis=1), valid


@jax.jit
def _advance(model, state, switch, component, expected_reward, inside_bonus):
    focus = model.focused_slots
    tracked = focus < model.tracked.size
    maps = model.outputs.affine[switch]
    positions = jnp.einsum("nsij,nsj->nsi", maps[:, :, :, :-1], state) + maps[:, :, :, -1]
    next_state = jnp.where(tracked[None, :, None], positions, state)
    positive = model.outputs.reward_probability[component, 2] > .45
    rewards = jnp.where(tracked, jnp.where(positive, 1., expected_reward), 0.)
    best = jnp.argmax(jnp.abs(rewards), axis=1)
    rewards = rewards[jnp.arange(best.size), best]
    outside = (jnp.abs(positions[:, :, :2]) > 1.1).any(axis=2)
    novelty = jnp.where(tracked, jnp.where(outside, -1., inside_bonus), 0.).mean(axis=1)
    return next_state, rewards, novelty * (focus.size / model.tracked.size)


@partial(jax.jit, static_argnames=("settings",))
def _finish(policies, rewards, bonuses, previous, valid, *, settings):
    shape = settings.candidates, settings.replicates, policies.shape[1]
    return refit(policies, rewards.reshape(shape).mean(axis=1), bonuses.reshape(shape).mean(axis=1),
                 previous, settings, valid=valid)


def _thresholds(cdf):
    # A parallel prefix sum may round downward at a positive boundary. The
    # reference counts thresholds <= the draw; sorting preserves that count.
    # Every draw is strictly below its original final mass.
    if np.any(cdf[..., 1:] < cdf[..., :-1]):
        return np.sort(np.minimum(cdf, cdf[..., -1:]), axis=-1)
    return cdf


def _draw(cdf, rows, uniform):
    """Same right-inclusive inverse CDF as the reference, without materializing repeated rows."""
    total = cdf[rows, -1]
    if not np.all(np.isfinite(total) & (total > 0)):
        raise ArithmeticError("no finite categorical mass for a sampled rollout")
    point = np.minimum(uniform * total, np.nextafter(total, np.zeros_like(total)))
    lo = np.zeros(rows.shape, np.int32)
    hi = np.full_like(lo, cdf.shape[-1])
    for _ in range(cdf.shape[-1].bit_length()):
        mid = (lo + hi) // 2
        below = cdf[rows, np.minimum(mid, cdf.shape[-1] - 1)] <= point
        lo = np.where(below, mid + 1, lo)
        hi = np.where(below, hi, mid)
    return np.minimum(lo, cdf.shape[-1] - 1)


class _Queries:
    """Fixed-memory LRU of distinct per-object evidence and its predicted scores."""
    def __init__(self, model):
        k = model.gaussian.active.size
        itemsize = np.dtype(model.gaussian.mu.dtype).itemsize
        self.capacity = max(1, min(16384, (64 << 20) // ((k + 2) * itemsize)))
        self.batch = max(1, min(128, self.capacity, 16384 // k))
        shape = (self.capacity,)
        self.cdf = np.empty((*shape, k), model.gaussian.mu.dtype)
        self.reward = np.empty(shape, model.gaussian.mu.dtype)
        self.bonus = np.empty_like(self.reward)
        self.index = OrderedDict()
        self.evaluations = 0
        self.requests = 0

    def load(self, keys, points, labels, model):
        self.requests += len(keys)
        indices = np.empty(len(keys), np.int32)
        missing = []
        for i, key in enumerate(keys):
            if key in self.index:
                indices[i] = self.index[key]
                self.index.move_to_end(key)
            else:
                missing.append(i)
        self.evaluations += len(missing)
        for start in range(0, len(missing), self.batch):
            wanted = missing[start:start + self.batch]
            padded = np.pad(wanted, (0, self.batch - len(wanted)), mode="edge")
            result = _queries(model, points[padded], labels[padded])
            cdf, reward, bonus, valid = map(np.asarray, result)
            cdf = _thresholds(cdf)
            if not valid[:len(wanted)].all():
                raise ArithmeticError("invalid interaction likelihood in an imagined state")
            for j, i in enumerate(wanted):
                row = len(self.index)
                if row == self.capacity:
                    _, row = self.index.popitem(last=False)
                self.index[keys[i]] = indices[i] = row
                self.cdf[row], self.reward[row], self.bonus[row] = cdf[j], reward[j], bonus[j]
        return indices


def cached_plan(model, observed, proposal, key, *, config, settings, visit=None):
    """Simulate the populated, zero-noise untuned model. All trajectories keep distinct draws."""
    if not bool(model.valid) or np.any(np.asarray(model.outputs.noise_sd) != 0):
        raise ValueError("cached prediction requires a valid snapshot and zero motion noise")
    if config.fixed_radius or config.static_contacts:
        raise ValueError("compressed neighborhoods require untuned dynamic-only contact geometry")
    x = np.asarray(observed)
    if not np.isfinite(x[:, :8]).all() or np.any(x[:, 6:8] < 0):
        raise ArithmeticError("invalid observed scene")
    policies, valid, samples = _streams(proposal, key, settings=settings)
    draws = np.asarray(samples).reshape(-1, policies.shape[1], 2)
    commands = np.repeat(np.asarray(policies), settings.replicates, axis=0)
    n, horizon = commands.shape
    focus = np.array(model.focused_slots)
    slots = len(focus)
    start = x[np.minimum(focus, len(x) - 1), :6]
    state = jnp.broadcast_to(start, (n, slots, 6))
    motion_cdf, valid_heads = _motion_cdf(model)
    if not bool(valid_heads):
        raise ArithmeticError("invalid outcome probabilities in the frozen snapshot")
    motion_cdf = _thresholds(np.asarray(motion_cdf))
    rewards = np.empty((n, horizon), x.dtype)
    bonuses = np.empty_like(rewards)
    cache = _Queries(model)
    chosen = np.empty(n * slots, np.int32)
    signed = np.empty(n * slots, x.dtype)
    inside = np.empty_like(signed)

    for t in range(horizon):
        features, categories, valid_features = _features(model, observed, state, commands[:, t], config=config)
        if not np.asarray(valid_features).all():
            raise ArithmeticError("invalid context in an imagined state")
        points = np.asarray(features).reshape(len(chosen), -1)
        labels = np.asarray(categories).reshape(len(chosen), 4)
        # Bytewise equality preserves every feature bit. Categorical fields
        # join the key because equal positions can imply different action odds.
        parts = (points.view(np.uint8).reshape(len(chosen), -1),
                 labels.view(np.uint8).reshape(len(chosen), -1))
        raw = np.concatenate(parts, axis=1)
        signatures = raw.view(f"V{raw.shape[1]}").reshape(len(chosen))
        keys, first, which = np.unique(signatures, return_index=True, return_inverse=True)
        comp_draw = np.broadcast_to(draws[:, t, 0, None], (n, slots)).reshape(-1)
        for offset in range(0, len(keys), cache.capacity):
            end = min(offset + cache.capacity, len(keys))
            group = keys[offset:end].tolist()
            source = first[offset:end]
            indices = cache.load(group, points[source], labels[source], model)
            members = np.flatnonzero((which >= offset) & (which < end))
            rows = indices[which[members] - offset]
            chosen[members] = _draw(cache.cdf, rows, comp_draw[members])
            signed[members], inside[members] = cache.reward[rows], cache.bonus[rows]
        components = chosen.reshape(n, slots)
        switch = _draw(motion_cdf, components, draws[:, t, 1, None])
        if visit is not None:
            live = focus < len(observed)
            visit(t, np.where(live, np.asarray(model.component_rows)[components], -1),
                  np.where(live, switch, -1))
        state, reward, bonus = _advance(model, state, switch, components,
                                        signed.reshape(n, slots), inside.reshape(n, slots))
        rewards[:, t], bonuses[:, t] = np.asarray(reward), np.asarray(bonus)
    if not np.isfinite(np.asarray(state)).all():
        raise ArithmeticError("nonfinite imagined motion")
    result = _finish(policies, rewards, bonuses, proposal, valid, settings=settings)
    info = dict(unique_context_evaluations=cache.evaluations, context_requests=cache.requests,
                likelihood_batch=cache.batch,
                cache_bytes=cache.cdf.nbytes + cache.reward.nbytes + cache.bonus.nbytes)
    return result, info

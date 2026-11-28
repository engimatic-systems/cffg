"""Sampled open-loop AXIOM policy search, independently implemented in JAX.

The release reference is VersesTech/axiom at b99f625, planner.py and
models/rmm.py. A refinement ranks discounted scores; the executed action
maximizes undiscounted return. The two decisions can select different policies.
"""

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from cffg.rollout_planning import sample_action_sequences


@dataclass(frozen=True)
class SearchSettings:
    candidates: int = 512
    replicates: int = 3
    repeat_probability: float = 0.
    novelty_weight: float = .1
    discount: float = .99
    batch_size: int | None = None

    def __post_init__(self):
        if self.candidates < 2 or self.replicates < 1:
            raise ValueError("expected multiple candidate policies and positive replicates")
        if not (0 <= self.repeat_probability <= 1 and 0 <= self.discount <= 1):
            raise ValueError("repeat probability and discount must lie in [0, 1]")
        if not math.isfinite(self.novelty_weight):
            raise ValueError("novelty_weight must be finite")


class Proposal(NamedTuple):
    """Checkpoint both arrays, and save the caller's next random key separately."""
    probabilities: jax.Array   # [time, action]
    best: jax.Array            # last winner, [time]


def initial_proposal(action_count, *, horizon=32, dtype=jnp.float32):
    if min(horizon, action_count) < 1 or not jnp.issubdtype(dtype, jnp.floating):
        raise ValueError("expected positive horizon/action count and floating probabilities")
    return Proposal(jnp.full((horizon, action_count), 1 / action_count, dtype=dtype),
                    jnp.zeros(horizon, dtype=jnp.int32))


class SearchResult(NamedTuple):
    action: jax.Array          # -1 if invalid
    valid: jax.Array
    proposal: Proposal
    policies: jax.Array        # [candidate, time]
    reward: jax.Array          # replicate means [candidate, time]
    novelty: jax.Array
    discounted_return: jax.Array
    total_return: jax.Array


def _candidates(previous, support, settings, key):
    probs, best = previous
    horizon, actions = probs.shape
    if (settings.candidates <= actions or horizon < 1 or actions < 1
            or best.shape != (horizon,) or support.shape != (actions,)):
        raise ValueError("need a prior-best slot, one slot per action and consistent axes")
    if (not jnp.issubdtype(probs.dtype, jnp.floating)
            or not jnp.issubdtype(best.dtype, jnp.integer) or support.dtype != jnp.bool_):
        raise TypeError("expected floating proposal, integer plan and boolean support")
    # Unit sums from a tree reduction may round by a few ulps per reduction level.
    tolerance = (8 + actions.bit_length()) * jnp.finfo(probs.dtype).eps
    valid = (support.any() & jnp.isfinite(probs).all() & (probs >= 0).all()
             & ((best >= 0) & (best < actions)).all() & (jnp.abs(probs.sum(axis=1) - 1) <= tolerance).all())
    # Keep categorical draws defined even on the invalid/no-action return path.
    allowed = support | ~support.any()
    uniform = allowed.astype(probs.dtype) / allowed.sum()
    shifted = jnp.concatenate((probs[1:], uniform[None, :]))
    # The released proposal adds this sampling mass to each categorical entry.
    # The explicit action mask also applies to exploration and injected policies.
    logits = jnp.where(allowed, jnp.log(shifted + 1e-16), -jnp.inf)
    learned_key, explore_key, repeat_key = jax.random.split(key, 3)
    count = settings.candidates
    random_count = count // 2
    policies = jax.random.categorical(learned_key, logits[None, :, :],
                                      shape=(count, horizon)).astype(jnp.int32)
    draws = jax.random.categorical(explore_key, jnp.log(uniform),
                                    shape=(horizon, random_count)).astype(jnp.int32)
    repeat = jax.random.uniform(repeat_key, draws.shape) < settings.repeat_probability

    def smooth(last, pair):
        draw, keep = pair
        # Emit the previous action, then advance. The first two actions coincide.
        return jnp.where(keep, last, draw), last

    _, explored = jax.lax.scan(smooth, draws[0], (draws, repeat))
    policies = policies.at[-random_count:].set(explored.T)
    first_legal = jnp.argmax(allowed).astype(jnp.int32)
    shifted_best = jnp.roll(best, -1)
    policies = policies.at[0].set(jnp.where(allowed[shifted_best], shifted_best, first_legal))
    constants = jnp.where(allowed, jnp.arange(actions, dtype=jnp.int32), first_legal)
    return policies.at[1:actions + 1].set(constants[:, None]), valid


def refit(policies, reward, novelty, previous, settings, *, valid=True):
    """Rank a supplied candidate pool, refit its categorical proposal and pick a move.

    Ties retain the earliest candidates for the elite histogram and the last
    candidate for the executed action, as in the release's stable sorted ranks.
    Any nonfinite score invalidates the decision and retains the previous proposal.
    """
    if (policies.shape != reward.shape or reward.shape != novelty.shape or reward.ndim != 2
            or policies.shape != (settings.candidates, previous.probabilities.shape[0])):
        raise ValueError("policies and both mean scores must share candidate and time axes")
    if not jnp.issubdtype(policies.dtype, jnp.integer):
        raise TypeError("policies must contain integer actions")
    value = reward + settings.novelty_weight * novelty
    discounts = jnp.asarray(settings.discount, value.dtype) ** jnp.arange(value.shape[1])
    discounted = (value * discounts).sum(axis=1)
    total = value.sum(axis=1)
    action_count = previous.probabilities.shape[1]
    valid = (valid & jnp.isfinite(discounted).all() & jnp.isfinite(total).all()
             & ((policies >= 0) & (policies < action_count)).all())
    # Scale before subtraction only when finite extrema straddle an overflowing range.
    low, high = discounted.min(), discounted.max()
    span = high - low
    scale = jnp.where(jnp.isinf(span), jnp.maximum(jnp.abs(high), jnp.abs(low)), 1.)
    relative = discounted / scale - low / scale
    width = high / scale - low / scale
    # The release's min/max normalization includes an additive 1e-16, which
    # changes elite weights when the entire reward range is small.
    normalized = relative / (width + 1e-16 / scale)
    elite = jnp.argsort(discounted, descending=True, stable=True)[:max(1, settings.candidates // 10)]
    weights = jax.nn.softmax(10 * normalized[elite])
    frequencies = jnp.einsum("e,eha->ha", weights,
                             jax.nn.one_hot(policies[elite], action_count, dtype=weights.dtype))
    winner = policies[policies.shape[0] - 1 - jnp.argmax(total[::-1])]
    frequencies = frequencies.astype(previous.probabilities.dtype)
    next_proposal = Proposal(jnp.where(valid, frequencies, previous.probabilities),
                             jnp.where(valid, winner, previous.best))
    return SearchResult(jnp.where(valid, winner[0], -1), valid, next_proposal, policies,
                        reward, novelty, discounted, total)


def plan(step, snapshot, observed_state, previous, key, *, settings=SearchSettings(), support=None):
    """Plan once after a real observation using step(snapshot, state, action, key).

    The pure step returns (imagined_next_state, (reward, novelty)), both scores
    scalar. Set novelty_weight=0 for the no-bonus ablation. The support vector
    declares globally legal actions. On an invalid decision action is -1.
    """
    action_count = previous.probabilities.shape[1]
    support = jnp.ones(action_count, dtype=bool) if support is None else jnp.asarray(support)
    proposal_key, simulation_key = jax.random.split(key)
    policies, valid = _candidates(previous, support, settings, proposal_key)
    _, (reward, novelty) = sample_action_sequences(
        step, snapshot, observed_state, policies, simulation_key,
        replicates=settings.replicates, batch_size=settings.batch_size)
    return refit(policies, reward.mean(axis=1), novelty.mean(axis=1),
                 previous, settings, valid=valid)

"""Released iMM/rMM assignment and per-component write policy.

Reference: VersesTech/axiom b99f625, models/utils_hybrid.py and
vi/models/hybrid_mixture_model.py. For a point with responsibility q, the
candidate natural parameters are prior + q*T(point) + h*(posterior-prior).
Here h=q if the winning component's previous mixture count exceeds one and
zero otherwise. This module uses centered Gaussian statistics for that
calculation. It accepts the ordinary CFFG mixed learner's bank and batch types.
"""

from enum import IntEnum
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from cffg import _compensated, component_bank_device as lifecycle
from cffg.component_bank import ComponentBank, SlotStatus
from cffg.experiments.axiom_evidence import release_view
from cffg.mixed_learning import MixedBank, MixedBatch, score
from cffg.mixed_scoring import validate_bank
from cffg.mixed_state import MixedCarry


class AssignmentStatus(IntEnum):
    SKIPPED = 0
    MATCHED = 1
    NOVEL = 2
    FULL_TO_ZERO = 3
    INVALID = 4
    ROLLED_BACK = 5


class LearningResult(NamedTuple):
    bank: lifecycle.DeviceBank | ComponentBank
    assignment: jax.Array
    written: jax.Array
    born: jax.Array
    status: jax.Array
    valid: jax.Array


def _plus(high, low, increment):
    high, low, _ = _compensated.add(high, low, increment)
    return high, low


def _interpolate(a, low_a, b, low_b, weight):
    """Centered convex combination of two stored high/low pairs."""
    weight = weight.reshape((-1,) + (1,) * (a.ndim - 1))
    near_b = weight > .5
    start = jnp.where(near_b, b, a)
    tail = jnp.where(near_b, low_b, low_a)
    step = jnp.where(near_b, weight - 1, weight)
    # Endpoints reproduce both original words, even when a and b are far apart.
    delta = jnp.where(step == 0, 0, b - a)
    high, low = _plus(start, tail, step * delta)
    return _plus(high, low, step * (low_b - low_a))


def _weighted_point(prior, old, point, q, history):
    """Combine the prior, previous NIW and new point by their centered moments."""
    def blend(name, weight):
        return _interpolate(getattr(prior, name), getattr(prior.carry, name),
                            getattr(old, name), getattr(old.carry, name), weight)

    k, kl = blend("kappa", history)
    old_share = history * (old.kappa / k)
    old_share = jnp.where(history == 1, 1., jnp.where(history == 0, 0., old_share))
    mean, ml = blend("mu", old_share)
    scatter, sl = blend("psi", history)

    # The centered NIW scale gains the between-mean scatter at each pooling step.
    delta = (old.mu - prior.mu) + (old.carry.mu - prior.carry.mu)
    mass = (1 - history) * prior.kappa * old_share
    residual = jnp.sqrt(mass)[:, None] * jnp.where(mass[:, None] > 0, delta, 0)
    scatter, sl = _plus(scatter, sl, residual[:, :, None] * residual[:, None, :])

    new_k, new_kl = _plus(k, kl, q)
    portion = q / new_k
    delta = (point.mean[0] - mean) - ml
    mean, ml = _plus(mean, ml, portion[:, None] * delta)
    residual = jnp.sqrt(k * portion)[:, None] * delta
    scatter, sl = _plus(scatter, sl, residual[:, :, None] * residual[:, None, :])
    nu, nl = _plus(*blend("nu", history), q)
    alpha, al = _plus(*blend("alpha", history), q)

    heads, lows = [], []
    for a, la, b, lb, label in zip(prior.categories, prior.carry.categories, old.categories,
                                    old.carry.categories, point.categories, strict=True):
        head, low = _interpolate(a, la, b, lb, history)
        category = jax.nn.one_hot(label[0], a.shape[1], dtype=q.dtype)
        head, low = _plus(head, low, q[:, None] * category)
        heads.append(head)
        lows.append(low)
    return MixedBank(mean, new_k, scatter, nu, alpha, tuple(heads), old.active,
                     MixedCarry(ml, new_kl, sl, nl, al, tuple(lows)))


def _conditional_write(prior, old, point, weights):
    # At most three responsibilities strictly exceed one quarter. Work on
    # that many rows even when a frame's bank has thousands of components.
    qs, rows = jax.lax.top_k(weights, min(3, weights.size))
    history = jnp.where(old.alpha[weights.argmax()] > 1, qs, 0)
    old_rows, prior_rows = jax.tree.map(lambda field: field[rows], (old, prior))
    candidate = _weighted_point(prior_rows, old_rows, point, qs, history)
    # The released gate reads the working-dtype concentration.
    eligible = qs > .25
    write = eligible & (candidate.alpha < 32)
    def select(new, previous):
        mask = write.reshape((-1,) + (1,) * (new.ndim - 1))
        return jnp.where(mask, new, previous)

    selected = jax.tree.map(select, candidate, old_rows)
    valid = (jnp.all(~eligible | jnp.isfinite(candidate.alpha))
             & validate_bank(selected._replace(active=jnp.ones_like(selected.active))))
    posterior = jax.tree.map(lambda whole, subset: whole.at[rows].set(subset), old, selected)
    return posterior, jnp.zeros_like(old.active).at[rows].set(write), valid


def _assimilate(bank, point, threshold, zero_count_head):
    old = bank.posterior
    active = old.active
    # The scoring primitive requires nonempty support; bootstrap only validates
    # a dormant prior. Its score never competes with an occupied component.
    support = active.at[0].set(active[0] | ~active.any())
    scores, soft, valid = score(release_view(old._replace(active=support)), point)
    match = active.any() & (jnp.max(scores) > threshold)
    full = active.all()
    need_birth = ~match & ~full
    target = jnp.argmax(~active)
    weights = jnp.where(match, soft[0], jax.nn.one_hot(target, active.size, dtype=old.mu.dtype))
    # The first max over an all-false vacancy mask is row 0 on a full bank.
    def allocate(current):
        born = lifecycle.birth(current)
        return born.bank, born.status == lifecycle.Status.OK

    staged, allocated = jax.lax.cond(need_birth, allocate, lambda current: (current, True), bank)
    posterior, written, proper = _conditional_write(bank.prior, staged.posterior, point, weights)
    keep_birth = need_birth & written[target]
    owner = jax.tree.map(lambda new, prev: jnp.where(keep_birth, new, prev), staged, bank)
    owner = owner._replace(posterior=posterior._replace(active=owner.posterior.active))
    # A designated all-zero categorical observation adds zero ELL and counts.
    # Its previous counts still receive the same discount as other heads.
    complete = point.gaussian_observed[0] & jnp.all(point.covariance == 0)
    for i, field in enumerate(point.categories):
        if i != zero_count_head:
            complete &= field[0] >= 0
    valid &= allocated & proper & complete & jnp.isfinite(threshold)
    kind = jnp.where(match, AssignmentStatus.MATCHED,
                     jnp.where(full, AssignmentStatus.FULL_TO_ZERO, AssignmentStatus.NOVEL))
    return owner, (weights, written, jnp.where(keep_birth, target, -1).astype(jnp.int32),
                   jnp.where(valid, kind, AssignmentStatus.INVALID).astype(jnp.int32))


@partial(jax.jit, static_argnames=("zero_count_head",))
def learn_frame(bank, records, eligibility, *, threshold, valid_input=True, zero_count_head=None):
    """Apply complete observations in slot order and atomically publish the frame.

    bank is a DeviceBank from component_bank_device.from_host. eligibility is
    boolean [N]. Empty categories select iMM learning. A zero-based
    zero_count_head permits label -1 to encode an observed all-zero vector
    only in that field. On failure the previous
    bank is returned and every write/birth receipt is cleared.
    """
    eligibility = jnp.asarray(eligibility)
    if eligibility.shape != (records.mean.shape[0],) or eligibility.dtype != jnp.bool_:
        raise ValueError("eligibility must be a boolean per observation")
    if zero_count_head is not None and not 0 <= zero_count_head < len(records.categories):
        raise ValueError("zero_count_head must index a categorical field")
    k = bank.posterior.active.size

    def visit(current, item):
        row, enabled = item
        point = jax.tree.map(lambda x: x[None], row)
        def skip(_):
            return current, (jnp.zeros(k, bank.posterior.mu.dtype), jnp.zeros(k, bool),
                             jnp.int32(-1), jnp.int32(AssignmentStatus.SKIPPED))
        return jax.lax.cond(enabled, lambda _: _assimilate(current, point, threshold, zero_count_head), skip, None)

    candidate, (weights, written, born, status) = jax.lax.scan(
        visit, bank, (records, eligibility & valid_input))
    valid = valid_input & jnp.all(status != AssignmentStatus.INVALID)
    status = jnp.where(valid_input, status, jnp.int32(AssignmentStatus.INVALID))
    final = jax.tree.map(lambda after, before: jnp.where(valid, after, before), candidate, bank)
    status = jnp.where(valid | (status == AssignmentStatus.INVALID), status,
                       jnp.int32(AssignmentStatus.ROLLED_BACK))
    return LearningResult(final, weights, written & valid, jnp.where(valid, born, -1), status, valid)


def learn_observation(bank, batch, *, threshold, zero_count_head=None):
    """Host transaction for a single observed point from either learned mixture.

    Return scalar status/born and one vector each for assignment/write mask.
    Source ComponentBank, reset prior and caller's input record are immutable.
    """
    if batch.mean.shape[0] != 1:
        raise ValueError("learn_observation takes one point")
    device = learn_frame(lifecycle.from_host(bank), batch, jnp.ones(1, bool), threshold=threshold,
                         zero_count_head=zero_count_head)
    valid = bool(device.valid)
    target = bank
    if valid and int(device.born[0]) >= 0:
        activation = bank.birth()
        valid = activation.status == SlotStatus.OK
        target = activation.bank
    if valid:
        result = target.replace_posterior(device.bank.posterior, previous=target.posterior)
        valid = result.status == SlotStatus.OK
        target = result.bank
    status = device.status[0] if valid else jnp.int32(AssignmentStatus.INVALID)
    return LearningResult(target if valid else bank, device.assignment[0], device.written[0] & valid,
                          device.born[0] if valid else jnp.int32(-1), status, jnp.asarray(valid))

"""Greedy interaction-rule reduction on accumulated model-generated records.

Call reduce_interactions after a completed action. Its 500-action cadence
uses a dedicated explicit JAX key. Checkpoint ReductionState alongside the
ComponentBank.state_dict(exact=True) and the factory configuration/key for
dormant_initial.
Each pass freezes its proposed pairs and tape, then evaluates them in order.
The score is mean held-out reward/switch expected log likelihood minus the
mixing Dirichlet KL, in nats. A contextual winner with zero excess mixing
mass contributes zero. Host comparisons use float64 expected log scores.

With N generated records and K slots, the initial competition evaluates
up to N*K contexts in blocks. A trial that changes the bank-wide likelihood
repair rebuilds the competition. Fixed dormant slots with diagonal precision
have an exact score upper bound. Their likelihoods are evaluated for a record
only if they could win the initial decision or a later trial.
When the global repair is unchanged, the two revised rows compete against
the best untouched row.
An accepted local merge refreshes up to two 64-row blocks. Pair selection
separately compares at most 4,000 sampled sources against all K targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from cffg.component_bank import ComponentBank, SlotStatus
from cffg.experiments._axiom_reduction_scores import (
    Competition, Records, Tables, mixing, mixing_kl, soft_neighbors, validate_records,
)
from cffg.experiments.axiom_evidence import initialized_view, release_minima, sampling_view


_minima = jax.jit(release_minima)


def _key(key):
    key = jax.random.wrap_key_data(jax.random.key_data(key), impl=jax.random.key_impl(key))
    if key.shape != ():
        raise ValueError("expected one reduction random key")
    return key


@dataclass(frozen=True, eq=False)
class ReductionState:
    key: jax.Array
    last_step: int
    records: Records

    @classmethod
    def start(cls, bank, key):
        d = bank.posterior.mu.shape[1]
        empty = Records(np.empty((0, d), bank.posterior.mu.dtype), np.empty((0, 6), np.int32))
        return cls(_key(key), 0, empty)

    def state_dict(self):
        return {"version": np.int32(1), "last_step": np.int64(self.last_step),
                "key": np.array(jax.random.key_data(self.key), copy=True),
                "key_impl": np.array(str(jax.random.key_impl(self.key))),
                "features": self.records.features.copy(), "categories": self.records.categories.copy()}

    @classmethod
    def from_state_dict(cls, state, bank):
        if set(state) != {"version", "last_step", "key", "key_impl", "features", "categories"}:
            raise ValueError("unexpected or missing reduction checkpoint field")
        ver, step, impl = (np.asarray(state[k]) for k in ("version", "last_step", "key_impl"))
        if ver.shape or ver.dtype != np.int32 or ver != 1:
            raise ValueError("unknown reduction checkpoint version")
        if step.shape or step.dtype != np.int64 or step < 0 or step % 500:
            raise ValueError("invalid last reduction step")
        if impl.shape or impl.dtype.kind != "U":
            raise ValueError("invalid reduction random generator")
        key = _key(jax.random.wrap_key_data(jnp.asarray(state["key"]), impl=str(impl.item())))
        records = Records(np.array(state["features"], copy=True), np.array(state["categories"], copy=True))
        _check_tape(bank, records)
        if records.features.dtype != bank.posterior.mu.dtype or (len(records.features) == 0) != (step == 0):
            raise ValueError("reduction history disagrees with its model or step")
        return cls(key, int(step), records)


class Decision(NamedTuple):
    receiver: int
    donor: int
    before: float
    proposed: float | None
    active_before: int
    active_after: int
    status: str
    minimum_before: float
    minimum_proposed: float | None
    full_rescore: bool


class Reduction(NamedTuple):
    bank: ComponentBank
    state: ReductionState
    status: str
    initial_score: float | None
    final_score: float | None
    decisions: tuple[Decision, ...]


def _check_tape(bank, records):
    b = bank.posterior
    validate_records(records, b.mu.shape[1], tuple(h.shape[1] for h in b.categories))


def _initial(bank, dormant_initial):
    return initialized_view(bank.posterior, bank.generations, dormant_initial)


def _draw(bank, view, key, n):
    """Draw observations at posterior mean parameters, using E[Sigma] for Gaussian noise."""
    mass, _, excess = mixing(bank.posterior, bank.reset_prior)
    weight = np.where(excess > 0, mass, 0.)
    if not np.any(weight):
        raise ValueError("no positive mixing evidence for generated observations")
    keys = jax.vmap(lambda k: jax.random.split(k, 9))(jax.random.split(key, n + 1)[1:])
    p = jnp.asarray(weight / weight.sum(), bank.posterior.mu.dtype)
    z = jax.vmap(lambda k: jax.random.choice(k, len(p), p=p))(keys[:, 1])
    b = sampling_view(jax.tree.map(lambda x: x[z], view))
    mu = b.mu + b.carry.mu
    d = mu.shape[1]
    denominator = (b.nu - d - 1) + b.carry.nu
    if not bool(jnp.all(denominator > 0)):
        raise ValueError("posterior-mean covariance is undefined for a sampled component")
    cov = (b.psi + b.carry.psi) / denominator[:, None, None]
    minimum = jnp.linalg.eigvalsh(cov)[:, 0]
    # The released sampler corrects a negative eigenvalue of the projected
    # sampling covariance by twice its distance from machine epsilon.
    shift = jnp.where(minimum < 0, 2 * (jnp.finfo(mu.dtype).eps - minimum), 0)
    cov += shift[:, None, None] * jnp.eye(d, dtype=mu.dtype)
    samples = jax.vmap(jax.random.multivariate_normal)(keys[:, 2], mu, cov)
    labels = []
    for i, (alpha, low) in enumerate(zip(b.categories, b.carry.categories, strict=True)):
        count = alpha + low
        probabilities = count / count.sum(axis=1, keepdims=True)
        labels.append(jax.vmap(lambda k, prob: jax.random.choice(k, len(prob), p=prob))
                      (keys[:, 3+i], probabilities))
    records = Records(np.asarray(samples), np.stack(labels, axis=1).astype(np.int32))
    _check_tape(bank, records)
    return records


def _pairs(tables, active, key, limit):
    live = np.flatnonzero(active)
    n = min(2 * limit, max(0, len(live) - 1))
    if n == 0:
        return np.empty((0, 2), dtype=np.intp)
    sources = live[np.asarray(jax.random.choice(key, len(live), shape=(n,), replace=False))]
    donors = soft_neighbors(tables, sources)
    eligible = np.ones(n, bool)
    for i in (0, 1, -2, -1):
        mode = tables.probability[i].argmax(axis=1)
        eligible &= mode[sources] == mode[donors]
    return np.column_stack((sources[eligible], donors[eligible]))[:limit]


def _model_scores(model):
    minima = np.asarray(_minima(model))
    return Tables.build(model, minimum=minima.min()), minima


def _score(outputs, mass, base, excess):
    value = float(outputs.mean() - mixing_kl(mass, base, excess))
    if not np.isfinite(value):
        raise ValueError("nonfinite reduction score")
    return value


def _local_outputs(cache, pair, new, evidence):
    winners = cache.trial_winners(pair, new)
    labels = cache.tape.categories
    output = cache.tables.output(labels, winners, evidence)
    for col, slot in enumerate(pair):
        use = winners == slot
        output[use] = new.output(labels[use], np.full(use.sum(), col, dtype=int), evidence[pair])
    return output


def _pass(bank, records, pairs, tables, minima, dormant_initial):
    if len(records.features) == 0:
        raise ValueError("a reduction score requires generated records")
    k = len(bank.logical_ids)
    pairs = np.asarray(pairs)
    if pairs.shape[1:] != (2,) or pairs.dtype.kind not in "iu" or np.any(pairs < 0) or np.any(pairs >= k):
        raise ValueError("merge pairs must contain two in-range integer slot indices")
    mass, base, excess = mixing(bank.posterior, bank.reset_prior)
    active = np.asarray(bank.posterior.active)
    mutable = pairs[active[pairs].all(axis=1)].ravel()
    cache = Competition(tables, records, mutable, active=active)
    scores = tables.output(records.categories, cache.winners(), excess > 0)
    current = _score(scores, mass, base, excess)
    initial = current
    count = int(active.sum())
    minimum = minima.min()
    log = []
    for pair in pairs:
        i, j = map(int, pair)
        receiver, donor = bank.reference(i), bank.reference(j)
        before, n_before, old_minimum = current, count, float(minimum)
        proposed = candidate_minimum = None
        full_rescore = False
        if receiver is None or donor is None:
            status = "inactive"
        else:
            candidate = bank.propose_merge(receiver, donor)
            status = candidate.status.value
        if status == SlotStatus.OK.value:
            # Both rows were live. After the merge the donor uses its reset prior.
            raw = jax.tree.map(lambda x: x[pair], candidate.bank.posterior)
            prior = jax.tree.map(lambda x: x[pair], bank.reset_prior)
            next_mass, next_excess = mass.copy(), excess.copy()
            next_mass[pair], _, next_excess[pair] = mixing(raw, prior)
            try:
                next_minima = minima.copy()
                next_minima[pair] = np.asarray(_minima(raw))
                candidate_minimum = next_minima.min()
                full_rescore = bool(candidate_minimum != minimum)
                if full_rescore:
                    model = _initial(candidate.bank, dormant_initial)
                    full = Tables.build(model, minimum=candidate_minimum)
                    trial = Competition(full, records, mutable, active=np.asarray(candidate.bank.posterior.active))
                    output = full.output(records.categories, trial.winners(), next_excess > 0)
                else:
                    new = Tables.build(raw, minimum=candidate_minimum)
                    output = _local_outputs(cache, pair, new, next_excess > 0)
                proposed = _score(output, next_mass, base, next_excess)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                status = "invalid_projection"
            else:
                if proposed >= current:
                    bank, current, count = candidate.bank, proposed, count - 1
                    mass, excess, minima, minimum = next_mass, next_excess, next_minima, candidate_minimum
                    if full_rescore:
                        cache = trial
                    else:
                        cache.accept(pair, new)
                    status = "accepted"
                else:
                    status = "rejected"
        log.append(Decision(i, j, before, proposed, n_before, count, status, old_minimum,
                            None if candidate_minimum is None else float(candidate_minimum), full_rescore))
    return bank, initial, current, tuple(log)


def reduce_on_tape(bank, records, pairs, *, dormant_initial=None):
    """Evaluate a prescribed sequence on complete held-out records with no random draws.

    Pair eligibility is the caller's input here. Both participants must
    still be live at evaluation. Returns (bank, initial score, final score,
    decisions), using the same sequential acceptance path as the scheduler.
    """
    _check_tape(bank, records)
    tables, minima = _model_scores(_initial(bank, dormant_initial))
    return _pass(bank, records, pairs, tables, minima, dormant_initial)


def reduce_interactions(bank, state, *, step, dormant_initial=None, sample_count=2000, pair_count=2000):
    """Run a scheduled reduction after every 500 completed actions.

    Each invocation adds sample_count draws to the accumulated tape, samples
    at most twice pair_count distinct live sources and keeps at most pair_count
    modal-compatible pairs. The first maximum wins a context tie. Pair limits
    and generated sample counts can be reduced for controlled fixtures.

    Returns ok on a completed pass. not_due, empty_support and invalid_model
    preserve the original bank, history and key. Decisions distinguish accepted,
    rejected, inactive, invalid_projection and host lifecycle status values.
    full_rescore records a trial with a changed global likelihood repair.
    """
    if not isinstance(step, (int, np.integer)) or step < state.last_step or step < 0:
        raise ValueError("invalid completed-action step")
    if (not isinstance(sample_count, (int, np.integer)) or sample_count < 1
            or not isinstance(pair_count, (int, np.integer)) or pair_count < 0):
        raise ValueError("invalid reduction sample or pair count")
    _check_tape(bank, state.records)
    if step == state.last_step or step % 500:
        return Reduction(bank, state, "not_due", None, None, ())
    _, _, excess = mixing(bank.posterior, bank.reset_prior)
    if not np.any(excess > 0):
        return Reduction(bank, state, "empty_support", None, None, ())
    key, draw_key = jax.random.split(state.key)
    key, pair_key = jax.random.split(key)
    model = _initial(bank, dormant_initial)
    try:
        batch = _draw(bank, model, draw_key, sample_count)
        records = Records(*(np.concatenate((old, new), axis=0)
                            for old, new in zip(state.records, batch, strict=True)))
        tables, minima = _model_scores(model)
        pairs = _pairs(tables, np.asarray(bank.posterior.active), pair_key, pair_count)
        reduced, before, after, decisions = _pass(bank, records, pairs, tables, minima, dormant_initial)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return Reduction(bank, state, f"invalid_model: {exc}", None, None, ())
    return Reduction(reduced, ReductionState(key, int(step), records), "ok", before, after, decisions)

"""Fixed-shape slot birth and merge proposals for JIT, scan and vmap.

Initialize with `from_host(component_bank)`. `birth` chooses the first free
slot. `reference(bank, slot)` returns a logical ID and generation that go
stale when the occupant leaves. IDs retain the host's signed int64 range with
JAX x64 disabled. `propose_merge(bank, receiver, donor)` transfers the
donor's excess over its reset prior, including low words and categorical
counts. A successful proposal has status OK. Keep its new bank if the model
accepts the proposal; otherwise keep the original. FULL and ID_EXHAUSTED
refer to allocation, STALE_REFERENCE and SAME_COMPONENT to the chosen pair,
and INVALID_POSTERIOR to invalid parameters. All refusals return the input
bank and a sentinel reference with slot -1.

NUMERICAL_RANGE means that two-word device arithmetic could not resolve the
requested change. For a model-requested merge, retry that pair through the
exact host operation on the unchanged snapshot. At the host boundary::

    proposal = propose_merge(bank, receiver, donor)
    if int(proposal.status) == Status.NUMERICAL_RANGE:
        original = to_host(proposal.bank)
        retry = original.propose_merge(original.reference(int(receiver.slot)),
                                       original.reference(int(donor.slot)))
        if retry.status == SlotStatus.OK and model_accepts(retry.bank):
            bank = from_host(retry.bank)

Here SlotStatus comes from cffg.component_bank and model_accepts is the
caller's ordinary acceptance test. A host-only model can call the exact
ComponentBank.propose_merge directly. The exact conversion preserves both
parameter words and all identity metadata. The retry crosses the host
boundary for the bank; its merger arithmetic uses just the selected rows.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from cffg._compensated import _two_sum
from cffg.component_bank import ComponentBank, _snapshot
from cffg.mixed_scoring import validate_bank
from cffg.mixed_state import MixedBank, MixedCarry


class Status(IntEnum):
    OK = 0
    FULL = 1
    STALE_REFERENCE = 2
    SAME_COMPONENT = 3
    INVALID_POSTERIOR = 4
    ID_EXHAUSTED = 5
    NUMERICAL_RANGE = 6


class DeviceBank(NamedTuple):
    prior: MixedBank
    posterior: MixedBank
    logical_ids: jax.Array  # [K,2], most significant uint32 word first
    generations: jax.Array
    next_id: jax.Array


class Reference(NamedTuple):
    slot: jax.Array
    logical_id: jax.Array
    generation: jax.Array


class Result(NamedTuple):
    bank: DeviceBank
    status: jax.Array
    component: Reference


def _words(value):
    ids = np.asarray(value, dtype=np.uint64)
    return jnp.asarray(np.stack((ids >> np.uint64(32), ids.astype(np.uint32)),
                                axis=-1).astype(np.uint32))


def from_host(bank: ComponentBank) -> DeviceBank:
    """Copy lifecycle metadata to the device; share immutable learner arrays."""
    return DeviceBank(bank.reset_prior, bank.posterior, _words(bank.logical_ids),
                      _words(bank.generations), _words(bank.next_id))


def to_host(bank: DeviceBank) -> ComponentBank:
    """Validate the device result through the host lifecycle's exact codec."""
    def integer(words):
        words = np.asarray(words)
        if words.dtype != np.uint32 or words.shape[-1:] != (2,):
            raise ValueError("an identifier must contain two uint32 words")
        value = (words[..., 0].astype(np.uint64) << np.uint64(32)) | words[..., 1]
        if np.any(value > np.iinfo(np.int64).max):
            raise ValueError("device identifier exceeds host int64 range")
        return value.astype(np.int64)

    state = {"version": np.int32(1), "logical_ids": integer(bank.logical_ids),
             "generations": integer(bank.generations), "next_id": integer(bank.next_id)}
    for name, mixed in (("prior", bank.prior), ("posterior", bank.posterior)):
        state.update({f"{name}/{k}": v for k, v in _snapshot(mixed, exact=True).items()})
    return ComponentBank.from_state_dict(state)


def _increment(words):
    low = words[1] + jnp.uint32(1)
    return jnp.stack((words[0] + (low == 0).astype(jnp.uint32), low))


def _missing():
    return Reference(jnp.int32(-1), jnp.zeros(2, jnp.uint32), jnp.zeros(2, jnp.uint32))


def reference(bank: DeviceBank, slot) -> Reference:
    """Read a live reference, or return a sentinel for a free or absent slot."""
    slot = jnp.asarray(slot)
    if slot.ndim or not jnp.issubdtype(slot.dtype, jnp.signedinteger):
        raise ValueError("slot must be a scalar signed integer")
    k = bank.posterior.active.shape[0]
    i = jnp.clip(slot, 0, k - 1).astype(jnp.int32)
    valid = (slot >= 0) & (slot < k) & bank.posterior.active[i]
    ref = Reference(i, bank.logical_ids[i], bank.generations[i])
    return jax.tree.map(lambda yes, no: jnp.where(valid, yes, no), ref, _missing())


def birth(bank: DeviceBank) -> Result:
    """Activate the first free slot, preserving prior state on failure."""
    free = ~bank.posterior.active
    i = jnp.argmax(free).astype(jnp.int32)
    exhausted = jnp.all(bank.next_id == jnp.array([0x7fffffff, 0xffffffff], jnp.uint32))
    status = jnp.where(~jnp.any(free), Status.FULL,
                       jnp.where(exhausted, Status.ID_EXHAUSTED, Status.OK)).astype(jnp.int32)

    def activate():
        prior = jax.tree.map(lambda x: x[i][None], bank.prior)
        dormant = jax.tree.map(lambda x: x[i][None], bank.posterior)
        prior = prior._replace(active=jnp.ones(1, bool))
        dormant = dormant._replace(active=prior.active)
        clean = validate_bank(prior) & jnp.all(bank.logical_ids[i] == 0)
        for initial, saved in zip(jax.tree.leaves(prior), jax.tree.leaves(dormant), strict=True):
            clean &= jnp.all(initial == saved)

        def apply():
            gen = _increment(bank.generations[i])
            posterior = bank.posterior._replace(active=bank.posterior.active.at[i].set(True))
            next_bank = bank._replace(posterior=posterior, logical_ids=bank.logical_ids.at[i].set(bank.next_id),
                                      generations=bank.generations.at[i].set(gen), next_id=_increment(bank.next_id))
            return Result(next_bank, status, Reference(i, bank.next_id, gen))
        return jax.lax.cond(clean, apply, lambda: Result(bank, jnp.int32(Status.INVALID_POSTERIOR), _missing()))

    return jax.lax.cond(status == Status.OK, activate, lambda: Result(bank, status, _missing()))


class _Number(NamedTuple):
    hi: jax.Array
    lo: jax.Array
    error: jax.Array  # bound on discarded terms, in the units of this value


def _number(hi, lo=None):
    zero = jnp.zeros_like(hi)
    return _Number(hi, zero if lo is None else lo, zero)


def _add(a, b):
    hi, tail = _two_sum(a.hi, b.hi)
    small, lost = _two_sum(a.lo, b.lo)
    tail, correction = _two_sum(tail, small)
    hi, lo = _two_sum(hi, tail)
    lo, remainder = _two_sum(lo, correction)
    hi, lo = _two_sum(hi, lo)
    return _Number(hi, lo, a.error + b.error + jnp.abs(lost) + jnp.abs(remainder))


def _sub(a, b):
    return _add(a, _Number(-b.hi, -b.lo, b.error))


_round = jax.lax.optimization_barrier


def _split_mantissa(x):
    """Round to the upper half of the significand for Dekker's product residual."""
    word = jnp.uint32 if x.dtype == jnp.float32 else jnp.uint64
    bits = 12 if x.dtype == jnp.float32 else 27
    raw = jax.lax.bitcast_convert_type(x, word)
    upper = (raw + word(1 << (bits - 1))) & ~word((1 << bits) - 1)
    high = jax.lax.bitcast_convert_type(upper, x.dtype)
    return high, _round(x - high)


def _mul(a, b):
    ah, al = _split_mantissa(a.hi)
    bh, bl = _split_mantissa(b.hi)
    prod = _round(a.hi * b.hi)
    err = _round(_round(ah * bh) - prod)
    err = _round(err + _round(ah * bl))
    err = _round(err + _round(al * bh))
    err = _round(err + _round(al * bl))
    finfo = jnp.finfo(prod.dtype)
    # Near underflow the split products can lose the residual of the high
    # product. Decline that range instead of treating a flushed zero as exact.
    normal = (a.hi == 0) | (b.hi == 0) | (jnp.abs(prod) >= finfo.tiny / finfo.eps**2)
    result = _Number(prod, err, jnp.where(normal, jnp.zeros_like(prod), jnp.inf))
    for left, right in ((a.hi, b.lo), (a.lo, b.hi), (a.lo, b.lo)):
        cross = _round(left * right)
        nonzero = (left != 0) & (right != 0)
        loss = jnp.where(nonzero, finfo.eps * jnp.abs(cross) + finfo.tiny, 0)
        result = _add(result, _Number(cross, jnp.zeros_like(cross), loss))
    propagated = ((jnp.abs(a.hi) + jnp.abs(a.lo) + a.error) * b.error
                  + (jnp.abs(b.hi) + jnp.abs(b.lo)) * a.error)
    return result._replace(error=result.error + propagated)


def _div(a, b):
    q = _round(a.hi / b.hi)
    residual = _sub(a, _mul(b, _number(q)))
    tail = _round(residual.hi / b.hi)
    result = _add(_number(q), _number(tail))
    denominator = jnp.abs(b.hi) - jnp.abs(b.lo) - b.error
    eps = jnp.finfo(q.dtype).eps
    error = ((residual.error + jnp.abs(residual.lo)
              + jnp.abs(tail) * (jnp.abs(b.lo) + b.error)
              + 2 * eps * jnp.abs(residual.hi)) / denominator)
    tiny = jnp.finfo(q.dtype).tiny
    quotient_normal = ((a.hi == 0) | (jnp.abs(q) >= tiny)) & (
        (residual.hi == 0) | (jnp.abs(tail) >= tiny))
    valid = (denominator > 0) & quotient_normal
    return result._replace(error=jnp.where(valid, result.error + error, jnp.inf))


def _outer(a):
    return _mul(jax.tree.map(lambda x: x[:, None], a), jax.tree.map(lambda x: x[None, :], a))


def _transfer(receiver, donor, reset):
    increment = _sub(donor, reset)
    updated = _add(receiver, increment)
    return updated, increment


def _resolved(value, increment):
    # Reserve headroom for the source-precision accumulation of the positive
    # error estimates. A proposal must resolve both its result and its change.
    scale = jnp.minimum(jnp.abs(value.hi) + jnp.abs(value.lo),
                        jnp.abs(increment.hi) + jnp.abs(increment.lo))
    tolerance = 4 * jnp.finfo(value.hi.dtype).eps * scale
    return jnp.all(jnp.isfinite(value.hi) & jnp.isfinite(value.lo)
                   & jnp.isfinite(value.error) & (value.error <= tolerance))


def _row_fields(row):
    highs = MixedCarry(*(getattr(row, field) for field in MixedCarry._fields))
    return jax.tree.map(_number, highs, row.carry)


def _has_subnormal(values):
    """Inspect source bits before a backend can flush arithmetic inputs to zero."""
    def field(x):
        word = jnp.uint32 if x.dtype == jnp.float32 else jnp.uint64
        raw = jax.lax.bitcast_convert_type(x, word)
        magnitude = raw & word(np.iinfo(np.dtype(word)).max >> 1)
        minimum_normal = word(1 << np.finfo(x.dtype).nmant)
        return jnp.any((magnitude != 0) & (magnitude < minimum_normal))
    flags = [field(x) for x in jax.tree.leaves(values) if jnp.issubdtype(x.dtype, jnp.floating)]
    return jnp.any(jnp.stack(flags))


def _merge_rows(receiver, donor, reset):
    r, d, p = map(_row_fields, (receiver, donor, reset))
    def count(a, b, c):
        value, increment = _transfer(a, b, c)
        return value, (_resolved(value, increment)
                       & jnp.all((increment.hi > 0) | ((increment.hi == 0) & (increment.lo >= 0))))

    k, ok_k = count(r.kappa, d.kappa, p.kappa)
    nu, ok_n = count(r.nu, d.nu, p.nu)
    alpha, ok_a = count(r.alpha, d.alpha, p.alpha)
    ok = ok_k & ok_n & ok_a
    categories = []
    for three in zip(r.categories, d.categories, p.categories, strict=True):
        value, valid = count(*three)
        categories.append(value)
        ok &= valid

    dd = _sub(d.mu, r.mu)
    pd = _sub(p.mu, r.mu)
    delta = _div(_sub(_mul(d.kappa, dd), _mul(p.kappa, pd)), k)
    mu = _add(r.mu, delta)
    scatter = _sub(_mul(d.kappa, _outer(dd)), _mul(p.kappa, _outer(pd)))
    increment = _add(_sub(d.psi, p.psi), _sub(scatter, _mul(k, _outer(delta))))
    psi = _add(r.psi, increment)
    gaussian_unchanged = jnp.asarray(True)
    for name in ("kappa", "mu", "psi", "nu"):
        before, after = getattr(reset, name), getattr(donor, name)
        before_low, after_low = getattr(reset.carry, name), getattr(donor.carry, name)
        gaussian_unchanged &= jnp.all(before == after) & jnp.all(before_low == after_low)
    ok &= gaussian_unchanged | (_resolved(mu, delta) & _resolved(psi, increment))
    mu = jax.tree.map(lambda old, new: jnp.where(gaussian_unchanged, old, new), r.mu, mu)
    psi = jax.tree.map(lambda old, new: jnp.where(gaussian_unchanged, old, new), r.psi, psi)
    values = MixedCarry(mu, k, psi, nu, alpha, tuple(categories))
    high = jax.tree.map(lambda v: v.hi, values, is_leaf=lambda x: isinstance(x, _Number))
    low = jax.tree.map(lambda v: v.lo, values, is_leaf=lambda x: isinstance(x, _Number))
    merged = MixedBank(*high, receiver.active, carry=low)
    inputs = jax.tree.map(lambda *xs: jnp.stack(xs), receiver, donor, reset)
    valid_inputs = validate_bank(inputs._replace(active=jnp.ones(3, dtype=bool)))
    ok &= ~_has_subnormal(inputs)
    proper = validate_bank(jax.tree.map(lambda x: x[None], merged))
    status = jnp.where(~valid_inputs, Status.INVALID_POSTERIOR,
                       jnp.where(~ok, Status.NUMERICAL_RANGE,
                                 jnp.where(proper, Status.OK, Status.INVALID_POSTERIOR)))
    return merged, status.astype(jnp.int32)


def _live(bank, ref):
    current = reference(bank, ref.slot)
    for words in (ref.logical_id, ref.generation):
        if words.shape != (2,) or words.dtype != jnp.uint32:
            raise ValueError("a device reference must contain two uint32 words per identifier")
    return ((current.slot >= 0) & jnp.all(ref.logical_id == current.logical_id)
            & jnp.all(ref.generation == current.generation))


def propose_merge(bank: DeviceBank, receiver: Reference, donor: Reference) -> Result:
    """Propose a prior-subtracted NIW/Dirichlet merge, or return the original state.

    NUMERICAL_RANGE declines subnormal input words, a nonzero high-word
    product below finfo.tiny / finfo.eps**2, or an estimated discarded term
    above four source epsilons times the smaller of the result and change.
    The exact selected-row host merge remains available for these inputs.
    INVALID_POSTERIOR identifies invalid component parameters.
    """
    live = _live(bank, receiver) & _live(bank, donor)
    status = jnp.where(~live, Status.STALE_REFERENCE,
                       jnp.where(receiver.slot == donor.slot, Status.SAME_COMPONENT, Status.OK)).astype(jnp.int32)
    def calculate():
        i, j = receiver.slot, donor.slot
        r = jax.tree.map(lambda x: x[i], bank.posterior)
        d = jax.tree.map(lambda x: x[j], bank.posterior)
        p = jax.tree.map(lambda x: x[j], bank.prior)
        merged, outcome = _merge_rows(r, d, p)

        def accept():
            dormant = p._replace(active=jnp.asarray(False))
            post = jax.tree.map(lambda x, a, b: x.at[i].set(a).at[j].set(b),
                                bank.posterior, merged, dormant)
            new = bank._replace(posterior=post, logical_ids=bank.logical_ids.at[j].set(0))
            return Result(new, outcome, receiver)

        return jax.lax.cond(outcome == Status.OK, accept, lambda: Result(bank, outcome, _missing()))
    return jax.lax.cond(status == Status.OK, calculate, lambda: Result(bank, status, _missing()))

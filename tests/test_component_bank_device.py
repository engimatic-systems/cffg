"""Compiled lifecycle parity with exact selected-row host mergers and snapshots."""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import component_bank_device as device
from cffg.component_bank import ComponentBank, SlotStatus
from cffg.mixed_learning import MixedBank, MixedBatch
from cffg.mixed_state import MixedCarry
from test_component_bank import assert_same, exact, prior


_merge = jax.jit(device.propose_merge)
_birth = jax.jit(device.birth)


def array_equal(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)


def merge(b, i=0, j=1):
    return _merge(b, device.reference(b, i), device.reference(b, j))


def compare_to_host(before):
    host_result = before.propose_merge(before.reference(0), before.reference(1))
    assert host_result.status == SlotStatus.OK
    d = device.from_host(before)
    result = merge(d)
    assert result.status == device.Status.OK
    host, compiled = host_result.bank, device.to_host(result.bank)
    assert host.logical_ids == compiled.logical_ids and host.generations == compiled.generations
    np.testing.assert_array_equal(host.posterior.active, compiled.posterior.active)
    for key in MixedCarry._fields:
        hs = jax.tree.leaves(getattr(host.posterior, key))
        ds = jax.tree.leaves(getattr(compiled.posterior, key))
        hls = jax.tree.leaves(getattr(host.posterior.carry, key))
        dls = jax.tree.leaves(getattr(compiled.posterior.carry, key))
        for h, d, hl, dl in zip(hs, ds, hls, dls, strict=True):
            got, want = exact(d[0], dl[0]), exact(h[0], hl[0])
            err = np.asarray(abs(got - want), dtype=float)
            eps = np.finfo(d.dtype).eps
            assert np.all(err <= 64 * eps**2 * np.maximum(1, np.abs(np.asarray(want, float))))
            np.testing.assert_array_equal(h[1:], d[1:])
            np.testing.assert_array_equal(hl[1:], dl[1:])
    return compiled


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compiled_merge_real_fractional_update(dtype):
    p = prior(dtype)
    p.mu[1] += [3, -1]
    p.kappa[1] = 3
    p.psi[1] = [[2, -.5], [-.5, 4]]
    base = ComponentBank(p)
    x = MixedBatch(np.array([[4097, -6], [4094, -7]], dtype),
                   np.array([[[.5, -.25], [-.25, 1]], [[1, .5], [.5, 2]]], dtype),
                   np.ones(2, bool), (np.array([0, 1]), np.array([-1, 2])))
    b = base.update(x, np.array([[.25, .75, 0], [0, 1, 0]], dtype)).bank
    compiled = compare_to_host(b)
    d = device.from_host(b)
    eager = device.propose_merge(d, device.reference(d, 0), device.reference(d, 1))
    assert eager.status == device.Status.OK
    array_equal(eager.bank.posterior, compiled.posterior)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compiled_transfer_of_sub_ulp_mass_and_gaussian_moments(dtype):
    p = prior(dtype)
    large = dtype(2 ** np.finfo(dtype).nmant)
    p.kappa[:] = p.nu[:] = p.alpha[:] = large
    p.psi[:] = np.eye(2) * large
    for head in p.categories:
        head[:] = large
    x = MixedBatch(np.array([[4097, -7]], dtype), np.array([[[.5, -.25], [-.25, .5]]], dtype),
                   np.array([True]), (np.array([1]), np.array([2])))
    base = ComponentBank(p)
    b = base.update(x, np.array([[.25, .75, 0]], dtype)).bank
    result = compare_to_host(b).posterior
    assert exact(result.kappa[0], result.carry.kappa[0]) == int(large) + 1
    # Restore the device output's exact codec and repeat the next transaction.
    old = device.from_host(compare_to_host(b))
    restored = device.from_host(ComponentBank.from_state_dict(device.to_host(old).state_dict(exact=True)))
    array_equal(_birth(old), _birth(restored))


def test_empty_full_and_id_reuse_across_uint32_rollover():
    empty = device.from_host(ComponentBank(prior(active=(False, False))))
    first = _birth(empty)
    assert first.status == device.Status.OK and first.component.slot == 0
    both = _birth(first.bank)
    full = _birth(both.bank)
    assert full.status == device.Status.FULL
    array_equal(full.bank, both.bank)
    record = device.to_host(both.bank).state_dict()
    base = 2**32 - 1
    record.update(logical_ids=np.array([1, base-1], np.int64),
                  generations=np.array([1, base-2], np.int64), next_id=np.int64(base))
    host = ComponentBank.from_state_dict(record)
    bank = device.from_host(host)
    stale = device.reference(bank, 1)

    def advance(b, _):
        proposed = device.propose_merge(b, device.reference(b, 0), device.reference(b, 1))
        born = device.birth(proposed.bank)
        return born.bank, (proposed.status, born.status)
    bank, statuses = jax.jit(lambda b: jax.lax.scan(advance, b, None, length=2))(bank)
    np.testing.assert_array_equal(statuses, np.zeros((2, 2)))
    resumed = device.to_host(bank)
    for _ in range(2):
        host = host.propose_merge(host.reference(0), host.reference(1)).bank.birth().bank
    assert_same(resumed.state_dict(exact=True), host.state_dict(exact=True))
    assert resumed.reference(1).logical_id == 2**32
    vacant = host.propose_merge(host.reference(0), host.reference(1)).bank.state_dict()
    limit = np.iinfo(np.int64).max
    vacant["next_id"] = np.int64(limit)
    vacant["generations"] = np.array([1, limit - 2], np.int64)
    exhausted = device.from_host(ComponentBank.from_state_dict(vacant))
    denied = _birth(exhausted)
    assert denied.status == device.Status.ID_EXHAUSTED
    array_equal(denied.bank, exhausted)
    reject = _merge(bank, device.reference(bank, 0), stale)
    assert reject.status == device.Status.STALE_REFERENCE
    array_equal(reject.bank, bank)


def test_mixed_batch_keeps_each_rejected_bank_intact():
    b = device.from_host(ComponentBank(prior()))
    bad_cov = b.posterior.psi.at[1].set(jnp.ones((2, 2), b.posterior.mu.dtype))
    invalid = b._replace(posterior=b.posterior._replace(psi=bad_cov))
    stale = device.Reference(jnp.int32(1), jnp.array([0, 99], jnp.uint32),
                             jnp.array([0, 1], jnp.uint32))
    r, d = device.reference(b, 0), device.reference(b, 1)
    banks, donors = (b, b, b, invalid), (d, stale, r, d)
    stack = lambda *values: jax.tree.map(lambda *x: jnp.stack(x), *values)
    result = jax.jit(jax.vmap(device.propose_merge))(stack(*banks), stack(*[r]*4), stack(*donors))
    np.testing.assert_array_equal(result.status, [device.Status.OK, device.Status.STALE_REFERENCE,
                                                 device.Status.SAME_COMPONENT, device.Status.INVALID_POSTERIOR])
    for i, (bank, donor) in enumerate(zip(banks, donors)):
        single = _merge(bank, r, donor)
        array_equal(jax.tree.map(lambda x: x[i], result), single)
        if i:
            array_equal(single.bank, bank)


@pytest.mark.parametrize(("dtype", "power"), [(np.float32, -59), (np.float64, -495)])
def test_exponent_range_declines_valid_host_proposal_atomically(dtype, power):
    many = lambda x: np.broadcast_to(np.asarray(x, dtype), (2, *np.shape(x))).copy()
    bank = ComponentBank(MixedBank(many([0]), many(1), many([[1]]), many(4),
                                   many(2), (), np.ones(2, bool)))
    x = MixedBatch(np.array([[2.**power]], dtype), np.zeros((1,1,1), dtype), np.ones(1,bool), ())
    before = bank.update(x, np.array([[0, 1]],dtype)).bank
    host = before.propose_merge(before.reference(0), before.reference(1))
    assert host.status == SlotStatus.OK
    b = device.from_host(before)
    result = merge(b)
    assert result.status == device.Status.NUMERICAL_RANGE
    array_equal(result.bank, b)


@pytest.mark.slow
@pytest.mark.parametrize("capacity", [32, 500, 5000])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compiled_large_fixed_capacity(capacity, dtype):
    active = np.zeros(capacity, bool)
    active[:2] = True
    p = prior(dtype, active)
    # A declared posterior with one observed token and one unit of spread.
    record = ComponentBank(p).state_dict()
    for key in ("alpha", "kappa", "nu"):
        record["posterior/"+key][1] += 1
    record["posterior/psi"][1] += np.eye(2)
    record["posterior/category_0"][1, 0] += 1
    before = ComponentBank.from_state_dict(record)
    result = compare_to_host(before)
    born = _birth(device.from_host(result))
    assert born.component.slot == 1
    assert np.asarray(born.bank.logical_ids).shape == (capacity, 2)


def test_vmap_birth_keeps_full_or_contaminated_bank_unchanged():
    vacant = device.from_host(ComponentBank(prior(active=(True, False))))
    full = _birth(vacant).bank
    contaminated = vacant._replace(posterior=vacant.posterior._replace(
        alpha=vacant.posterior.alpha.at[1].add(1)))
    batch = jax.tree.map(lambda *xs: jnp.stack(xs), vacant, full, contaminated)
    result = jax.jit(jax.vmap(device.birth))(batch)
    np.testing.assert_array_equal(result.status, [device.Status.OK, device.Status.FULL,
                                                 device.Status.INVALID_POSTERIOR])
    for i, expected in enumerate((full, full, contaminated)):
        array_equal(jax.tree.map(lambda x: x[i], result.bank), expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_unresolved_prior_subtraction_uses_explicit_range_status(dtype):
    # Donor prior m=1, kappa=2 and one point at zero give true posterior
    # m=2/3 and psi=5/3. The receiver is already centered at that datum.
    # Source pairs for the two repeating fractions have independent tails.
    p = MixedBank(np.array([[0], [1]], dtype), np.array([1, 2], dtype),
                  np.ones((2, 1, 1), dtype), np.array([4, 4], dtype),
                  np.ones(2, dtype), (), np.ones(2, bool))
    record = ComponentBank(p).state_dict(exact=True)
    for name, location, value in (("mu", (1, 0), Fraction(2, 3)),
                                  ("psi", (1, 0, 0), Fraction(5, 3))):
        high = dtype(float(value))
        record["posterior/" + name][location] = high
        record["posterior/carry_" + name][location] = dtype(float(value - Fraction(float(high))))
    for name in ("kappa", "nu", "alpha"):
        record["posterior/" + name][1] += 1
    host = ComponentBank.from_state_dict(record)
    assert host.propose_merge(host.reference(0), host.reference(1)).status == SlotStatus.OK
    b = device.from_host(host)
    result = merge(b)
    assert result.status == device.Status.NUMERICAL_RANGE
    array_equal(result.bank, b)
    original = device.to_host(result.bank)
    retry = original.propose_merge(original.reference(0), original.reference(1))
    assert retry.status == SlotStatus.OK
    expected = host.propose_merge(host.reference(0), host.reference(1)).bank
    assert_same(retry.bank.state_dict(exact=True), expected.state_dict(exact=True))

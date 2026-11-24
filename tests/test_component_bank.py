"""Numerical and identity contracts for host component-bank transactions."""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.component_bank import ComponentBank, SlotStatus
from cffg.mixed_learning import MixedBank, MixedBatch, score
from cffg.mixed_state import MixedCarry


def prior(dtype=np.float32, active=(True, True, False)):
    k = len(active)
    def many(x):
        return np.broadcast_to(np.array(x, dtype=dtype), (k, *np.shape(x))).copy()

    return MixedBank(many([4096, -8]), many(2), many([[3, 1], [1, 2]]), many(4),
                     many(5), (many([2, 3]), many([1, 2, 1])), np.array(active))


def assert_same(a, b):
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        assert a[key].dtype == b[key].dtype


def exact(hi, lo):
    return np.vectorize(lambda h, l: Fraction(float(h)) + Fraction(float(l)),
                        otypes=[object])(np.asarray(hi), np.asarray(lo))


def words(bank, field, i):
    return exact(getattr(bank, field)[i], getattr(bank.carry, field)[i])


def test_empty_full_and_donor_generation_after_replay(tmp_path):
    p = prior(active=(False, False))
    empty = ComponentBank(p)
    p.mu[0, 0] = -123  # caller-owned storage cannot change the reset prior
    assert empty.reset_prior.mu[0, 0] == 4096
    empty = ComponentBank.from_state_dict(empty.state_dict())
    assert empty.reference(0) is None
    first = empty.birth()
    second = first.bank.birth()
    assert (first.component.slot, second.component.slot) == (0, 1)
    b = second.bank
    assert b.birth().status == SlotStatus.FULL and b.birth().bank is b
    assert b.propose_merge(first.component, first.component).status == SlotStatus.SAME_COMPONENT
    before = b.state_dict()
    path = tmp_path / "slots.npz"
    np.savez(path, **before)
    with np.load(path, allow_pickle=False) as state:
        restored = ComponentBank.from_state_dict(state)
    proposals = []
    for original in (b, restored):
        result = original.propose_merge(first.component, second.component)
        assert result.status == SlotStatus.OK
        assert result.component == first.component
        assert result.bank.reference(1) is None
        assert result.bank.propose_merge(first.component, second.component).status == SlotStatus.STALE_REFERENCE
        next_birth = result.bank.birth()
        new_ref = next_birth.component
        assert new_ref.slot == second.component.slot
        assert new_ref.generation == 2 and new_ref.logical_id == 3
        stale = next_birth.bank.propose_merge(first.component, second.component)
        assert stale.status == SlotStatus.STALE_REFERENCE and stale.bank is next_birth.bank
        proposals.append(next_birth.bank.state_dict())
    assert_same(before, b.state_dict())
    assert_same(*proposals)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_real_update_merge_subtracts_donor_prior_once(dtype):
    p = prior(dtype)
    # The donor's prior and its learned evidence must be separated.
    p.mu[1] += [3, -1]
    p.kappa[1] = 3
    p.nu[1] = 5
    p.psi[1] = [[2, -.5], [-.5, 5]]
    p.alpha[1] = 8
    p.categories[0][1] = [3, 7]
    base = ComponentBank(p)
    means = np.array([[4097, -6], [4094, -7], [np.nan, np.nan]], dtype)
    cov = np.array([[[.5, -.25], [-.25, 1]], [[1, .5], [.5, 2]], np.full((2, 2), np.nan)], dtype)
    x = MixedBatch(means, cov, np.array([True, True, False]),
                   (np.array([0, 1, -1]), np.array([-1, 2, 1])))
    weights = np.array([[.25, .75, 0], [0, 1, 0], [1, 0, 0]], dtype)
    learned = base.update(x, weights)
    assert learned.status == SlotStatus.OK
    s = learned.bank
    result = s.propose_merge(s.reference(0), s.reference(1))
    assert result.status == SlotStatus.OK
    post = result.bank.posterior
    # Receiver prior plus two independent unit-weight Gaussian measurements.
    m = (2 * p.mu[0].astype(float) + means[:2].astype(float).sum(axis=0)) / 4
    between = means[:2].astype(float) - m
    psi = p.psi[0] + cov[:2].sum(axis=0) + between.T @ between + 2*np.outer(p.mu[0] - m, p.mu[0] - m)
    np.testing.assert_allclose(np.asarray(words(post, "mu", 0), float), m, rtol=0, atol=2e-6)
    np.testing.assert_allclose(np.asarray(words(post, "psi", 0), float), psi, rtol=2e-6)
    assert words(post, "alpha", 0) == 8
    assert words(post, "kappa", 0) == 4 and words(post, "nu", 0) == 6
    np.testing.assert_array_equal(post.categories[0][0], [3, 4])
    np.testing.assert_array_equal(post.categories[1][0], [1, 3, 2])
    for key in MixedCarry._fields:
        for a, b in zip(jax.tree.leaves(getattr(post, key)),
                        jax.tree.leaves(getattr(s.reset_prior, key)), strict=True):
            np.testing.assert_array_equal(a[1], b[1])
    for a, b in zip(jax.tree.leaves(post.carry), jax.tree.leaves(s.reset_prior.carry), strict=True):
        np.testing.assert_array_equal(a[1], b[1])
    _, r, ok = score(post, x)
    assert ok
    np.testing.assert_array_equal(r, [[1, 0, 0]] * 3)
    bad = result.bank.update(x, weights)
    assert bad.status == SlotStatus.INVALID_POSTERIOR and bad.bank is result.bank


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exact_fraction_oracle_for_every_high_and_low_word(dtype):
    p = prior(dtype)
    large = dtype(2 ** np.finfo(dtype).nmant)
    p.kappa[:] = p.nu[:] = p.alpha[:] = large
    p.psi[:] = np.eye(2) * large
    for head in p.categories:
        head[:] = large
    zero = np.zeros_like
    carry = MixedCarry(zero(p.mu), zero(p.kappa) + dtype(.125),
                       zero(p.psi) + np.eye(2, dtype=dtype) * dtype(.125),
                       zero(p.nu) + dtype(.125), zero(p.alpha) + dtype(.125),
                       tuple(zero(head) + dtype(.125) for head in p.categories))
    base = ComponentBank(p._replace(carry=carry))
    x = MixedBatch(np.array([[4097, -7]], dtype), np.array([[[.5, -.25], [-.25, .5]]], dtype),
                   np.array([True]), (np.array([0]), np.array([2])))
    learned = base.update(x, np.array([[.25, .75, 0]], dtype))
    assert learned.status == SlotStatus.OK
    s = learned.bank
    # Both slots contain evidence below the spacing of their primary coordinates.
    assert np.any(np.asarray(s.posterior.carry.mu[:2]) != 0)
    restored = ComponentBank.from_state_dict(s.state_dict())
    result = restored.propose_merge(restored.reference(0), restored.reference(1))
    assert result.status == SlotStatus.OK
    post = result.bank.posterior
    # Independent raw natural-coordinate oracle using rational numbers.
    triplet = [(s.posterior, 0), (s.posterior, 1), (s.reset_prior, 1)]
    def combine(items):
        a, b, c = items
        return a + b - c
    def totals(field):
        return combine(words(b, field, i) for b, i in triplet)
    def scalar(value):
        return np.asarray(value, dtype=object).item()
    k = scalar(totals("kappa"))
    m = combine(words(b, "kappa", i) * words(b, "mu", i) for b, i in triplet) / k
    moment = combine(words(b, "psi", i) + words(b, "kappa", i) * np.outer(
        words(b, "mu", i), words(b, "mu", i)) for b, i in triplet)
    expected = MixedCarry(m, k, moment - k * np.outer(m, m), totals("nu"), totals("alpha"),
                          tuple(combine(exact(b.categories[h][i], b.carry.categories[h][i])
                                        for b, i in triplet) for h in range(len(p.categories))))
    row = jax.tree.map(lambda a: a[0], post)
    primary = MixedCarry(*(getattr(row, name) for name in MixedCarry._fields))
    for hi, lo, target in zip(jax.tree.leaves(primary), jax.tree.leaves(row.carry),
                              jax.tree.leaves(expected), strict=True):
        value = exact(hi, lo)
        # Two rounded source words can leave at most one low-word ULP here.
        for actual, want, low in zip(np.ravel(value), np.ravel(target), np.ravel(lo), strict=True):
            assert abs(actual - want) <= Fraction(float(abs(np.spacing(low))))
    uninterrupted = s.propose_merge(s.reference(0), s.reference(1))
    assert_same(uninterrupted.bank.birth().bank.state_dict(), result.bank.birth().bank.state_dict())


def test_snapshot_rejects_missing_carry_reused_ids_and_contaminated_dormant_slot():
    b = ComponentBank(prior())
    for key, value in (("posterior/version", np.int32(2)), ("next_id", np.int64(5)),
                       ("logical_ids", np.array([1, 1, 0], np.int64)),
                       ("generations", np.array([0, 2, 0], np.int64))):
        with pytest.raises(ValueError):
            ComponentBank.from_state_dict({**b.state_dict(), key: value})
    state = b.state_dict()
    state["posterior/carry_alpha"][0] = -np.spacing(np.float32(5)) / 4
    with pytest.raises(ValueError, match="below its reset prior"):
        ComponentBank.from_state_dict(state)
    state = b.state_dict()
    del state["prior/carry_psi"]
    with pytest.raises(ValueError):
        ComponentBank.from_state_dict(state)
    state = b.state_dict()
    state["posterior/category_0"][2, 0] += 1
    with pytest.raises(ValueError, match="dormant"):
        ComponentBank.from_state_dict(state)


@pytest.mark.parametrize("capacity", [32, 500, 5000])
def test_lifecycle_preserves_fixed_pytree_shapes_at_capacity(capacity):
    active = np.zeros(capacity, bool)
    active[[0, -1]] = True
    b = ComponentBank(prior(active=active))
    result = b.propose_merge(b.reference(0), b.reference(capacity-1))
    assert result.status == SlotStatus.OK
    born = result.bank.birth()
    assert born.component.slot == 1
    before, tree = jax.tree.flatten(b.posterior)
    after, next_tree = jax.tree.flatten(born.bank.posterior)
    assert tree == next_tree
    assert [(a.shape, a.dtype) for a in before] == [(a.shape, a.dtype) for a in after]


def test_invalid_prior_and_unrepresentable_merge_are_atomic():
    p = prior()
    p.psi[0] = [[1, 2], [2, 1]]
    with pytest.raises(ValueError, match="prior"):
        ComponentBank(p)
    record = ComponentBank(prior()).state_dict()
    # Each valid posterior contains many observed category-only records.
    mass = np.float32(np.finfo(np.float32).max * .75)
    record["posterior/alpha"][:2] = mass
    record["posterior/category_0"][:2, 0] = mass
    b = ComponentBank.from_state_dict(record)
    result = b.propose_merge(b.reference(0), b.reference(1))
    assert result.status == SlotStatus.INVALID_POSTERIOR and result.bank is b
    assert_same(b.state_dict(), record)


def test_bank_refuses_staged_host_transaction():
    b = prior()
    with pytest.raises(TypeError, match="concrete host bank"):
        jax.jit(lambda inputs: ComponentBank(inputs).posterior)(b)


def test_policy_can_install_valid_heads_only_on_the_source_layout():
    b = ComponentBank(prior())
    previous = b.posterior
    proposed = previous._replace(alpha=previous.alpha.at[0].add(1),
                                 categories=(previous.categories[0].at[0, 0].add(1),
                                             previous.categories[1]))
    result = b.replace_posterior(proposed, previous=previous)
    assert result.status == SlotStatus.OK and result.bank.reference(0) == b.reference(0)
    assert result.bank.posterior.alpha[0] == b.reset_prior.alpha[0] + 1
    stale = result.bank.replace_posterior(proposed, previous=previous)
    assert stale.status == SlotStatus.STALE_REFERENCE and stale.bank is result.bank
    for invalid in (proposed._replace(active=~proposed.active),
                    proposed._replace(alpha=proposed.alpha.at[2].add(1)),
                    proposed._replace(alpha=proposed.alpha.at[0].set(-1))):
        result = b.replace_posterior(invalid, previous=previous)
        assert result.status == SlotStatus.INVALID_POSTERIOR and result.bank is b

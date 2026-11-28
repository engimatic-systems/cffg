"""Atomic soft updates, missing fields, and resuming a mixed component bank."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import mixed_learning as mixed


def fixture(dtype):
    def f(values):
        return jnp.asarray(values, dtype=dtype)

    bank = mixed.MixedBank(
        f([[1, -1]] * 3), f([2] * 3), f([[[2, 1], [1, 2]]] * 3), f([4] * 3),
        f([2, 5, .1]), (f([[2, 3]] * 3), f([[1, 2, 1]] * 3)),
        jnp.array([True, True, False]),
    )
    batch = mixed.MixedBatch(
        f([[2, 0], [0, 0]]), f([[[0, 0], [0, 0]], [[.5, -.25], [-.25, .5]]]),
        jnp.array([True, True]), (jnp.array([0, 1]), jnp.array([-1, 1])),
    )
    return bank, batch, f([[.25, .75, 0], [.5, .5, 0]])


def same_bank(left, right):
    assert left._fields == right._fields and len(left.categories) == len(right.categories)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_independent_fractional_posterior_and_resume(tmp_path, dtype):
    bank, batch, weights = fixture(dtype)
    update = jax.jit(mixed.update)
    full, valid = update(bank, batch, weights)
    assert valid
    tol = 8 * np.finfo(dtype).eps
    np.testing.assert_allclose(full.mu[:2], [[10/11, -8/11], [14/13, -8/13]], rtol=tol)
    expected = np.array([np.array([[262, 61], [61, 246]]) / 88,
                         np.array([[362, 107], [107, 314]]) / 104])
    np.testing.assert_allclose(full.psi[:2], expected, rtol=tol)
    np.testing.assert_array_equal(full.alpha, bank.alpha + jnp.asarray([.75, 1.25, 0], dtype))
    np.testing.assert_array_equal(full.kappa, [2.75, 3.25, 2])
    np.testing.assert_array_equal(full.nu, [4.75, 5.25, 4])
    np.testing.assert_array_equal(full.categories[0], [[2.25, 3.5], [2.75, 3.5], [2, 3]])
    np.testing.assert_array_equal(full.categories[1], [[1, 2.5, 1], [1, 2.5, 1], [1, 2, 1]])

    first, valid = update(bank, jax.tree.map(lambda x: x[:1], batch), weights[:1])
    assert valid
    path = tmp_path / "posterior.npz"
    np.savez(path, **first.state_dict())
    with np.load(path, allow_pickle=False) as state:
        resumed = mixed.MixedBank.from_state_dict(state)
    same_bank(resumed, first)
    tail = jax.tree.map(lambda x: x[1:], batch)
    split, valid = update(resumed, tail, weights[1:])
    assert valid
    same_bank(split, update(first, tail, weights[1:])[0])
    for a, b in zip(jax.tree.leaves(split), jax.tree.leaves(full), strict=True):
        np.testing.assert_allclose(a, b, rtol=tol, atol=tol)


def test_each_field_updates_only_when_observed():
    bank, batch, weights = fixture(jnp.float32)
    batch = batch._replace(
        mean=jnp.full_like(batch.mean, jnp.nan), covariance=jnp.full_like(batch.covariance, jnp.nan),
        gaussian_observed=jnp.array([False, False]),
        categories=(jnp.array([1, -1]), jnp.array([-1, -1])),
    )
    result, valid = mixed.update(bank, batch, weights)
    assert valid
    np.testing.assert_array_equal(result.alpha, bank.alpha + weights[0])
    np.testing.assert_array_equal(result.categories[0][:, 1], bank.categories[0][:, 1] + weights[0])
    np.testing.assert_array_equal(result.categories[1], bank.categories[1])
    for field in ("mu", "psi", "nu", "kappa"):
        np.testing.assert_array_equal(getattr(result, field), getattr(bank, field))
    empty = jax.tree.map(lambda x: x[:0], batch)
    result, valid = mixed.update(bank, empty, weights[:0])
    assert valid
    same_bank(result, bank)


def test_update_rejects_invalid_batch_without_partial_commit():
    bank, batch, weights = fixture(jnp.float32)
    update = jax.jit(mixed.update)
    invalid = [
        (batch, weights.at[1, 1].set(jnp.nan)),
        (batch, weights.at[0].set(jnp.array([0., .5, .5], weights.dtype))),
        (batch, weights.at[0].set(jnp.array([0., .5, 0.], weights.dtype))),
        (batch._replace(categories=(jnp.array([0, 3]), batch.categories[1])), weights),
        (batch._replace(covariance=batch.covariance.at[1, 0, 0].set(-1.)), weights),
    ]
    for observations, w in invalid:
        result, valid = update(bank, observations, w)
        assert not valid
        same_bank(result, bank)


def test_restore_rejects_invalid_snapshot():
    bank, _, _ = fixture(jnp.float32)
    for extra in ({"version": np.int32(3)}, {"category_9": np.ones(1)},
                  {"psi": np.zeros_like(bank.psi)}):
        with pytest.raises(ValueError):
            mixed.MixedBank.from_state_dict({**bank.state_dict(), **extra})


@pytest.mark.parametrize(("width", "indices", "valid"), [
    (256, np.array([-1, 127], dtype=np.int8), True),
    (257, np.array([0, 255], dtype=np.uint8), True),
    (256, np.array([-2, 0], dtype=np.int8), False),
    (255, np.array([0, 255], dtype=np.uint8), False),
])
def test_narrow_category_indices_score_and_update_only_observed_bins(width, indices, valid):
    bank, batch, weights = fixture(jnp.float32)
    head = jnp.broadcast_to(jnp.arange(1, width + 1, dtype=bank.mu.dtype), (len(bank.mu), width))
    bank = bank._replace(categories=(head,))
    batch = batch._replace(gaussian_observed=jnp.array([False, False]),
                           categories=(jnp.asarray(indices),))
    score = jax.jit(mixed.score)
    scores, _, accepted = score(bank, batch)
    assert bool(accepted) == valid
    reference, _, _ = score(bank, batch._replace(categories=(jnp.asarray(indices, jnp.int32),)))
    np.testing.assert_array_equal(scores, reference)
    updated, accepted = jax.jit(mixed.update)(bank, batch, weights)
    assert bool(accepted) == valid
    counts = np.zeros_like(head)
    if valid:
        for row, index in enumerate(indices):
            if int(index) >= 0:
                counts[:, index] += weights[row]
    same_bank(updated, bank._replace(categories=(head + counts,),
                                     alpha=bank.alpha + counts.sum(axis=1)))


def test_snapshot_checks_ordered_head_inventory_and_reads_legacy():
    bank, _, _ = fixture(jnp.float32)
    state = bank.state_dict()
    assert state["version"] == 2
    np.testing.assert_array_equal(state["category_cardinalities"], [2, 3])
    legacy = {k: v for k, v in state.items() if k != "category_cardinalities"}
    legacy["version"] = np.int32(1)
    same_bank(mixed.MixedBank.from_state_dict(legacy), bank)
    for missing in ("category_0", "category_1", "category_cardinalities"):
        with pytest.raises(ValueError, match="missing"):
            mixed.MixedBank.from_state_dict({k: v for k, v in state.items() if k != missing})
    for widths in ([3, 2], [2], [2, 0], [[2, 3]], [2., 3.]):
        with pytest.raises(ValueError):
            mixed.MixedBank.from_state_dict({**state, "category_cardinalities": np.array(widths)})
    with pytest.raises(ValueError, match="shape"):
        mixed.MixedBank.from_state_dict({**state, "category_1": state["category_1"][:, :-1]})
    empty = bank._replace(categories=())
    same_bank(mixed.MixedBank.from_state_dict(empty.state_dict()), empty)

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
    for extra in ({"version": np.int32(2)}, {"category_9": np.ones(1)},
                  {"psi": np.zeros_like(bank.psi)}):
        with pytest.raises(ValueError):
            mixed.MixedBank.from_state_dict({**bank.state_dict(), **extra})

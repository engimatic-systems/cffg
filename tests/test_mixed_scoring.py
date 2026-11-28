"""Independent absolute expected-log and assignment oracles for mixed fields."""

from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import digamma, softmax

from cffg.mixed_scoring import score, validate_bank, validate_inputs


Bank = namedtuple("Bank", "mu kappa psi nu alpha categories active")
Batch = namedtuple("Batch", "mean covariance gaussian_observed categories")


def fixture(dtype):
    def floats(value):
        return jnp.asarray(value, dtype=dtype)

    bank = Bank(floats([[1, -1]] * 3), floats([2] * 3), floats([[[2, 1], [1, 2]]] * 3),
                floats([4] * 3), floats([2, 5, 100]), (floats([[2, 3]] * 3),
                floats([[1, 2, 1]] * 3)), jnp.array([True, True, False]))
    batch = Batch(floats([[2, 0], [0, 0], [np.nan, np.inf]]),
                  floats([[[0, 0], [0, 0]], [[.5, -.25], [-.25, .5]], [[np.nan] * 2] * 2]),
                  jnp.array([True, True, False]), (jnp.array([0, 1, -1]), jnp.array([-1, 1, -1])))
    return bank, batch


def oracle(bank, batch, mixing):
    """Scalar NumPy/SciPy evaluation against the NIW and Dirichlet definitions."""
    bank, batch = jax.tree.map(np.asarray, (bank, batch))
    n, d = batch.mean.shape
    absolute = np.full((n, len(bank.mu)), -np.inf)
    for k in np.flatnonzero(bank.active):
        scale = bank.psi[k].astype(float)
        precision = float(bank.nu[k]) * np.linalg.inv(scale)
        elogdet = (digamma((float(bank.nu[k]) - np.arange(d)) / 2).sum()
                   + d * np.log(2) - np.linalg.slogdet(scale)[1])
        for i in range(n):
            value = 0.
            if batch.gaussian_observed[i]:
                delta = batch.mean[i].astype(float) - bank.mu[k]
                value = .5 * (elogdet - d * np.log(2 * np.pi) - d / float(bank.kappa[k])
                              - delta @ precision @ delta - np.trace(precision @ batch.covariance[i]))
            for head, labels in zip(bank.categories, batch.categories, strict=True):
                if labels[i] >= 0:
                    counts = head[k].astype(float)
                    value += digamma(counts[labels[i]]) - digamma(counts.sum())
            absolute[i, k] = value
    logits = absolute.copy()
    if mixing:
        # The total digamma is common to all occupied logits and cancels.
        logits[:, bank.active] += digamma(bank.alpha[bank.active].astype(float))
    return absolute, softmax(logits, axis=-1)


@pytest.mark.parametrize("dtype", [jnp.float32, pytest.param(jnp.float64, marks=pytest.mark.slow)])
def test_scores_match_absolute_oracle_and_masked_assignments(dtype):
    bank, batch = fixture(dtype)
    atol = 3e-6 if dtype == jnp.float32 else 2e-12
    compiled = jax.jit(score)
    for mixing in (False, True):
        expected, assignments = oracle(bank, batch, mixing)
        np.testing.assert_allclose(expected[:, 0], [-4.381065542, -9.047732209, 0], atol=1e-9)
        for run in (score, compiled):
            values, weights, valid = run(bank, batch, include_mixing=mixing)
            assert bool(valid) and valid.shape == () and valid.dtype == jnp.bool_
            assert values.dtype == weights.dtype == dtype
            np.testing.assert_allclose(values, expected, rtol=0, atol=atol)
            np.testing.assert_allclose(weights, assignments, rtol=0, atol=atol)
            np.testing.assert_array_equal(weights[:, 2], 0)
            np.testing.assert_array_equal(values[:, 2], -np.inf)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_large_counts_keep_small_absolute_expected_log(dtype):
    bank, batch = fixture(dtype)
    huge = 2**22
    head = jnp.array([[huge, 1], [2, 5], [1, 1]], dtype)
    bank = bank._replace(categories=(head,))
    batch = batch._replace(gaussian_observed=jnp.zeros(3, bool), categories=(jnp.array([0, 1, -1]),))
    values, _, valid = jax.jit(score)(bank, batch)
    assert valid
    # psi(a) - psi(a+1) = -1/a exactly, including at this representable power of two.
    np.testing.assert_allclose(values[0, 0], -1 / huge, atol=0, rtol=3 * np.finfo(dtype).eps)
    np.testing.assert_allclose(values[1, 0], digamma(1.) - digamma(huge + 1.), rtol=2e-7)
    assert values[2, 0] == 0


def test_two_categorical_heads_match_harmonic_score_and_small_weights_float64():
    # Record 4 of the stress fixture has missing Gaussian evidence. Integer
    # counts give exact digamma differences via the harmonic recurrence.
    group = np.arange(32) % 8
    heads = tuple((1 + (group[:, None] + np.arange(c)) % 5).astype(np.float64) for c in (3, 5))
    bank = Bank(np.zeros((32, 1)), np.ones(32), np.ones((32, 1, 1)), np.ones(32),
                np.ones(32), heads, group != 7)
    batch = Batch(np.array([[np.nan]]), np.array([[[np.nan]]]), np.array([False]),
                  (np.array([1]), np.array([4])))
    scores, weights, valid = jax.jit(score)(bank, batch)
    assert valid
    expected = np.zeros(32)
    for counts, label in zip(heads, (1, 4), strict=True):
        for k, row in enumerate(counts.astype(int)):
            expected[k] -= sum(1 / i for i in range(row[label], row.sum()))
    expected[~bank.active] = -np.inf
    tolerance = 64 * np.finfo(np.float64).eps
    np.testing.assert_allclose(scores[0, 1], -40265/9009, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(scores[0], expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(weights[0], softmax(expected), atol=0, rtol=tolerance)


def test_categorical_odds_survive_identical_distant_gaussians_float32():
    bank, batch = fixture(jnp.float32)
    head = jnp.array([[20, 1], [10, 1], [1, 1]], dtype=jnp.float32)
    bank = bank._replace(categories=(head,))
    batch = batch._replace(mean=jnp.full_like(batch.mean, 4096),
                           categories=(jnp.array([0, 0, 0]),))
    values, weights, valid = jax.jit(score)(bank, batch)
    assert valid
    np.testing.assert_array_equal(values[:2, 0], values[:2, 1])
    expected = softmax([-1/20, -1/10])
    np.testing.assert_allclose(weights[:, :2], np.broadcast_to(expected, (3, 2)), atol=1e-7)
    np.testing.assert_array_equal(weights[:, 2], 0)


@pytest.mark.parametrize("dtype", [jnp.float32, pytest.param(jnp.float64, marks=pytest.mark.slow)])
def test_batched_banks_and_rank_deficient_observations(dtype):
    bank, batch = fixture(dtype)
    other = bank._replace(mu=bank.mu.at[1].set(jnp.asarray([-.5, 3], dtype=dtype)),
                          active=jnp.array([False, True, True]))
    rank_one = batch._replace(covariance=batch.covariance.at[1].set(
        jnp.asarray([[.25, -.5], [-.5, 1]], dtype=dtype)))
    banks = jax.tree.map(lambda x, y: jnp.stack((x, y)), bank, other)
    batches = jax.tree.map(lambda x, y: jnp.stack((x, y)), batch, rank_one)
    values, weights, valid = jax.jit(jax.vmap(score))(banks, batches)
    np.testing.assert_array_equal(valid, [True, True])
    for i, (one, observations) in enumerate(((bank, batch), (other, rank_one))):
        expected, assignments = oracle(one, observations, False)
        atol = 3e-6 if dtype == jnp.float32 else 2e-12
        np.testing.assert_allclose(values[i], expected, atol=atol, rtol=3*np.finfo(dtype).eps)
        np.testing.assert_allclose(weights[i], assignments, atol=atol, rtol=0)


def test_numerically_invalid_batches_fail_atomically_under_vmap():
    bank, batch = fixture(jnp.float32)
    bad_banks = [
        bank._replace(active=jnp.zeros(3, bool)),
        bank._replace(mu=bank.mu.at[2, 0].set(jnp.nan)),
        bank._replace(kappa=bank.kappa.at[2].set(0)),
        bank._replace(nu=bank.nu.at[2].set(1)),
        bank._replace(alpha=bank.alpha.at[2].set(-1)),
        bank._replace(psi=bank.psi.at[2].set(jnp.array([[9., 3.], [3., 1.]], jnp.float32))),
        bank._replace(categories=(bank.categories[0].at[2, 0].set(jnp.inf), bank.categories[1])),
    ]
    bad_batches = [
        batch._replace(mean=batch.mean.at[0, 0].set(jnp.nan)),
        batch._replace(covariance=batch.covariance.at[1, 0, 0].set(jnp.inf)),
        batch._replace(covariance=batch.covariance.at[0, 0, 1].set(1)),
        batch._replace(covariance=batch.covariance.at[1, 0, 1].set(.4)),
        batch._replace(covariance=batch.covariance.at[1].set(jnp.array([[-1., 0.], [0., 1e24]], jnp.float32))),
        batch._replace(covariance=batch.covariance.at[1].set(jnp.array([[1., 2.], [2., 1.]], jnp.float32))),
        batch._replace(categories=(batch.categories[0].at[1].set(-2), batch.categories[1])),
        batch._replace(categories=(batch.categories[0], batch.categories[1].at[0].set(3))),
    ]
    pairs = [(b, batch) for b in bad_banks] + [(bank, b) for b in bad_batches]
    banks, batches = jax.tree.map(lambda *xs: jnp.stack(xs), *pairs)
    values, weights, valid = jax.jit(jax.vmap(score))(banks, batches)
    np.testing.assert_array_equal(valid, False)
    np.testing.assert_array_equal(values, -np.inf)
    np.testing.assert_array_equal(weights, 0)


def test_rank_one_observation_trace_with_ill_conditioned_scale():
    bank, batch = fixture(jnp.float32)
    scale = jnp.array([[1, 1-1e-6], [1-1e-6, 1]], jnp.float32)
    bank = bank._replace(psi=jnp.broadcast_to(scale, bank.psi.shape), mu=jnp.zeros_like(bank.mu))
    batch = batch._replace(mean=jnp.zeros_like(batch.mean), covariance=jnp.ones_like(batch.covariance))
    values, _, valid = jax.jit(score)(bank, batch)
    expected, _ = oracle(bank, batch, False)
    assert valid
    np.testing.assert_allclose(values, expected, atol=4e-6, rtol=0)


def test_large_sparse_bank_matches_active_component_oracle():
    k, n, d = 256, 64, 6
    rng = np.random.default_rng(79)
    dtype = np.float32
    scale = np.broadcast_to(np.eye(d, dtype=dtype) + dtype(.125), (k, d, d))
    bank = Bank(rng.normal(size=(k, d)).astype(dtype), np.full(k, 3, dtype), scale,
                np.full(k, 12, dtype), rng.uniform(1, 8, k).astype(dtype),
                (rng.uniform(1, 5, (k, 5)).astype(dtype),), np.arange(k) % 61 == 0)
    batch = Batch(rng.normal(size=(n, d)).astype(dtype), scale[:n] * dtype(.25),
                  np.arange(n) % 5 != 0, (rng.integers(-1, 5, n, dtype=np.int32),))
    values, weights, valid = jax.jit(score)(bank, batch, include_mixing=True)
    assert valid
    expected, assignments = oracle(bank, batch, True)
    np.testing.assert_allclose(values, expected, atol=4e-6, rtol=6*np.finfo(dtype).eps)
    np.testing.assert_allclose(weights, assignments, atol=3e-6)
    np.testing.assert_array_equal(weights[:, ~bank.active], 0)


def test_empty_records_validate_all_priors():
    bank, batch = fixture(jnp.float64)
    empty = jax.tree.map(lambda x: x[:0], batch)
    run = jax.jit(score)
    for b, expected in ((bank, True), (bank._replace(alpha=bank.alpha.at[2].set(0)), False)):
        scores, weights, valid = run(b, empty)
        assert scores.shape == weights.shape == (0, 3)
        assert bool(valid) == bool(validate_inputs(b, empty)) == bool(validate_bank(b)) == expected


def test_static_layout_errors_fail_at_trace_time():
    bank, batch = fixture(jnp.float32)
    cases = [
        (bank._replace(psi=jnp.ones((3, 2), jnp.float32)), batch, ValueError),
        (bank._replace(kappa=bank.kappa.astype(jnp.float64)), batch, TypeError),
        (bank._replace(mu=bank.mu.astype(jnp.int32)), batch, TypeError),
        (bank._replace(active=bank.active.astype(jnp.int32)), batch, TypeError),
        (bank, batch._replace(mean=batch.mean.astype(jnp.float64)), TypeError),
        (bank, batch._replace(covariance=batch.covariance[:1]), ValueError),
        (bank, batch._replace(gaussian_observed=batch.gaussian_observed.astype(int)), TypeError),
        (bank, batch._replace(categories=()), ValueError),
        (bank, batch._replace(categories=(batch.mean[:, 0], batch.categories[1])), TypeError),
        (bank, batch._replace(categories=(batch.categories[0][:1], batch.categories[1])), ValueError),
    ]
    for b, x, error in cases:
        with pytest.raises(error):
            jax.eval_shape(score, b, x)
    with pytest.raises(TypeError):
        jax.eval_shape(score, bank, batch, include_mixing=1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scale_response_accounts_for_distant_observations(dtype):
    from cffg.mixed_scoring import _scale_response

    with jax.enable_x64(dtype == np.float64):
        # Same scale perturbation, two different observation energies. The
        # distant observation magnifies the inverse's second-order term.
        r = dtype(np.sqrt(np.finfo(dtype).eps) / 16)
        delta = np.array([[[0, r], [r, 0]]], dtype)
        active = np.array([True])
        evaluate = jax.jit(_scale_response)
        local, volume = evaluate(delta, np.array([2], dtype), active)
        np.testing.assert_array_equal(local, delta)
        np.testing.assert_array_equal(volume, 0)
        remote, volume = evaluate(delta, np.array([1/r**2], dtype), active)
        exact = np.array([[[-r*r, r], [r, -r*r]]]) / (1 - r*r)
        np.testing.assert_allclose(remote, exact, rtol=8*np.finfo(dtype).eps, atol=0)
        # log1p of opposite eigenvalues can cancel to source precision.
        np.testing.assert_allclose(volume, np.log1p(-r*r), atol=4*np.finfo(dtype).eps*r)

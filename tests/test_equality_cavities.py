"""Independent exclusion oracles for additive Gaussian and EF equality rules."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import efrules, gauss


def excluding_each(values):
    return np.stack([
        np.delete(values, leg, axis=1).sum(axis=1)
        for leg in range(values.shape[1])
    ], axis=1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("arity", [3, 300])
def test_dominant_recipient_does_not_erase_weak_information(dtype, arity):
    # Binary powers make the oracle exact in both dtypes, even at high arity.
    dominant = dtype(2 ** (30 if dtype is np.float32 else 60))
    eta = np.ones((2, arity, 2), dtype=dtype)
    eta[:, 0] = dominant
    eta[1, :, 1] *= -1
    precision = np.broadcast_to(np.eye(2, dtype=dtype), (2, arity, 2, 2)).copy()
    precision[:, 0] *= dominant
    expected_eta, expected_precision = excluding_each(eta), excluding_each(precision)
    for rule in (gauss.equality_messages, jax.jit(gauss.equality_messages)):
        actual_eta, actual_precision = rule(eta, precision)
        np.testing.assert_array_equal(actual_eta[:, 0], expected_eta[:, 0])
        np.testing.assert_array_equal(actual_precision[:, 0], expected_precision[:, 0])
        tolerance = 4 * arity * np.finfo(dtype).eps
        np.testing.assert_allclose(actual_eta, expected_eta, rtol=tolerance, atol=0)
        np.testing.assert_allclose(actual_precision, expected_precision, rtol=tolerance, atol=0)
    for rule in (efrules.equality_messages, jax.jit(efrules.equality_messages)):
        actual = rule(eta)
        np.testing.assert_array_equal(actual[:, 0], expected_eta[:, 0])
        np.testing.assert_allclose(actual, expected_eta, rtol=tolerance, atol=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_weak_precision_from_planning_mechanism_is_preserved(dtype):
    weak = dtype(6.083413961142839e-16)
    precision = np.zeros((1, 3, 2, 2), dtype=dtype)
    precision[0, 0] = np.diag([400., 1.])
    precision[0, 1] = weak * np.eye(2)
    eta = np.zeros((1, 3, 2), dtype=dtype)
    actual_eta, actual_precision = gauss.equality_messages(eta, precision)
    np.testing.assert_array_equal(actual_precision[0, 0], precision[0, 1])
    np.testing.assert_array_equal(actual_eta, eta)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_vmap_matches_independent_exclusion_and_analytic_jacobian(dtype):
    values = np.arange(24, dtype=dtype).reshape(3, 1, 4, 2) - 12
    actual = jax.jit(jax.vmap(efrules.equality_messages))(values)
    for instance, result in zip(values, actual, strict=True):
        np.testing.assert_array_equal(result, excluding_each(instance))

    def rule(legs):
        return efrules.equality_messages(legs[None, :, None])[0, :, 0]

    expected = np.ones((4, 4), dtype=dtype) - np.eye(4, dtype=dtype)
    for derivative in (jax.jacfwd, jax.jacrev):
        np.testing.assert_array_equal(jax.jit(derivative(rule))(jnp.ones(4, dtype)), expected)
    eta = values
    precision = np.broadcast_to(np.eye(2, dtype=dtype), (*values.shape[:-1], 2, 2))
    batched_eta, batched_precision = jax.jit(jax.vmap(gauss.equality_messages))(eta, precision)
    for index in range(len(values)):
        np.testing.assert_array_equal(batched_eta[index], excluding_each(eta[index]))
        np.testing.assert_array_equal(batched_precision[index], excluding_each(precision[index]))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_other_leg_is_not_sanitized(bad):
    values = jnp.asarray([[[bad], [2.], [3.]]])
    actual = efrules.equality_messages(values)
    assert float(actual[0, 0, 0]) == 5.
    assert not np.isfinite(actual[0, 1:]).any()


def test_vacuous_and_single_leg_outputs_are_exact_zero():
    for arity in (1, 3, 300):
        eta = jnp.zeros((2, arity, 2))
        precision = jnp.zeros((2, arity, 2, 2))
        gaussian = gauss.equality_messages(eta, precision)
        np.testing.assert_array_equal(gaussian[0], eta)
        np.testing.assert_array_equal(gaussian[1], precision)
        np.testing.assert_array_equal(efrules.equality_messages(eta), eta)

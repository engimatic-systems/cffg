"""Independent affine and dense-grid oracles for local nonlinear projections."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import gauss
from cffg.nonlinear import NonlinearFailure, approximation, local_result


@pytest.mark.parametrize("method", ["linearization", "sigma_point"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("observed", [False, True])
def test_affine_messages_and_node_energy_match_analytic_gaussian(method, dtype, observed):
    a = jnp.asarray([[1.2, -0.3], [0.4, 0.8]], dtype)
    b = jnp.asarray([0.2, -0.1], dtype)
    noise = jnp.asarray([[0.5, 0.1], [0.1, 0.8]], dtype)
    li = jnp.asarray([[1.5, 0.2], [0.2, 0.9]], dtype)
    lo = jnp.asarray([[0.8, -0.1], [-0.1, 1.1]], dtype)
    ei, eo = jnp.asarray([0.3, -0.4], dtype), jnp.asarray([0.6, 0.1], dtype)
    y = jnp.asarray([0.4, -0.2], dtype)
    evaluate = lambda x, parameters: a @ x + b
    actual = local_result(evaluate, ei, li, eo, lo, jnp.zeros(0, dtype), noise, method,
                          y if observed else None, jnp.asarray(1, dtype) if observed else None)
    batch = lambda value: value[None]
    w = jnp.linalg.inv(noise)
    expected_out = gauss.linear_forward(*map(batch, (ei, li, a, b, w)))
    if observed:
        expected_in = gauss.linear_observed_backward(batch(y), jnp.ones(1, dtype), batch(a), batch(b), batch(w))
        energy = gauss.linear_observed_energy(batch(ei), batch(li), batch(y), jnp.ones(1, dtype), batch(a), batch(b), batch(w))
    else:
        expected_in = gauss.linear_backward(*map(batch, (eo, lo, a, b, w)))
        energy = gauss.linear_energy(*map(batch, (eo, lo, ei, li, a, b, w)))
    tolerance = 256 * np.finfo(np.dtype(dtype)).eps
    for role, expected in (("in", expected_in), ("out", expected_out)):
        np.testing.assert_allclose(actual[f"eta_{role}"], expected[0][0], rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(actual[f"precision_{role}"], expected[1][0], rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(actual["energy"], energy[0], rtol=tolerance, atol=tolerance)
    assert bool(actual["proper"])


def _scalar(method, *, li=2.0, lo=1 / 0.09, noise=0.16, theta=0.3, ei=0.5):
    scalar = lambda x: jnp.asarray([x], dtype=jnp.float64)
    matrix = lambda x: jnp.asarray([[x]], dtype=jnp.float64)
    return local_result(lambda x, p: x + p[0] * x ** 3,
                        scalar(ei), matrix(li), scalar(0.8 / 0.09), matrix(lo),
                        scalar(theta), matrix(noise), method)


def _grid_errors(size, gaussians):
    x = np.linspace(-8, 8, size)
    logp = -0.5 * ((x - 0.25) ** 2 / 0.5 + (0.8 - x - 0.3 * x ** 3) ** 2 / 0.25)
    density = np.exp(logp - logp.max())
    density /= np.trapezoid(density, x)
    errors = []
    for mean, variance in gaussians:
        q = np.exp(-0.5 * (x - mean) ** 2 / variance) / np.sqrt(2 * np.pi * variance)
        errors.append(np.sqrt(0.5 * np.trapezoid((np.sqrt(density) - np.sqrt(q)) ** 2, x)))
    return np.asarray(errors)


def test_curved_posterior_meets_frozen_grid_thresholds():
    values = [_scalar(method) for method in ("linearization", "sigma_point")]
    gaussians = [(float(v["mean_in"][0]), float(v["covariance_in"][0, 0])) for v in values]
    fine = _grid_errors(131073, gaussians)
    coarse = _grid_errors(65537, gaussians)
    discrepancy = np.max(np.abs(fine - coarse))
    assert fine[0] <= 0.16
    assert fine[1] <= 0.10
    assert fine[1] <= fine[0] + discrepancy


@pytest.mark.parametrize("precision", [1e-4, 1e4, 1e8, 1e38])
def test_float32_strong_prior_does_not_cancel_backward_site(precision):
    vector = lambda x: jnp.asarray([x], jnp.float32)
    matrix = lambda x: jnp.asarray([[x]], jnp.float32)
    value = local_result(lambda x, p: x, vector(.25 * precision), matrix(precision),
                         vector(1), matrix(1), vector(0), matrix(1), "linearization")
    if precision == 1e38:
        assert not value["proper"]
        assert value["failure_code"] == NonlinearFailure.IMPROPER_TILT
        assert np.isinf(value["energy"])
    else:
        assert value["proper"]
        np.testing.assert_allclose(value["precision_in"], [[.5]], rtol=32 * np.finfo(np.float32).eps)
        np.testing.assert_allclose(value["eta_in"], [.5], rtol=32 * np.finfo(np.float32).eps)


def test_float32_sigma_points_cannot_silently_lose_the_input_variance():
    vector = lambda x: jnp.asarray([x], jnp.float32)
    matrix = lambda x: jnp.asarray([[x]], jnp.float32)
    value = local_result(lambda x, p: x, vector(1e8), matrix(1), vector(0), matrix(0),
                         vector(0), matrix(1), "sigma_point")
    assert not value["proper"]
    assert value["failure_code"] == NonlinearFailure.UNREPRESENTABLE_SIGMA_POINTS
    assert np.isinf(value["energy"])


def test_sigma_points_retain_nonzero_curvature_covariance():
    matrix, offset, noise = approximation(
        lambda x, p: x + p[0] * x ** 3,
        jnp.asarray([0.25]), jnp.asarray([[0.5]]), jnp.asarray([[0.16]]),
        jnp.asarray([0.3]), "sigma_point",
    )
    np.testing.assert_allclose(matrix, [[1.20625]])
    np.testing.assert_allclose(offset, [0.065625])
    np.testing.assert_allclose(noise, [[0.1853125]])


def test_vacuous_input_is_neutral_and_has_no_finite_belief():
    value = _scalar("sigma_point", ei=0, li=0)
    assert value["failure_code"] == NonlinearFailure.VACUOUS_INPUT
    for role in ("in", "out"):
        assert np.count_nonzero(value[f"eta_{role}"]) == 0
        assert np.count_nonzero(value[f"precision_{role}"]) == 0
    assert np.isinf(value["energy"]) and not value["proper"]


@pytest.mark.parametrize("kwargs,reason", [
    ({"li": -1}, NonlinearFailure.IMPROPER_CAVITY),
    ({"noise": 0}, NonlinearFailure.INVALID_NOISE),
    ({"theta": np.nan}, NonlinearFailure.NONFINITE_FUNCTION),
    ({"lo": -100}, NonlinearFailure.IMPROPER_TILT),
])
def test_invalid_local_inputs_have_explicit_status(kwargs, reason):
    value = _scalar("linearization", **kwargs)
    assert value["status"] == 2 and value["failure_code"] == reason
    assert np.isinf(value["energy"]) and not value["proper"]


def test_finite_negative_site_precision_is_not_clamped():
    value = _scalar("linearization", lo=-0.1, theta=0, noise=1)
    assert value["proper"] and value["precision_in"][0, 0] < 0
    assert value["precision_in"][0, 0] + 2 > 0


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_dimension32_jit_and_member_batching(dtype):
    dimension = 32
    identity = jnp.eye(dimension, dtype=dtype)
    e = jnp.full((dimension,), 0.1, dtype)
    parameters = jnp.asarray([[0.1], [0.2]], dtype)
    function = lambda theta: local_result(
        lambda x, p: x + p[0] * jnp.tanh(x), e, identity, -e, identity,
        theta, identity, "sigma_point",
    )
    batched = jax.jit(jax.vmap(function))(parameters)
    for index in range(2):
        direct = function(parameters[index])
        assert direct["proper"]
        for key in direct:
            np.testing.assert_allclose(batched[key][index], direct[key],
                                       atol=512 * np.finfo(np.dtype(dtype)).eps,
                                       rtol=512 * np.finfo(np.dtype(dtype)).eps)


def test_adaptive_energy_derivative_includes_changing_coefficients():
    function = lambda eta: _scalar("linearization", ei=eta)["energy"]
    gradient = jax.grad(function)(jnp.asarray(0.5))
    step = 1e-5
    finite_difference = (function(0.5 + step) - function(0.5 - step)) / (2 * step)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-6, atol=1e-8)

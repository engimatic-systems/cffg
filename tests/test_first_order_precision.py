"""Separate exact-pass consistency from retained-controller accuracy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import Engine, RunStatus, compile_template, gauss, set_params
from cffg.colony import Colony
from cffg.runtime_types import _replace
from first_order_fixture import CANDIDATES, dense_reference, first_order_model


def assert_moments(engine, result, mean, covariance, data, dtype):
    # Accuracy targets are separate from the solver's message certificate:
    # 1% of posterior standard deviation/covariance and 1 millinat per datum
    # in f32; tighter comparisons require f64. These are fixture bounds only.
    moment_target = .01 if dtype is np.float32 else 1e-8
    actual = engine.gaussian_marginals(result.state)
    for index in range(len(data)):
        block = slice(3 * index, 3 * index + 3)
        reference_covariance = covariance[block, block]
        root = np.linalg.cholesky(reference_covariance)
        actual_mean, actual_covariance = actual[f"s{index}"]
        assert np.isfinite(actual_mean).all() and np.isfinite(actual_covariance).all()
        assert np.linalg.eigvalsh(actual_covariance).min() > 0
        mean_error = np.linalg.solve(root, actual_mean - mean[block])
        left = np.linalg.solve(root, actual_covariance - reference_covariance)
        covariance_error = np.linalg.solve(root, left.T).T
        assert np.linalg.norm(mean_error) <= moment_target
        assert np.linalg.norm(covariance_error, ord=2) <= moment_target


def assert_energy(engine, result, energy, data, dtype):
    energy_per_datum = .001 if dtype is np.float32 else 1e-9
    observed_dimensions = sum(np.size(value) for value in data.values() if value is not None)
    energy_target = energy_per_datum * max(1, observed_dimensions)
    assert abs(float(engine.bfe(result.state)) - energy) <= energy_target
    assert abs(result.free_energies[-1] - energy) <= energy_target


def assert_reference(engine, result, arrays, data, dtype):
    mean, covariance, energy = dense_reference(arrays, data, runtime_dtype=dtype)
    assert_moments(engine, result, mean, covariance, data, dtype)
    assert_energy(engine, result, energy, data, dtype)
    return mean, covariance, energy


def run_model(dtype, *, candidate="v3a2", horizon=15, schedule="auto", source_dtype=np.float64):
    template, arrays, data = first_order_model(candidate, horizon=horizon, source_dtype=source_dtype)
    engine = Engine(compile_template(template, arrays=arrays), schedule=schedule)
    initial = engine.bind_data(engine.init_state(), data)
    result = engine.run(initial, 4 if schedule == "auto" else 8 * len(data), tol=1e-6)
    result.require_converged()
    assert_reference(engine, result, arrays, data, dtype)
    return engine, result, template, arrays, data


@pytest.mark.parametrize("dtype", [np.float32, pytest.param(np.float64, marks=pytest.mark.slow)])
def test_retained_controller_certifies_without_relaxing_precision(dtype):
    with jax.enable_x64(dtype is np.float64):
        engine, result, template, arrays, data = run_model(dtype)
        assert engine.schedule.kind == "two_pass" and result.iterations == 1
        assert result.residual <= max(1e-6, 64 * np.finfo(dtype).eps)
        assert np.asarray(result.state.g_eta).dtype == np.dtype(dtype)
        assert list(template.factors)[:4] == ["prior", "eq0", "em0", "tr1"]
        assert all(value.dtype == np.float64 for value in arrays.values())
        assert all(data[f"e{index}"] is None for index in range(4))

        rounded_mean, rounded_covariance, _ = dense_reference(arrays, data, runtime_dtype=dtype)
        original_mean, original_covariance, _ = dense_reference(arrays, data)
        input_target = 128 * np.finfo(dtype).eps
        np.testing.assert_allclose(rounded_mean, original_mean, atol=input_target, rtol=input_target)
        np.testing.assert_allclose(rounded_covariance, original_covariance, atol=input_target, rtol=input_target)


def test_recipient_rounding_cannot_change_its_outgoing_cavity():
    with jax.enable_x64(False):
        # The other legs stay fixed while the reverse message becomes proper.
        eta = jnp.array([[[0.], [4096.], [.0003]]], dtype=jnp.float32)
        precision = jnp.array([[[[0.]], [[4096.]], [[.0003]]]], dtype=jnp.float32)
        changed_eta, changed_precision = eta.at[:, 0].set(10000.), precision.at[:, 0].set(10000.)
        for rule in (gauss.equality_messages, jax.jit(gauss.equality_messages)):
            before, after = rule(eta, precision), rule(changed_eta, changed_precision)
            np.testing.assert_array_equal(before[0][:, 0], after[0][:, 0])
            np.testing.assert_array_equal(before[1][:, 0], after[1][:, 0])
        before = jnp.sum(eta, axis=1) - eta[:, 0]
        after = jnp.sum(changed_eta, axis=1) - changed_eta[:, 0]
        assert not np.array_equal(before, after)


@pytest.mark.slow
@pytest.mark.parametrize("source_dtype", [np.float32, np.float64])
def test_historical_equality_alone_recreates_failed_certificate(monkeypatch, source_dtype):
    def total_minus_recipient(eta, precision):
        total_eta, total_precision = gauss.GAUSSIAN.product_axis((eta, precision), axis=1, keepdims=True)
        return total_eta - eta, total_precision - precision

    with jax.enable_x64(False):
        engine, result, template, arrays, data = run_model(np.float32, source_dtype=source_dtype)
        assert result.residual == 0.
        monkeypatch.setattr(gauss, "equality_messages", total_minus_recipient)
        historical_engine = Engine(compile_template(template, arrays=arrays))
        failed = historical_engine.run(historical_engine.bind_data(historical_engine.init_state(), data), 4, tol=1e-6)
        assert failed.status is RunStatus.BUDGET_EXHAUSTED
        assert failed.residual > 64 * np.finfo(np.float32).eps
        assert_reference(historical_engine, failed, arrays, data, np.float32)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_all_retained_parameter_candidates(dtype):
    with jax.enable_x64(dtype is np.float64):
        template, arrays, data = first_order_model()
        engine = Engine(compile_template(template, arrays=arrays))
        initial = engine.bind_data(engine.init_state(), data)
        for candidate in CANDIDATES:
            _, arrays, _ = first_order_model(candidate)
            updates = {f"tr{index}": dict(matrix=arrays["F"], offset=arrays["c"], cov=arrays["Q"])
                       for index in range(1, len(data))}
            state = _replace(initial, params=set_params(engine, initial.params, updates))
            result = engine.run(state, 4, tol=1e-6)
            result.require_converged()
            assert_reference(engine, result, arrays, data, dtype)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("horizon", [15, 150])
def test_flooding_and_tenfold_horizon(dtype, horizon):
    with jax.enable_x64(dtype is np.float64):
        run_model(dtype, horizon=horizon, schedule="auto")
        if horizon == 15:
            run_model(dtype, horizon=horizon, schedule="flood")


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_missing_likelihoods_preserve_zero_information_and_improper_readouts(dtype):
    with jax.enable_x64(dtype is np.float64):
        template, arrays, data = first_order_model()
        data = {name: None for name in data}
        engine = Engine(compile_template(template, arrays=arrays))
        initial = engine.bind_data(engine.init_state(), data)
        np.testing.assert_array_equal(initial.g_eta, 0.)
        np.testing.assert_array_equal(initial.g_lam, 0.)
        improper_mean, improper_covariance = engine.gaussian_marginals(initial)["s4"]
        assert np.isnan(improper_mean).all() and np.isnan(improper_covariance).all()
        result = engine.run(initial, 4, tol=1e-6)
        result.require_converged()
        mean, covariance, energy = dense_reference(arrays, data, runtime_dtype=dtype)
        assert_moments(engine, result, mean, covariance, data, dtype)
        assert energy == 0. and np.isfinite(result.free_energies[-1])
        if dtype is np.float64:
            assert_energy(engine, result, energy, data, dtype)


@pytest.mark.slow
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="cff-d63w: unobserved f32 tree energy loses cancellation accuracy")
def test_unobserved_float32_tree_has_zero_energy():
    with jax.enable_x64(False):
        template, arrays, data = first_order_model()
        data = {name: None for name in data}
        engine = Engine(compile_template(template, arrays=arrays))
        result = engine.run(engine.bind_data(engine.init_state(), data), 4, tol=1e-6)
        result.require_converged()
        assert_energy(engine, result, 0., data, np.float32)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_colony_uses_the_same_retained_exact_certificate(dtype):
    with jax.enable_x64(dtype is np.float64):
        engine, solo, template, arrays, data = run_model(dtype)
        colony = Colony()
        members = [colony.spawn(template, arrays, observations=data) for _ in range(2)]
        for result in colony.run(4, tol=1e-6).values():
            result.require_converged()
            np.testing.assert_array_equal(result.state.g_eta, solo.state.g_eta)
            np.testing.assert_array_equal(result.state.g_lam, solo.state.g_lam)
            assert_reference(engine, result, arrays, data, dtype)
        assert len(members) == 2

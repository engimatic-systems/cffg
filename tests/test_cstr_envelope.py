"""Explicit reactor candidates, not replay of the missing PCGym invocation."""

from math import exp
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from cffg import Engine, compile_template, load_bundle, save_bundle, set_params
from cffg import outer
from cffg.outer_chart import OuterArrays, build_outer_chart, chart_coordinates
from cffg.runtime_types import _replace
from cstr_planning_fixture import CONFIGURATION, NOMINAL_PARAMETERS, cstr_rk4, planning_fixture, planning_model
from test_cstr_planning import affine_reference, cavity


def independent_transition(state, parameters):
    """Physical ODE integrated by DOP853, independently of JAX/RK4."""
    ca = np.clip(.85 + .15 * state[0], .1, 1.6)
    temperature = np.clip(325 + 25 * state[1], 200, 450)
    coolant = np.clip(298.5 + 3.5 * state[3], 275, 322)
    rate, activation, transfer, enthalpy = (
        exp(float(parameters[0])), float(parameters[1]) * 1e4,
        exp(float(parameters[2])), exp(float(parameters[3])),
    )

    def derivative(time, value):
        ca, temperature = value
        reaction = rate * exp(-activation * (1 / temperature - 1 / 330)) * ca
        return [1 - ca - reaction,
                350 - temperature + enthalpy * reaction / 239 + transfer * (coolant - temperature) / 23900]

    solution = solve_ivp(derivative, (0, CONFIGURATION["dt"]), [ca, temperature],
                         method="DOP853", rtol=1e-12, atol=1e-12)
    assert solution.success
    ca, temperature = solution.y[:, -1]
    return np.array([(ca - .85) / .15 + state[2], (temperature - 325) / 25, state[2], state[3]])


def reconstruct_state(engine, state):
    if engine._reconstruct is None:
        return state
    values = engine._reconstruct(
        state.messages, state.g_eta, state.g_lam, state.ef_eta,
        state.g_obs, state.g_mask, state.ef_obs, state.ef_mask, state.params,
    )
    assert int(values[4].reason) == 0
    return _replace(state, messages=values[0], g_eta=values[1], g_lam=values[2], ef_eta=values[3])


def assert_certificate(engine, result, dtype, chart):
    state = result.require_converged()
    tolerance = max(1e-6, 64 * np.finfo(dtype).eps)
    assert np.isfinite(result.free_energies).all() and result.free_energies
    assert result.failure is None and result.outer_residual <= tolerance
    for mean, covariance in engine.gaussian_marginals(state).values():
        assert np.isfinite(mean).all() and np.isfinite(covariance).all()
        assert np.linalg.eigvalsh(covariance).min() > 0
    state = reconstruct_state(engine, state)
    values = engine._iter_step(
        state.messages, state.g_eta, state.g_lam, state.g_obs, state.g_mask,
        state.ef_eta, state.ef_obs, state.ef_mask, state.params, 0., False,
    )
    assert int(values[6].reason) == 0
    mapped = _replace(state, messages=values[0], g_eta=values[1], g_lam=values[2], ef_eta=values[3])
    mapped = reconstruct_state(engine, mapped)
    before = OuterArrays(state.messages, state.g_eta, state.g_lam, state.ef_eta)
    after = OuterArrays(mapped.messages, mapped.g_eta, mapped.g_lam, mapped.ef_eta)
    u, v = np.asarray(chart_coordinates(chart, before)), np.asarray(chart_coordinates(chart, after))
    assert np.max(np.abs(v - u) / (1 + np.abs(u))) <= tolerance
    energy, mapped_energy = float(engine.bfe(state)), float(engine.bfe(mapped))
    assert np.isfinite([energy, mapped_energy]).all()
    assert abs(mapped_energy - energy) <= tolerance * (1 + abs(energy))


def run_certified(engine, state, dtype):
    charts = []

    def capture_chart(*args, **kwargs):
        chart = build_outer_chart(*args, **kwargs)
        charts.append(chart)
        return chart

    # The residual metric freezes whitening at solve entry, not at the result.
    with patch.object(outer, "build_outer_chart", side_effect=capture_chart):
        result = engine.run(state, 60, tol=1e-6)
    assert_certificate(engine, result, dtype, charts[-1])
    return result


def assert_physical_cavities(engine, result, dtype):
    evaluate = jax.jit(jax.vmap(lambda x, p: cstr_rk4(x, p, CONFIGURATION), in_axes=(0, None)))
    for name in ("tr1", "tr2"):
        eta, precision = cavity(engine, result.state, name)
        covariance = np.linalg.inv(np.asarray(precision[1]))
        mean = covariance @ np.asarray(eta[1])
        offsets = 2 * np.linalg.cholesky(covariance).T
        points = np.concatenate([mean[None], mean + offsets, mean - offsets]).astype(dtype)
        parameters = jnp.asarray(NOMINAL_PARAMETERS, dtype=dtype)
        values = np.asarray(evaluate(jnp.asarray(points), parameters))
        assert np.isfinite(values).all()
        assert np.isfinite(values.T @ values).all()
        independent = np.stack([independent_transition(point, np.asarray(parameters)) for point in points])
        np.testing.assert_allclose(values, independent, rtol=0., atol=max(1e-5, 512 * np.finfo(dtype).eps))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_recipe_sources_roundtrip_and_preserve_investigation_inputs(tmp_path, dtype):
    template, arrays, data = planning_model(dtype=dtype)
    np.testing.assert_array_equal(arrays["P0"], np.diag(np.array([.01, .001, 1e-4, .1], dtype=dtype)))
    assert set(data) == {"e0", "e1", "e2"}
    assert data["e0"].dtype == dtype
    assert template.factors["tr1"].function.configuration == CONFIGURATION
    path = tmp_path / "reactor.cffgz"
    save_bundle(path, template, arrays)
    loaded = load_bundle(path)
    assert loaded.template.factors["tr1"].function == template.factors["tr1"].function
    for name, value in arrays.items():
        np.testing.assert_array_equal(loaded.arrays[name], value)
    with jax.enable_x64(dtype is np.float64):
        rebuilt = compile_template(loaded.template, arrays=loaded.arrays)
        assert rebuilt.gauss_groups
    legacy_template, legacy_arrays, legacy_data = planning_fixture(dtype=dtype)
    np.testing.assert_array_equal(legacy_arrays["P0"], np.diag(np.array([1., 1., 1e-4, 1.], dtype=dtype)))
    assert len(legacy_data) == 20 and legacy_data["e0"] is None
    assert legacy_template.factors["tr1"].function == template.factors["tr1"].function
    arrays["theta"][0] += 1
    _, independent_arrays, _ = planning_model(dtype=dtype)
    np.testing.assert_array_equal(independent_arrays["theta"], NOMINAL_PARAMETERS.astype(dtype))


def test_bundle_loading_requires_explicit_trusted_install_in_a_fresh_process(tmp_path):
    template, arrays, data = planning_model()
    path = tmp_path / "reactor.cffgz"
    save_bundle(path, template, arrays)
    example = Path(__file__).resolve().parents[1] / "examples/cstr_planning.py"
    source = """
import runpy
import sys
from cffg import load_bundle
try:
    load_bundle(sys.argv[1])
except ValueError as error:
    assert 'missing mean implementation' in str(error), str(error)
else:
    raise AssertionError('unknown callback was accepted')
runpy.run_path(sys.argv[2])['install_mean']()
bundle = load_bundle(sys.argv[1])
assert bundle.template.factors['tr1'].function.impl_id == 'cffg.example.cstr_rk4'
assert set(bundle.arrays) == set(bundle.template.blobs)
print('trusted reconstruction passed')
"""
    completed = subprocess.run([sys.executable, "-c", source, str(path), str(example)],
                               cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "trusted reconstruction passed" in completed.stdout


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_local_physics_against_independent_ode_at_declared_prior(dtype):
    with jax.enable_x64(dtype is np.float64):
        mean = np.zeros(4, dtype=dtype)
        offsets = 2 * np.sqrt(np.array([.01, .001, 1e-4, .1], dtype=dtype)) * np.eye(4, dtype=dtype)
        points = np.concatenate([mean[None], mean + offsets, mean - offsets])
        parameters = jnp.asarray(NOMINAL_PARAMETERS, dtype=dtype)
        values = np.asarray(jax.jit(jax.vmap(lambda x: cstr_rk4(x, parameters, CONFIGURATION)))(points))
        independent = np.stack([independent_transition(point, np.asarray(parameters)) for point in points])
        assert np.isfinite(values).all()
        np.testing.assert_allclose(values, independent, rtol=0., atol=max(1e-5, 512 * np.finfo(dtype).eps))


@pytest.mark.slow
@pytest.mark.parametrize("method", ["linearization", "sigma_point"])
def test_float32_affine_recipe_against_dense_conditioning(method):
    with jax.enable_x64(False):
        template, arrays, data = planning_model(method, dtype=np.float32, affine=True)
        engine = Engine(compile_template(template, arrays=arrays), schedule="flood")
        result = run_certified(engine, engine.bind_data(engine.init_state(), data), np.float32)
        mean, covariance, energy = affine_reference({k: v.astype(np.float64) for k, v in arrays.items()}, data)
        tolerance = 512 * np.finfo(np.float32).eps
        np.testing.assert_allclose(engine.bfe(result.state), energy, rtol=tolerance, atol=tolerance)
        for index in range(3):
            actual_mean, actual_covariance = engine.gaussian_marginals(result.state)[f"s{index}"]
            np.testing.assert_allclose(actual_mean, mean[index*4:(index+1)*4], rtol=tolerance, atol=tolerance)
            np.testing.assert_allclose(actual_covariance, covariance[index*4:(index+1)*4, index*4:(index+1)*4],
                                       rtol=tolerance, atol=tolerance)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["linearization", "sigma_point"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("schedule", ["auto", "flood"])
def test_default_reactor_fresh_and_warm(method, dtype, schedule):
    with jax.enable_x64(dtype is np.float64):
        template, arrays, data = planning_model(method, dtype=dtype)
        engine = Engine(compile_template(template, arrays=arrays), schedule=schedule)
        initial = engine.bind_data(engine.init_state(), data)
        fresh = run_certified(engine, initial, dtype)
        assert_physical_cavities(engine, fresh, dtype)
        run_certified(engine, fresh.state, dtype)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["linearization", "sigma_point"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_declared_temperature_goal_precision_grid_auto(method, dtype):
    with jax.enable_x64(dtype is np.float64):
        template, arrays, data = planning_model(method, dtype=dtype, temperature_variance=1.)
        engine = Engine(compile_template(template, arrays=arrays), schedule="auto")
        initial = engine.bind_data(engine.init_state(), data)
        for process_scale in (.01, 1., 100.):
            for temperature_variance in (.01, 1.):
                changes = {f"tr{i}": {"cov": arrays["Qp"] * process_scale} for i in (1, 2)}
                changes.update({f"em{i}": {"cov": np.diag([4e-4, temperature_variance, 1.]).astype(dtype)}
                               for i in (1, 2)})
                state = _replace(initial, params=set_params(engine, initial.params, changes))
                result = run_certified(engine, state, dtype)
                assert_physical_cavities(engine, result, dtype)

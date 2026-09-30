"""Classify reconstructed reactor failures without claiming incident replay."""

from copy import deepcopy
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import Engine, FailureDiagnostic, RunStatus, compile_template, set_params
from cffg.core import BlobRef
from cffg.execution.transport import _gauss_rows
from cffg.execution.failures import FailureIndex
from cffg.failure_facts import FailureReason
from cffg.nonlinear import NonlinearFailure, local_result
from cffg.runtime_types import RunResult, _replace
from cstr_planning_fixture import CONFIGURATION, NOMINAL_PARAMETERS, cstr_rk4, planning_fixture


def cavity(engine, state, factor):
    group = next(group for group in engine.arena.gauss_groups if factor in group.members)
    index = group.members.index(factor)
    eta, precision = _gauss_rows(engine.arena, state.g_eta, state.g_lam, group.gather, group.dims)
    return eta[index], precision[index]


def assert_partial_cavity(engine, result):
    assert result.status is RunStatus.DIVERGED
    assert result.failure.reason == "improper_cavity"
    eta, precision = cavity(engine, result.state, result.failure.factor)
    assert np.isfinite(eta).all()
    eigenvalues = np.linalg.eigvalsh(precision[1])
    assert eigenvalues[0] == 0.
    assert np.count_nonzero(eigenvalues > 0.) in (2, 3)
    assert np.all(eigenvalues >= 0.)
    return eta, precision


def strict_failure(engine, state, budget):
    """Expose strict local-update failures, without admission or trust discovery."""
    if engine._prime is not None:
        values = engine._prime(state.messages, state.g_eta, state.g_lam, state.ef_eta,
                               state.g_obs, state.g_mask, state.ef_obs, state.ef_mask, state.params)
        state = _replace(state, messages=values[0], g_eta=values[1], g_lam=values[2], ef_eta=values[3])
    names = FailureIndex(engine.arena).names
    for iteration in range(1, budget + 1):
        values = engine._iter_step(state.messages, state.g_eta, state.g_lam, state.g_obs, state.g_mask,
                                   state.ef_eta, state.ef_obs, state.ef_mask, state.params, 0., False)
        if int(values[6].reason) != 0:
            fact = values[6]
            diagnostic = FailureDiagnostic(FailureReason(int(fact.reason)).name.lower(),
                                            names[int(fact.factor)], "soft", iteration)
            return RunResult(state, RunStatus.DIVERGED, iteration, (), (), float("inf"), failure=diagnostic)
        state = _replace(state, messages=values[0], g_eta=values[1], g_lam=values[2], ef_eta=values[3])
    raise AssertionError("strict update fixture did not reproduce a local failure")


@pytest.mark.slow
@pytest.mark.parametrize("method", ["linearization", "sigma_point"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_strict_updates_reject_partial_goals_before_a_forward_prior_arrives(method, dtype):
    with jax.enable_x64(dtype is np.float64):
        template, arrays, data = planning_fixture(method, past=0, horizon=2, dtype=dtype)
        engine = Engine(compile_template(template, arrays=arrays))
        initial = engine.bind_data(engine.init_state(), data)
        result = strict_failure(engine, initial, 20)
        eta, precision = assert_partial_cavity(engine, result)
        assert result.failure.factor == "tr2"
        # Goal observes concentration and action, never the disturbance coordinate.
        np.testing.assert_allclose(np.linalg.eigvalsh(precision[1]), [0., 0., 1., 2500.])


@pytest.mark.parametrize("method", ["linearization", "sigma_point"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_partial_information_is_not_a_normalized_gaussian(method, dtype):
    with jax.enable_x64(dtype is np.float64):
        eta = jnp.array([1., 0., 0., 1.], dtype=dtype)
        precision = jnp.diag(jnp.array([2500., 0., 0., 1.], dtype=dtype))
        update = lambda ei, li: local_result(
            lambda x, p: jnp.full_like(x, jnp.nan), ei, li, jnp.zeros(4, dtype=dtype),
            jnp.zeros((4, 4), dtype=dtype), jnp.zeros(0, dtype=dtype),
            jnp.eye(4, dtype=dtype), method,
        )
        host = update(eta, precision)
        batched = jax.jit(jax.vmap(update))(jnp.stack([eta, eta]), jnp.stack([precision, precision]))
        assert int(host["failure_code"]) == NonlinearFailure.IMPROPER_CAVITY
        np.testing.assert_array_equal(batched["failure_code"], [int(host["failure_code"])] * 2)
        assert np.isnan(batched["eta_out"]).all()
        vacuous = update(jnp.zeros_like(eta), jnp.zeros_like(precision))
        assert int(vacuous["failure_code"]) == NonlinearFailure.VACUOUS_INPUT
        assert int(vacuous["status"]) == 0 and not vacuous["proper"]


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_retained_rk4_exceeds_sigma_transform_range_at_a_wide_prior(dtype):
    with jax.enable_x64(dtype is np.float64):
        evaluate = lambda x: cstr_rk4(x, jnp.asarray(NOMINAL_PARAMETERS, dtype=dtype), CONFIGURATION)
        center = jnp.zeros(4, dtype=dtype)
        assert np.isfinite(evaluate(center)).all()
        # P0 temperature variance one; sqrt(d)=2 gives normalized T=2 (375 K).
        broad_point = center.at[1].set(2.)
        value = np.asarray(evaluate(broad_point))
        if dtype is np.float32:
            assert not np.isfinite(value).all()
        else:
            assert np.isfinite(value).all()
            assert np.max(np.abs(value)) > np.sqrt(np.finfo(dtype).max)
        result = local_result(
            lambda x, p: cstr_rk4(x, p, CONFIGURATION), center,
            jnp.diag(jnp.array([1., 1., 1e4, 1.], dtype=dtype)),
            center, jnp.zeros((4, 4), dtype=dtype), jnp.asarray(NOMINAL_PARAMETERS, dtype=dtype),
            jnp.diag(jnp.array([1e-4, 4e-4, 4e-6, .25], dtype=dtype)), "sigma_point",
        )
        assert int(result["failure_code"]) == NonlinearFailure.NONFINITE_FUNCTION
        assert int(result["status"]) == 2 and not result["proper"]


@pytest.mark.slow
@pytest.mark.parametrize("method", ["linearization", "sigma_point"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("schedule", ["auto", "flood"])
def test_strict_precision_grid_records_distinct_failure_mechanisms(method, dtype, schedule):
    with jax.enable_x64(dtype is np.float64):
        template, arrays, data = planning_fixture(method, temperature_variance=1., dtype=dtype)
        engine = Engine(compile_template(template, arrays=arrays), schedule=schedule)
        initial = engine.bind_data(engine.init_state(), data)
        for process_scale in (.01, 1., 100.):
            for temperature_variance in (.01, 1.):
                changes = {f"tr{index}": {"cov": arrays["Qp"] * process_scale} for index in range(5, 20)}
                changes.update({f"em{index}": {"cov": np.diag([4e-4, temperature_variance, 1.]).astype(dtype)}
                               for index in range(5, 20)})
                state = _replace(initial, params=set_params(engine, initial.params, changes))
                result = strict_failure(engine, state, 20)
                if method == "linearization":
                    _, precision = assert_partial_cavity(engine, result)
                    assert result.failure.factor == "tr10"
                    np.testing.assert_allclose(np.linalg.eigvalsh(precision[1]),
                                               [0., 1., 1 / temperature_variance, 2500.])
                else:
                    assert result.status is RunStatus.DIVERGED
                    assert result.failure.reason == "nonfinite_function"
                    assert result.failure.factor == "tr1"
                    _, precision = cavity(engine, result.state, "tr1")
                    np.testing.assert_allclose(np.linalg.eigvalsh(precision[1]), [1., 1., 1., 1e4])
                assert result.iterations <= 3


def linear_equivalent(template, arrays):
    template, arrays = deepcopy(template), dict(arrays)
    arrays.update(A=arrays["theta"][:16].reshape(4, 4), b=arrays["theta"][16:])
    template.blobs.update({name: BlobRef(name, name + ".npy") for name in ("A", "b")})
    for name, factor in tuple(template.factors.items()):
        if factor.type == "gaussian_nonlinear":
            template.factors[name] = replace(factor, type="gaussian_linear", function=None,
                                             param_blobs={"matrix": "A", "offset": "b", "cov": factor.param_blobs["cov"]})
    return template, arrays


def transfer_messages(source, state, target, initial):
    messages = {}
    for group in source.arena.gauss_groups:
        for row, name in enumerate(group.members):
            for column, role in enumerate(group.roles):
                slot, dimension = group.scatter[row, column], group.dims[column]
                messages[name, role] = (state.g_eta[source.arena.g_eta_indices(slot, dimension)],
                                        state.g_lam[source.arena.g_lam_indices(slot, dimension)])
    eta, precision = initial.g_eta, initial.g_lam
    for group in target.arena.gauss_groups:
        for row, name in enumerate(group.members):
            for column, role in enumerate(group.roles):
                slot, dimension = group.scatter[row, column], group.dims[column]
                incoming_eta, incoming_precision = messages[name, role]
                eta = eta.at[target.arena.g_eta_indices(slot, dimension)].set(incoming_eta)
                precision = precision.at[target.arena.g_lam_indices(slot, dimension)].set(incoming_precision)
    return _replace(initial, g_eta=eta, g_lam=precision)


def affine_reference(arrays, data):
    """Independent dense Gaussian conditioning for the past=0 fixture."""
    dimension, length = 4, len(data)
    matrix, offset = arrays["theta"][:16].reshape(4, 4), arrays["theta"][16:]
    mean = np.zeros(length * dimension)
    covariance = np.zeros((len(mean), len(mean)))
    mean[:dimension], covariance[:dimension, :dimension] = arrays["m0"], arrays["P0"]
    for index in range(1, length):
        start, end = index * dimension, (index + 1) * dimension
        mean[start:end] = matrix @ mean[start-dimension:start] + offset
        covariance[start:end, :start] = matrix @ covariance[start-dimension:start, :start]
        covariance[:start, start:end] = covariance[start:end, :start].T
        covariance[start:end, start:end] = arrays["Qp"] + matrix @ covariance[start-dimension:start, start-dimension:start] @ matrix.T
    rows, noise, observations = [], [], []
    for index in range(length):
        kind = "now" if index == 0 else "goal"
        projection = arrays[f"G_{kind}"]
        row = np.zeros((len(projection), len(mean)))
        row[:, index*dimension:(index+1)*dimension] = projection
        rows.append(row)
        noise.extend(np.diag(arrays[f"R_{kind}"]))
        observations.extend(data[f"e{index}"])
    projection = np.concatenate(rows)
    innovation = np.diag(noise) + projection @ covariance @ projection.T
    residual = np.asarray(observations) - projection @ mean
    gain = np.linalg.solve(innovation, projection @ covariance).T
    posterior_mean = mean + gain @ residual
    posterior_covariance = covariance - gain @ projection @ covariance
    energy = .5 * (len(residual) * np.log(2 * np.pi) + np.linalg.slogdet(innovation)[1]
                    + residual @ np.linalg.solve(innovation, residual))
    return posterior_mean, posterior_covariance, energy


@pytest.mark.slow
@pytest.mark.parametrize("method", ["linearization", "sigma_point"])
def test_affine_limit_agrees_from_fresh_and_exact_proper_initialization(method):
    template, arrays, data = planning_fixture(method, past=0, horizon=2, affine=True)
    linear_template, linear_arrays = linear_equivalent(template, arrays)
    reference = Engine(compile_template(linear_template, arrays=linear_arrays))
    expected = reference.run(reference.bind_data(reference.init_state(), data), 100, tol=1e-9)
    expected.require_converged()
    mean, covariance, energy = affine_reference(arrays, data)
    np.testing.assert_allclose(reference.bfe(expected.state), energy, atol=1e-8)
    for index in range(3):
        actual_mean, actual_covariance = reference.gaussian_marginals(expected.state)[f"s{index}"]
        np.testing.assert_allclose(actual_mean, mean[index*4:(index+1)*4], atol=1e-8)
        np.testing.assert_allclose(actual_covariance, covariance[index*4:(index+1)*4, index*4:(index+1)*4], atol=1e-8)
    nonlinear = Engine(compile_template(template, arrays=arrays))
    initial = nonlinear.bind_data(nonlinear.init_state(), data)
    fresh = nonlinear.run(initial, 20, tol=1e-9)
    fresh.require_converged()
    np.testing.assert_allclose(nonlinear.bfe(fresh.state), energy, atol=1e-8)
    for index in range(3):
        actual_mean, actual_covariance = nonlinear.gaussian_marginals(fresh.state)[f"s{index}"]
        np.testing.assert_allclose(actual_mean, mean[index*4:(index+1)*4], atol=1e-8)
        np.testing.assert_allclose(actual_covariance, covariance[index*4:(index+1)*4, index*4:(index+1)*4], atol=1e-8)
    seeded = transfer_messages(reference, expected.state, nonlinear, initial)
    result = nonlinear.run(seeded, 30, tol=1e-9)
    result.require_converged()
    assert result.outer_residual <= 1e-9
    np.testing.assert_allclose(nonlinear.bfe(result.state), reference.bfe(expected.state), atol=1e-8)
    actual, wanted = nonlinear.gaussian_marginals(result.state), reference.gaussian_marginals(expected.state)
    for name in actual:
        np.testing.assert_allclose(actual[name][0], wanted[name][0], atol=1e-8)
        np.testing.assert_allclose(actual[name][1], wanted[name][1], atol=1e-8)


@pytest.mark.slow
def test_tenfold_horizon_preserves_strict_partial_input_validation():
    template, arrays, data = planning_fixture(past=0, horizon=150)
    engine = Engine(compile_template(template, arrays=arrays))
    result = strict_failure(engine, engine.bind_data(engine.init_state(), data), 4)
    assert_partial_cavity(engine, result)
    assert result.iterations == 1

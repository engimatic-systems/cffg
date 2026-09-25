"""Complete adaptive neural derivatives against independently re-solved energies."""

from dataclasses import replace

import jax
import numpy as np
import pytest
from nonlinear_fixture import adaptive_pair
from test_neural import neural_model

from cffg import neural_mean, set_params
from cffg.compiler import compile_template
from cffg.engine import Engine, RunStatus
from cffg.implicit import GradientStatus, fixed_point_gradient
from cffg.runtime_types import _replace
from cffg.trainable import NeuralParameters


def coupled_neural_model(method):
    template, arrays = adaptive_pair(method)
    for name in ("sensor1", "sensor2"):
        template.factors[name] = replace(template.factors[name], function=neural_mean((1, 1, 1), method=method))
    arrays["theta"] = np.array([.8, .15, 1.1, -.05])
    return template, arrays


@pytest.mark.slow
@pytest.mark.parametrize("method", ["linearization", "sigma_point"])
def test_complete_tied_neural_gradient_matches_resolved_central_differences(method):
    template, arrays = coupled_neural_model(method)
    engine = Engine(compile_template(template, arrays=arrays))
    result = engine.run(engine.init_state(), 50, tol=1e-10)
    state = result.require_converged()
    derivative = fixed_point_gradient(engine, result, tolerance=1e-8)
    actual = np.asarray(derivative.require_valid())
    plan = derivative.parameters
    assert len(plan.blobs) == 1 and len(plan.blobs[0].rows) == 2
    theta = np.asarray(derivative.weights)

    def energy(weights):
        updated = _replace(state, params=plan.inject(weights, state.params))
        solved = engine.run(updated, 50, tol=1e-10).require_converged()
        return engine.bfe(solved)

    finite_difference = []
    for index in range(theta.size):
        step = 2e-4 * (1 + abs(theta[index]))
        offset = np.eye(theta.size)[index] * step
        finite_difference.append((energy(theta + offset) - energy(theta - offset)) / (2 * step))
    np.testing.assert_allclose(actual, finite_difference, rtol=2e-3, atol=2e-5)
    assert derivative.adjoint_residual < 1e-8
    step = 1e-4 * (1 + abs(theta[0]))
    offset = np.eye(theta.size)[0] * step
    refined = (energy(theta + offset) - energy(theta - offset)) / (2 * step)
    np.testing.assert_allclose(refined, finite_difference[0], rtol=2e-3, atol=2e-5)
    limited = fixed_point_gradient(engine, result, tolerance=1e-10,
                                   max_iterations=1, krylov_dimension=1)
    assert limited.status == GradientStatus.ADJOINT_BUDGET


def test_stale_or_unsettled_states_refuse_an_applicable_gradient():
    template, arrays = neural_model()
    engine = Engine(compile_template(template, arrays=arrays))
    initial = engine.bind_data(engine.init_state(), {"y": np.array([.4])})
    pending = engine.run(initial, 0, tol=1e-9)
    assert fixed_point_gradient(engine, pending).status == GradientStatus.UNCERTIFIED
    result = engine.run(initial, 30, tol=1e-9)
    state = result.require_converged()
    params = set_params(engine, state.params, {"sensor": {"parameters": arrays["theta"] + .5}})
    stale = replace(result, state=_replace(state, params=params))
    refused = fixed_point_gradient(engine, stale)
    assert refused.status == GradientStatus.UNCERTIFIED
    with pytest.raises(ValueError, match="current weights"):
        refused.require_valid()
    assert fixed_point_gradient(engine, result, max_iterations=0).status == GradientStatus.ADJOINT_BUDGET


def test_tied_weight_injection_sums_reader_derivatives():
    template, arrays = coupled_neural_model("linearization")
    engine = Engine(compile_template(template, arrays=arrays))
    params = engine.init_state().params
    plan = NeuralParameters.from_engine(engine, params, ["sensor1"])
    weights = plan.extract(params)
    rows = plan.blobs[0].rows

    def cost(theta):
        injected = plan.inject(theta, params)
        return sum((index + 1) * injected[key][row].sum() for index, (key, row) in enumerate(rows))

    np.testing.assert_array_equal(jax.grad(cost)(weights), np.full(weights.shape, 3.))


def test_float32_gradient_uses_dtype_scaled_certificates():
    with jax.enable_x64(False):
        template, arrays = neural_model()
        engine = Engine(compile_template(template, arrays=arrays))
        initial = engine.bind_data(engine.init_state(), {"y": np.array([.4])})
        result = engine.run(initial, 30, tol=2e-5)
        state = result.require_converged()
        derivative = fixed_point_gradient(engine, result)
        gradient = derivative.require_valid()
        theta = np.asarray(derivative.weights)
        eps = np.finfo(theta.dtype).eps
        step = eps ** (1 / 3) * (1 + abs(theta[0]))
        offset = np.eye(theta.size, dtype=theta.dtype)[0] * step
        energies = []
        for value in (theta + offset, theta - offset):
            trial = _replace(state, params=derivative.parameters.inject(value, state.params))
            settled = engine.run(trial, 30, tol=2e-5).require_converged()
            energies.append(engine.bfe(settled))
        reference = (energies[0] - energies[1]) / (2 * step)
        np.testing.assert_allclose(gradient[0], reference, atol=64 * eps ** (2 / 3), rtol=64 * eps ** (2 / 3))

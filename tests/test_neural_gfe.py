"""Sparse fixed-support reverse derivatives and complete hybrid neural energy."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import compile_template, Engine, fixed_point_gradient, neural_mean
from cffg import hybrid_gfe
from cffg.compiled import NEG
from cffg.core import BlobRef, FactorSpec, MomentProjection, VariableSpec, moment_matching
from cffg.runtime_types import _replace
from cffg.rules import gfe_tm_update
from test_hybrid_gfe import _linear_template


@pytest.mark.parametrize("dtype, regimes", [(jnp.float64, 2), (jnp.float32, 32)])
def test_sparse_hybrid_energy_reverse_derivative_matches_forward_mode(dtype, regimes):
    selector = np.ones(regimes) / (regimes - 1)
    selector[-1] = 0.
    log_incoming = jnp.asarray(np.where(selector > 0, np.log(np.where(selector > 0, selector, 1.)), NEG), dtype)[None]
    matrix = jnp.linspace(.2, .8, regimes, dtype=dtype).reshape(1, regimes, 1, 1)
    offset = jnp.linspace(-.2, .3, regimes, dtype=dtype).reshape(1, regimes, 1)
    noise = jnp.full((1, regimes, 1, 1), 2., dtype)
    precision = jnp.ones((1, 1, 1), dtype)
    goal_mean, goal_precision = jnp.asarray([[.3]], dtype), jnp.asarray([[[.5]]], dtype)

    def energy(mean):
        return hybrid_gfe.linear_energy(log_incoming, mean.reshape(1, 1), precision,
            matrix, offset, noise, goal_mean, goal_precision).sum()

    mean = jnp.asarray(.2, dtype)
    value, forward = jax.jvp(energy, (mean,), (jnp.ones_like(mean),))
    backward = jax.jit(jax.grad(energy))(mean)
    assert np.isfinite([value, forward, backward]).all()
    np.testing.assert_allclose(backward, forward, rtol=256 * np.finfo(dtype).eps, atol=256 * np.finfo(dtype).eps)


def test_zero_predictive_row_keeps_transition_mixture_reverse_derivatives_finite():
    transition = jnp.asarray([[[[.7, .2], [.3, .8], [0., 0.]]]])
    log_transition = jnp.where(transition > 0, jnp.log(jnp.where(transition > 0, transition, 1.)), NEG)
    incoming = jnp.log(jnp.asarray([[.4, .6]]))
    downstream = jnp.asarray([[np.log(.4), np.log(.6), NEG]])

    def message(value):
        return gfe_tm_update(value, jnp.zeros((1, 1)), downstream, log_transition, 3)[0][0, 0]

    direction = jnp.asarray([[.3, -.2]])
    value, forward = jax.jvp(message, (incoming,), (direction,))
    backward = jnp.sum(jax.grad(message)(incoming) * direction)
    assert np.isfinite([value, forward, backward]).all()
    np.testing.assert_allclose(backward, forward, rtol=1e-10, atol=1e-10)


def neural_hybrid_model():
    template, arrays = _linear_template(matrix=np.array([[[.5]], [[.8]], [[.4]]]),
        offset=np.array([[.1], [-.3], [.5]]), noise=np.array([[[.4]], [[.6]], [[.7]]]),
        state_mean=np.array([.2]), state_cov=np.array([[.3]]),
        selector=np.array([.6, .4, 0.]), goal_mean=np.array([.1]), goal_cov=np.array([[1.5]]))
    template.variables["w"] = VariableSpec("w", dim=1)
    template.factors["state"] = replace(template.factors["state"], edges={"out": "w"})
    template.factors["neural"] = FactorSpec("neural", "gaussian_nonlinear", {"in": "w", "out": "z"},
        param_blobs={"parameters": "theta", "cov": "V"}, function=neural_mean((1, 1, 1)))
    arrays.update(theta=np.array([.8, .15, 1.1, -.05]), V=np.array([[.2]]))
    template.blobs.update({name: BlobRef(name, name + ".npy") for name in ("theta", "V")})
    template.constraints.append(moment_matching("neural", "in", MomentProjection.GAUSSIAN))
    return template, arrays


def test_complete_neural_hybrid_gradient_preserves_exact_support():
    template, arrays = neural_hybrid_model()
    engine = Engine(compile_template(template, arrays=arrays))
    result = engine.run(engine.init_state(), 50, tol=1e-10)
    state = result.require_converged()
    derivative = fixed_point_gradient(engine, result, tolerance=1e-8)
    gradient = derivative.require_valid()
    assert np.all(engine.marginals(state)["s"][:2] > 0)
    assert engine.marginals(state)["s"][2] == 0
    weights = np.asarray(derivative.weights)
    direction = np.array([.3, -.2, .4, .1])
    step = 2e-4
    energies = []
    for sign in (1, -1):
        params = derivative.parameters.inject(weights + sign * step * direction, state.params)
        settled = engine.run(_replace(state, params=params), 50, tol=1e-10).require_converged()
        assert engine.marginals(settled)["s"][2] == 0
        energies.append(engine.bfe(settled))
    reference = (energies[0] - energies[1]) / (2 * step)
    np.testing.assert_allclose(jnp.dot(gradient, direction), reference, rtol=2e-3, atol=2e-5)

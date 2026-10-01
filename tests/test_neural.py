"""Portable neural means reuse certified nonlinear Gaussian inference."""

from dataclasses import replace
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from nonlinear_fixture import model

from cffg import initialize_neural_parameters, neural_mean
from cffg.bundle import load_bundle, save_bundle
from cffg.compiler import CompileError, compile_template
from cffg.core.serialize import from_json, structure_hash, to_json
from cffg.engine import Engine
from cffg.estimate import estimate
from cffg.neural import neural_value
from cffg.neural_schema import neural_architecture


def neural_model(method="linearization", *, width=2):
    template, arrays = model(method, observed=True)
    function = neural_mean((1, width, 1), method=method)
    template.factors["sensor"] = replace(template.factors["sensor"], function=function)
    arrays["theta"] = np.asarray(initialize_neural_parameters(function, jax.random.key(71), dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32))
    return template, arrays


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_network_values_and_jacobians_match_independent_affine_tanh_formula(dtype):
    architecture = neural_architecture(neural_mean((2, 3, 2)).configuration)
    parameters = jnp.linspace(-.8, .9, architecture.parameter_count, dtype=dtype)
    x = jnp.asarray([.3, -.5], dtype)
    p = np.asarray(parameters, dtype=np.float64)
    w1, b1, w2, b2 = p[:6].reshape(3, 2), p[6:9], p[9:15].reshape(2, 3), p[15:]
    hidden = np.tanh(w1 @ np.asarray(x) + b1)
    expected = w2 @ hidden + b2
    jacobian = w2 @ np.diag(1 - hidden ** 2) @ w1
    evaluate = lambda x: neural_value(x, parameters, architecture)
    tolerance = 64 * np.finfo(np.dtype(dtype)).eps
    np.testing.assert_allclose(jax.jit(evaluate)(x), expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(jax.jacfwd(evaluate)(x), jacobian, rtol=tolerance, atol=tolerance)


def test_explicit_key_and_architecture_identity():
    function = neural_mean((1, 3, 1))
    first = initialize_neural_parameters(function, jax.random.key(3))
    np.testing.assert_array_equal(first, initialize_neural_parameters(function, jax.random.key(3)))
    assert not np.array_equal(first, initialize_neural_parameters(function, jax.random.key(4)))
    with pytest.raises(ValueError, match="positive integers"):
        neural_mean((1, True, 1))
    with pytest.raises(ValueError, match="activation"):
        neural_mean((1, 2, 1), activation="unknown")
    template, arrays = neural_model()
    template.factors["sensor"] = replace(template.factors["sensor"], function=neural_mean((2, 2, 1)))
    with pytest.raises(CompileError, match="dimensions"):
        compile_template(template, arrays=arrays)


@pytest.mark.parametrize("method", [
    "linearization", pytest.param("sigma_point", marks=pytest.mark.slow),
])
@pytest.mark.slow
def test_neural_bundle_reconstruction_and_schedule_parity(tmp_path, method):
    template, arrays = neural_model(method)
    path = tmp_path / "network.cffgz"
    save_bundle(path, template, arrays)
    restored = load_bundle(path)
    assert structure_hash(restored.template) == structure_hash(from_json(to_json(template)))
    np.testing.assert_array_equal(restored.arrays["theta"], arrays["theta"])
    results = []
    for source, values, schedule in ((template, arrays, "auto"), (restored.template, restored.arrays, "flood")):
        engine = Engine(compile_template(source, arrays=values), schedule=schedule)
        initial = engine.bind_data(engine.init_state(), {"y": np.array([.4])})
        result = engine.run(initial, 30, tol=1e-8)
        state = result.require_converged()
        results.append((state, engine.bfe(state), engine.gaussian_marginals(state)["x"]))
        assert engine.objective_metadata["sensor"]["objective"] == method + "_frozen_gaussian_bethe"
    for name in ("g_eta", "g_lam"):
        np.testing.assert_allclose(getattr(results[0][0], name), getattr(results[1][0], name), atol=1e-7)
    np.testing.assert_allclose(results[0][1], results[1][1], atol=1e-7)


def test_tenfold_parameter_scale_has_exact_memory_and_declared_work():
    template, arrays = neural_model(width=32)
    assert arrays["theta"].size >= 10 * 7
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.bind_data(engine.init_state(), {"y": np.array([.4])})
    started = time.monotonic()
    result = engine.run(state, 20, tol=1e-7)
    result.require_converged()
    assert time.monotonic() - started < 60
    report = estimate(engine, calibrate=False)
    assert report.state_bytes == sum(v.nbytes for v in jax.tree.leaves(state))
    assert report.state_bytes < 8192
    work = next(iter(report.declared_work.values()))
    assert work["members"] == 1 and work["network_parameters"] == 97
    assert work["dense_products_per_value"] == 64
    assert work["jacobian_forward_directions"] == 1

"""Local substituted moments, missing data, rejected topologies and independently bound lanes."""

import jax
import numpy as np
import pytest

from cffg import conj
from cffg.colony import Colony
from cffg.compiler import CompileError, compile_template
from cffg.core import LearnableSpec, observe_interior
from cffg.engine import Engine
from cffg.learn import Learnables
from cffg.models.kalman import lgssm


def chain(n):
    f = np.array([[.8, .2], [.1, .7]])
    return lgssm(n, f, np.diag([.25, .5]), np.array([[1., -.3]]), np.eye(1),
                 np.zeros(2), np.diag([2., .25]))


def test_substituted_sufficient_statistics_and_terminal_predictive():
    model, arrays = chain(3)
    model, p0 = observe_interior(model, "x0")
    model, p1 = observe_interior(model, "x1")
    e = Engine(compile_template(model, arrays=arrays))
    a, b, y = np.array([1., -.5]), np.array([.2, .4]), np.array([.75])
    s = e.run(e.bind_data(e.init_state(), {p0: a, p1: b, "y0": y}), 1, tol=1e-9).require_converged()
    stats = e.gaussian_node_stats(s)
    np.testing.assert_array_equal(stats["prior"]["second"], np.outer(a, a))
    for key, value in (("Syy", np.outer(b, b)), ("Syx", np.outer(b, np.r_[a, 1.])),
                       ("Sxx", np.outer(np.r_[a, 1.], np.r_[a, 1.]))):
        np.testing.assert_array_equal(stats["trans1"][key], value)
        assert stats["trans1"]["proper"]
    np.testing.assert_array_equal(stats["emis0"]["Syy"], np.outer(y, y))
    np.testing.assert_array_equal(stats["emis0"]["Syx"], np.outer(y, np.r_[a, 1.]))
    mean, cov = e.gaussian_marginal(s, "y0")
    np.testing.assert_allclose(mean, arrays["H"] @ a)
    np.testing.assert_allclose(cov, arrays["R"])
    # The unobserved last state has this predictive conditional on the known middle state.
    mx = arrays["F"] @ b
    np.testing.assert_allclose(stats["trans2"]["Syy"], arrays["Q"] + np.outer(mx, mx))


def test_centered_tap_statistics_drive_tied_niw_and_mniw_learning():
    template, arrays = chain(3)
    template, port0 = observe_interior(template, "x0")
    template, port1 = observe_interior(template, "x1")
    template.learnables = {
        "state": LearnableSpec("state", "niw", ("m0", "P0")),
        "motion": LearnableSpec("motion", "mniw", ("F", "zn", "Q")),
        "sensor": LearnableSpec("sensor", "mniw", ("H", "zm", "R")),
    }
    engine = Engine(compile_template(template, arrays=arrays))
    x0, x1, y0 = np.array([1., -.5]), np.array([.2, .4]), np.array([.75])
    data = engine.bind_data(engine.init_state(), {port0: x0, port1: x1, "y0": y0})
    result = engine.run(data, 1, tol=1e-9)
    centered = engine.gaussian_node_stats(result.require_converged(), centered=True)
    np.testing.assert_array_equal(centered["prior"]["mean"], x0)
    np.testing.assert_array_equal(centered["prior"]["covariance"], np.zeros((2, 2)))
    for name, x, y, cov_y in (("trans1", x0, x1, np.zeros((2, 2))),
                              ("trans2", x1, arrays["F"] @ x1, arrays["Q"])):
        assert centered[name]["proper"]
        np.testing.assert_allclose(centered[name]["mean"], np.r_[y, x])
        np.testing.assert_allclose(centered[name]["cov_yy"], cov_y)
        np.testing.assert_array_equal(centered[name]["cov_yx"], np.zeros((2, 2)))
        np.testing.assert_array_equal(centered[name]["cov_xx"], np.zeros((2, 2)))

    state_prior = conj.vague_niw(2)
    motion_prior = conj.vague_mniw(2, 2)
    sensor_prior = conj.vague_mniw(1, 2)
    learner = Learnables(engine, priors={"state": state_prior, "motion": motion_prior, "sensor": sensor_prior})
    learner.accumulate(engine, result)
    expected_state = state_prior.update(x0[None], np.outer(x0, x0)[None])
    x2 = arrays["F"] @ x1
    inputs = np.array([np.r_[x0, 1.], np.r_[x1, 1.]])
    outputs = np.array([x1, x2])
    expected_motion = motion_prior.update(outputs.T @ outputs + arrays["Q"], outputs.T @ inputs,
                                          inputs.T @ inputs, 2.)
    # Only the first DATA sensor was observed; its two tied missing siblings have zero weight.
    expected_sensor = sensor_prior.update(np.outer(y0, y0), np.outer(y0, inputs[0]),
                                          np.outer(inputs[0], inputs[0]), 1.)
    for name, expected in (("state", expected_state), ("motion", expected_motion), ("sensor", expected_sensor)):
        for key, actual in learner.params[name].items():
            np.testing.assert_allclose(actual, getattr(expected, key), atol=1e-10)

    # Release the input, keep only the output tap; x0 now has conditional uncertainty.
    data = engine.bind_data(data, {port0: None, "y0": None})
    state = engine.run(data, 1, tol=1e-9).require_converged()
    row = engine.gaussian_node_stats(state, centered=True)["trans1"]
    gain = arrays["F"].T @ np.linalg.inv(arrays["Q"])
    cov = np.linalg.inv(np.linalg.inv(arrays["P0"]) + gain @ arrays["F"])
    np.testing.assert_allclose(row["mean"], np.r_[x1, cov @ gain @ x1])
    np.testing.assert_allclose(row["cov_xx"], cov)
    np.testing.assert_array_equal(row["cov_yy"], np.zeros((2, 2)))
    np.testing.assert_array_equal(row["cov_yx"], np.zeros((2, 2)))


def test_independent_masks_per_colony_lane_and_partial_unbind():
    template, arrays = chain(3)
    template, port = observe_interior(template, "x1")
    c = Colony()
    first = c.spawn(template, arrays, observations={port: [1., 2.]})
    second = c.spawn(template, arrays, observations={"y1": [.5]})
    for result in c.run(12, tol=1e-9).values():
        result.require_converged()
    np.testing.assert_array_equal(c.gaussian_marginals(first, ["x1"])["x1"][1], 0.)
    cov = c.gaussian_marginals(second, ["x1"])["x1"][1]
    assert np.linalg.eigvalsh(cov).min() > 0
    c.bind(first, {port: None, "y1": [.5]})
    for result in c.run(12, tol=1e-9).values():
        result.require_converged()
    np.testing.assert_allclose(c.gaussian_marginals(first)["x1"][1], cov, atol=1e-10)


def test_reject_duplicate_sensor_and_unsupported_endpoint():
    template, arrays = chain(2)
    template, _ = observe_interior(template, "x0")
    template, _ = observe_interior(template, "x0c")
    with pytest.raises(CompileError, match="two observations on the same equality class"):
        compile_template(template, arrays=arrays)

    from cstr_planning_fixture import planning_fixture
    model, params, _ = planning_fixture(horizon=1)
    model, _ = observe_interior(model, "s0")
    with pytest.raises(CompileError, match="unsupported endpoint.*gaussian_nonlinear"):
        compile_template(model, arrays=params)


def test_invalid_tap_values_are_rejected_before_state_mutation():
    with jax.enable_x64(False):
        template, arrays = chain(2)
        template, port = observe_interior(template, "x0")
        e = Engine(compile_template(template, arrays=arrays))
        state = e.init_state()
        for value in ([np.nan, 0.], [np.inf, 0.], [1e100, 0.], np.array([1j, 0.]), [0.]):
            with pytest.raises(ValueError, match="port"):
                e.bind_data(state, {"y0": [1.], port: value})
        np.testing.assert_array_equal(state.g_mask, 0.)

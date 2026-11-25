"""Local substituted moments, missing data, rejected topologies and independently bound lanes."""

import jax
import numpy as np
import pytest

from cffg.colony import Colony
from cffg.compiler import CompileError, compile_template
from cffg.core import observe_interior
from cffg.engine import Engine
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

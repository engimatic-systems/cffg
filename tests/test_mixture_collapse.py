"""Checked Gaussian projection, independent moment oracles, and failure routing."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_hybrid_engine import mixlin_template

from cffg import Engine, RunStatus, compile_template, mixg
from cffg.colony import Colony
from cffg.compiled import NEG


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("components,dimension", [(2, 3), (32, 30)])
def test_checked_collapse_matches_independent_multidimensional_moments(dtype, components, dimension):
    rng = np.random.default_rng(11)
    mean = rng.normal(size=(components, dimension)).astype(dtype)
    variance = rng.uniform(.5, 2., size=(components, dimension)).astype(dtype)
    precision = np.eye(dimension, dtype=dtype)[None] / variance[:, :, None]
    eta = mean / variance
    weights = np.arange(1, components + 1, dtype=dtype)
    weights /= weights.sum()
    log_weights = np.log(weights)
    reference_mean = weights.astype(np.float64) @ mean.astype(np.float64)
    centered = mean.astype(np.float64) - reference_mean
    reference_cov = np.diag(weights.astype(np.float64) @ variance.astype(np.float64))
    reference_cov += sum(float(w) * np.outer(m, m) for w, m in zip(weights, centered))
    reference_precision = np.linalg.inv(reference_cov)
    tolerance = 64 * dimension * np.finfo(dtype).eps
    for rule in (mixg._collapse, jax.jit(mixg._collapse)):
        result = rule(jnp.asarray(log_weights), jnp.asarray(eta), jnp.asarray(precision))
        assert bool(result.valid)
        np.testing.assert_allclose(result.precision, reference_precision, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(result.eta, reference_precision @ reference_mean,
                                   rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("zero", [-np.inf, NEG])
def test_zero_support_is_masked_before_moment_arithmetic(zero):
    # Both active and inactive huge means would overflow squared float32 moments.
    means = jnp.array([[1e30], [-1e30]], jnp.float32)
    precision = jnp.ones((2, 1, 1), jnp.float32)
    result = jax.jit(mixg._collapse)(jnp.array([0., zero], jnp.float32), means, precision)
    assert bool(result.valid)
    np.testing.assert_array_equal(result.eta, means[0])
    np.testing.assert_array_equal(result.precision, precision[0])


@pytest.mark.parametrize("dtype,large", [(np.float32, 1e30), (np.float64, 1e200)])
def test_overflow_never_exposes_the_safe_inverse_placeholder(dtype, large):
    means = jnp.asarray([[large], [-large]], dtype)
    precision = jnp.ones((2, 1, 1), dtype)
    result = jax.jit(mixg._collapse)(jnp.zeros(2, dtype), means, precision)
    assert not bool(result.valid)
    assert np.isnan(result.eta).all()
    assert np.isnan(result.precision).all()


@pytest.mark.parametrize("problem", ["weights", "empty_support", "mean", "precision"])
def test_invalid_inputs_are_not_replaced_by_finite_information(problem):
    weights = jnp.zeros(2)
    means = jnp.zeros((2, 2))
    precision = jnp.broadcast_to(jnp.eye(2), (2, 2, 2))
    if problem == "weights":
        weights = weights.at[0].set(jnp.nan)
    elif problem == "empty_support":
        weights = jnp.full_like(weights, -jnp.inf)
    elif problem == "mean":
        means = means.at[0, 0].set(jnp.inf)
    else:
        precision = precision.at[0, 0, 0].set(-1.)
    result = mixg._collapse(weights, means, precision)
    assert not bool(result.valid)
    assert np.isnan(result.precision).all()


def test_vmap_keeps_failed_and_valid_collapses_separate():
    means = jnp.array([[[0.], [2.]], [[1e30], [-1e30]]], jnp.float32)
    precision = jnp.ones((2, 2, 1, 1), jnp.float32)
    weights = jnp.zeros((2, 2), jnp.float32)
    batched = jax.jit(jax.vmap(mixg._collapse))(weights, means, precision)
    np.testing.assert_array_equal(batched.valid, [True, False])
    for index in range(2):
        single = mixg._collapse(weights[index], means[index], precision[index])
        np.testing.assert_allclose(batched.eta[index], single.eta, equal_nan=True)
        np.testing.assert_allclose(batched.precision[index], single.precision, equal_nan=True)


def test_proper_collapse_derivatives_match_closed_form_moments():
    weights = jnp.array([.3, .7])
    precision = jnp.ones((2, 1, 1))

    def production(means):
        result = mixg._collapse(jnp.log(weights), means[:, None], precision)
        return jnp.stack((result.eta[0], result.precision[0, 0]))

    def independent(means):
        mean = weights @ means
        variance = 1. + weights @ ((means - mean) ** 2)
        return jnp.stack((mean / variance, 1. / variance))

    point = jnp.array([-1., 2.])
    for derivative in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(jax.jit(derivative(production))(point),
                                   derivative(independent)(point), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("direction", ["prior", "forward", "backward"])
def test_all_message_callers_reject_numerical_collapse_failure(direction):
    means = jnp.array([[[1e30], [-1e30]]], jnp.float32)
    precision = jnp.ones((1, 2, 1, 1), jnp.float32)
    weights = jnp.zeros((1, 2), jnp.float32)
    if direction == "prior":
        result = mixg.mix_forward(weights, means, precision)
    else:
        rule = mixg.mixlin_forward if direction == "forward" else mixg.mixlin_backward
        result = rule(jnp.zeros((1, 1), jnp.float32), jnp.ones((1, 1, 1), jnp.float32),
                      weights, jnp.ones_like(precision), means, precision)
    assert all(np.isnan(value).all() for value in result)


@pytest.mark.parametrize("zero", [-np.inf, NEG])
@pytest.mark.parametrize("direction", ["prior", "forward", "backward"])
def test_callers_mask_zero_support_before_scores_and_derivatives(zero, direction):
    means = jnp.array([[[0.], [1e30]]], jnp.float32)
    precision = jnp.ones((1, 2, 1, 1), jnp.float32)
    weights = jnp.array([[0., zero]], jnp.float32)

    def evaluate(offsets):
        if direction == "prior":
            eta, lam = mixg.mix_forward(weights, offsets, precision)
        else:
            rule = mixg.mixlin_forward if direction == "forward" else mixg.mixlin_backward
            eta, lam = rule(jnp.zeros((1, 1), jnp.float32), jnp.ones((1, 1, 1), jnp.float32),
                            weights, jnp.ones_like(precision), offsets, precision)
        return jnp.concatenate((eta.ravel(), lam.ravel()))

    np.testing.assert_allclose(evaluate(means), [0., 1. if direction == "prior" else .5],
                               rtol=1e-6)
    jacobian = jax.jit(jax.jacrev(evaluate))(means)
    assert np.isfinite(jacobian).all()
    np.testing.assert_array_equal(jacobian[:, 0, 1], 0.)


def test_inactive_rank_deficient_backward_component_does_not_erase_active_information():
    eta, precision = mixg.mixlin_backward(
        jnp.zeros((1, 1)), jnp.ones((1, 1, 1)), jnp.array([[0., NEG]]),
        jnp.array([[[[1.]], [[0.]]]]), jnp.zeros((1, 2, 1)), jnp.ones((1, 2, 1, 1)),
    )
    np.testing.assert_allclose(eta, 0., atol=0)
    np.testing.assert_allclose(precision, .5, rtol=1e-12)


@pytest.mark.parametrize("schedule", ["auto", "flood"])
def test_actual_failed_factor_has_public_attribution(schedule):
    with jax.enable_x64(False):
        template, arrays = mixlin_template()
        arrays["b"] = np.array([[1e30], [-1e30]], np.float32)
        engine = Engine(compile_template(template, arrays=arrays), schedule=schedule)
        result = engine.run(engine.init_state(), 5, tol=1e-5)
        assert result.status is RunStatus.DIVERGED
        assert result.failure.factor == "mix"
        assert result.failure.reason == "nonfinite_output"


def test_colony_does_not_hide_bad_collapse_or_contaminate_healthy_member():
    with jax.enable_x64(False):
        template, arrays = mixlin_template()
        bad = {**arrays, "b": np.array([[1e30], [-1e30]], np.float32)}
        colony = Colony()
        ids = [colony.spawn(template, params) for params in (arrays, bad)]
        results = colony.run(10, tol=1e-5)
        assert results[ids[0]].status is RunStatus.CONVERGED
        assert results[ids[0]].failure is None
        assert results[ids[1]].status is RunStatus.DIVERGED
        assert results[ids[1]].failure.factor == "mix"

"""D10 objective, learning, and numerical-contract checks."""

import math

import jax
import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from cffg import structured_hybrid as hybrid


def _gaussian_canonical(mean, covariance):
    mean = np.asarray(mean)
    covariance = np.asarray(covariance)
    precision = np.linalg.inv(covariance)
    eta = precision @ mean
    constant = (
        -0.5 * mean @ precision @ mean
        - 0.5 * mean.size * np.log(2 * np.pi)
        - 0.5 * np.linalg.slogdet(covariance)[1]
    )
    return eta, precision, constant


def _observed_linear_canonical(observation, matrix, covariance):
    observation = np.asarray(observation)
    matrix = np.asarray(matrix)
    covariance = np.asarray(covariance)
    precision = np.linalg.inv(covariance)
    eta = matrix.T @ precision @ observation
    natural_precision = matrix.T @ precision @ matrix
    constant = (
        -0.5 * observation @ precision @ observation
        - 0.5 * observation.size * np.log(2 * np.pi)
        - 0.5 * np.linalg.slogdet(covariance)[1]
    )
    return eta, natural_precision, constant


def _small_model(dtype=np.float64):
    time, card, dimension = 3, 2, 1
    external_eta = np.asarray([[0.25], [-0.15], [0.35]], dtype=dtype)
    external_precision = np.asarray([[[1.3]], [[0.8]], [[1.1]]], dtype=dtype)
    return hybrid.StructuredHybridModel(
        np.log(np.asarray([0.58, 0.42], dtype=dtype)),
        np.log(
            np.asarray(
                [[0.75, 0.25], [0.35, 0.65], [0.55, 0.45]], dtype=dtype
            )
        ),
        external_eta,
        external_precision,
        np.zeros(time, dtype=dtype),
        np.asarray([[[0.72]], [[-0.28]]], dtype=dtype),
        np.asarray([[0.12], [-0.18]], dtype=dtype),
        np.asarray([[[0.45]], [[0.8]]], dtype=dtype),
        np.asarray([[[1.05]], [[-0.85]]], dtype=dtype),
        np.asarray([[-0.12], [0.3]], dtype=dtype),
    )


def _solve(model, *, iterations=80, tolerance=1e-11):
    belief = hybrid.structured_hybrid_initial(model)
    sweep = jax.jit(hybrid.structured_hybrid_sweep)
    objectives = []
    result = None
    for _ in range(iterations):
        result = sweep(belief, model)
        assert int(result.status) == hybrid.SWEEP_FINITE
        objectives.append(float(result.objective))
        belief = result.belief
        if float(result.residual) <= tolerance:
            break
    assert result is not None
    return result, np.asarray(objectives)


def test_complete_expanded_sweeps_are_monotone_and_reach_a_joint_fixed_point():
    result, objectives = _solve(_small_model())
    assert len(objectives) > 1
    assert np.all(np.diff(objectives) <= 2e-12)
    assert float(result.categorical_objective) <= float(
        result.previous_objective
    ) + 2e-12
    assert float(result.fixed_auxiliary_objective) <= float(
        result.categorical_objective
    ) + 2e-12
    assert float(result.objective) <= float(
        result.fixed_auxiliary_objective
    ) + 2e-12
    assert float(result.residual) < 1e-9


def test_objective_explicitly_owns_arbitrary_categorical_unary_evidence():
    result, _ = _solve(_small_model())
    model = _small_model()
    delta = np.asarray([[0.2, -0.1], [-0.3, 0.4], [0.15, -0.25]])
    changed = hybrid.structured_hybrid_objective(
        result.belief.categorical,
        result.belief.gaussian,
        model.log_initial,
        model.log_unary + delta,
        model.gaussian_eta,
        model.gaussian_precision,
        model.gaussian_log_constant,
        result.dynamics_log_potential,
        result.recurrence.log_potential,
    )
    expected_change = -np.einsum(
        "tk,tk->", result.belief.categorical.singleton, delta
    )
    assert float(changed - result.objective) == pytest.approx(
        expected_change, abs=2e-13
    )


def _normal_expectation(mean, variance, function, order=100):
    nodes, weights = hermgauss(order)
    points = mean + np.sqrt(2.0 * variance) * nodes
    return np.sum(weights * function(points)) / np.sqrt(np.pi)


def _normal_density(value, mean, variance):
    return np.exp(-0.5 * np.square(value - mean) / variance) / np.sqrt(
        2 * np.pi * variance
    )


def test_jj_objective_is_conservative_without_claiming_exact_evidence_equality():
    pi = np.asarray([0.55, 0.45])
    prior_mean, prior_variance = 0.2, 0.8
    observations = np.asarray([0.1, -0.25])
    observation_variance = np.asarray([0.7, 0.6])
    matrices = np.asarray([[[0.75]], [[-0.35]]])
    offsets = np.asarray([[0.15], [-0.2]])
    process_variance = np.asarray([[[0.4]], [[0.9]]])
    weights = np.asarray([[[1.1]], [[-0.8]]])
    recurrence_offsets = np.asarray([[-0.15], [0.25]])
    prior = _gaussian_canonical(
        np.asarray([prior_mean]), np.asarray([[prior_variance]])
    )
    emissions = [
        _observed_linear_canonical(
            np.asarray([observations[step]]),
            np.asarray([[1.0]]),
            np.asarray([[observation_variance[step]]]),
        )
        for step in range(2)
    ]
    gaussian_eta = np.stack([prior[0] + emissions[0][0], emissions[1][0]])
    gaussian_precision = np.stack(
        [prior[1] + emissions[0][1], emissions[1][1]]
    )
    gaussian_constant = np.asarray(
        [prior[2] + emissions[0][2], emissions[1][2]]
    )
    model = hybrid.StructuredHybridModel(
        np.log(pi),
        np.zeros((2, 2)),
        gaussian_eta,
        gaussian_precision,
        gaussian_constant,
        matrices,
        offsets,
        process_variance,
        weights,
        recurrence_offsets,
    )
    result, _ = _solve(model, iterations=150, tolerance=2e-12)
    mean = float(result.belief.gaussian.mean[0, 0])
    variance = float(result.belief.gaussian.covariance[0, 0, 0])
    exact_log_transition = np.empty((2, 2))
    for source in range(2):
        slope = weights[source, 0, 0]
        offset = recurrence_offsets[source, 0]
        exact_log_transition[source, 0] = _normal_expectation(
            mean,
            variance,
            lambda value, slope=slope, offset=offset: -np.logaddexp(
                0.0, -(slope * value + offset)
            ),
        )
        exact_log_transition[source, 1] = _normal_expectation(
            mean,
            variance,
            lambda value, slope=slope, offset=offset: -np.logaddexp(
                0.0, slope * value + offset
            ),
        )
    pair = np.asarray(result.belief.categorical.pair[0])
    exact_recurrence = np.einsum("ij,ij->", pair, exact_log_transition)
    bounded_recurrence = np.einsum(
        "ij,ij->", pair, np.asarray(result.recurrence.log_potential[0])
    )
    exact_variational = float(result.objective) - (
        exact_recurrence - bounded_recurrence
    )

    nodes, quadrature_weights = hermgauss(120)
    x0 = prior_mean + np.sqrt(2 * prior_variance) * nodes
    likelihood0 = _normal_density(
        observations[0], x0, observation_variance[0]
    )
    evidence_integrand = np.zeros_like(x0)
    for source in range(2):
        logit = weights[source, 0, 0] * x0 + recurrence_offsets[source, 0]
        transition = np.stack(
            [
                np.exp(-np.logaddexp(0.0, -logit)),
                np.exp(-np.logaddexp(0.0, logit)),
            ],
            axis=1,
        )
        future = np.zeros_like(x0)
        for destination in range(2):
            predictive_mean = matrices[destination, 0, 0] * x0 + offsets[
                destination, 0
            ]
            predictive_variance = (
                process_variance[destination, 0, 0]
                + observation_variance[1]
            )
            future += transition[:, destination] * _normal_density(
                observations[1], predictive_mean, predictive_variance
            )
        evidence_integrand += pi[source] * likelihood0 * future
    evidence = np.sum(quadrature_weights * evidence_integrand) / np.sqrt(np.pi)
    negative_log_evidence = -np.log(evidence)
    tolerance = 2e-10
    assert float(result.objective) >= exact_variational - tolerance
    assert exact_variational >= negative_log_evidence - tolerance
    assert float(result.objective) - exact_variational > 1e-5


def test_learning_statistics_match_independent_loop_algebra():
    result, _ = _solve(_small_model())
    categorical = np.asarray(result.belief.categorical.singleton)
    pair = np.asarray(result.belief.categorical.pair)
    gaussian = result.belief.gaussian
    means = np.asarray(gaussian.mean)
    seconds = np.asarray(gaussian.second_moment)
    lag_seconds = np.asarray(gaussian.lag_second_moment)
    card = categorical.shape[1]
    dimension = means.shape[1]
    count = np.zeros(card)
    syy = np.zeros((card, dimension, dimension))
    syx = np.zeros((card, dimension, dimension + 1))
    sxx = np.zeros((card, dimension + 1, dimension + 1))
    for step in range(1, len(means)):
        feature_second = np.block(
            [[seconds[step - 1], means[step - 1, :, None]],
             [means[step - 1, None, :], np.ones((1, 1))]]
        )
        cross = np.concatenate(
            [lag_seconds[step - 1], means[step, :, None]], axis=1
        )
        for regime in range(card):
            responsibility = categorical[step, regime]
            count[regime] += responsibility
            syy[regime] += responsibility * seconds[step]
            syx[regime] += responsibility * cross
            sxx[regime] += responsibility * feature_second
    np.testing.assert_allclose(result.dynamics_statistics.count, count)
    np.testing.assert_allclose(result.dynamics_statistics.syy, syy)
    np.testing.assert_allclose(result.dynamics_statistics.syx, syx)
    np.testing.assert_allclose(result.dynamics_statistics.sxx, sxx)

    exposure = np.cumsum(pair[:, :, ::-1], axis=2)[:, :, ::-1][:, :, :-1]
    signed = pair[:, :, :-1] - 0.5 * exposure
    np.testing.assert_allclose(result.recurrence_statistics.exposure, exposure)
    np.testing.assert_allclose(
        result.recurrence_statistics.signed_success, signed
    )
    linear = np.zeros_like(np.asarray(result.recurrence_statistics.linear))
    quadratic = np.zeros_like(np.asarray(result.recurrence_statistics.quadratic))
    omega = np.asarray(result.recurrence_statistics.omega)
    for step in range(len(pair)):
        feature_mean = np.concatenate([means[step], [1.0]])
        feature_second = np.block(
            [[seconds[step], means[step, :, None]],
             [means[step, None, :], np.ones((1, 1))]]
        )
        linear += signed[step, :, :, None] * feature_mean
        quadratic += (
            exposure[step, :, :, None, None]
            * omega[step, :, :, None, None]
            * feature_second
        )
    np.testing.assert_allclose(result.recurrence_statistics.linear, linear)
    np.testing.assert_allclose(result.recurrence_statistics.quadratic, quadratic)


def test_exact_support_gives_bit_zero_unused_learning_statistics():
    log_initial = np.asarray([0.0, -np.inf, -np.inf])
    log_unary = np.zeros((3, 3))
    transition = np.full((2, 3, 3), -np.inf)
    transition[:, 0, 0] = 0.0
    categorical = hybrid.categorical_coordinate(
        log_initial, log_unary, np.zeros((2, 3)), transition
    )
    assert bool(categorical.proper)
    np.testing.assert_array_equal(categorical.singleton[:, 0], 1.0)
    np.testing.assert_array_equal(categorical.singleton[:, 1:], 0.0)
    gaussian = hybrid.gaussian_coordinate(
        np.zeros((3, 1)),
        np.ones((3, 1, 1)),
        categorical.singleton,
        np.asarray([[[0.5]], [[-0.2]], [[0.8]]]),
        np.zeros((3, 1)),
        np.ones((3, 1, 1)),
        np.zeros((2, 1)),
        np.zeros((2, 1, 1)),
    )
    dynamics = hybrid.dynamics_statistics(categorical.singleton, gaussian)
    recurrence = hybrid.recurrence_statistics(
        categorical.pair,
        gaussian,
        np.ones((3, 2, 1)),
        np.zeros((3, 2)),
    )
    assert bool(dynamics.proper) and bool(recurrence.proper)
    np.testing.assert_array_equal(dynamics.count[1:], 0.0)
    np.testing.assert_array_equal(dynamics.syy[1:], 0.0)
    np.testing.assert_array_equal(recurrence.exposure[:, 1:], 0.0)
    np.testing.assert_array_equal(recurrence.linear[1:], 0.0)
    np.testing.assert_array_equal(recurrence.quadratic[1:], 0.0)


def test_empty_improper_and_nonfinite_inputs_fail_loudly():
    empty = hybrid.categorical_coordinate(
        np.asarray([-np.inf, -np.inf]),
        np.zeros((2, 2)),
        np.zeros((1, 2)),
        np.zeros((1, 2, 2)),
    )
    assert not bool(empty.proper)
    assert np.isnan(empty.singleton).all()
    improper = hybrid.gaussian_coordinate(
        np.zeros((2, 1)),
        np.asarray([[[-1.0]], [[1.0]]]),
        np.full((2, 2), 0.5),
        np.ones((2, 1, 1)),
        np.zeros((2, 1)),
        np.ones((2, 1, 1)),
        np.zeros((1, 1)),
        np.zeros((1, 1, 1)),
    )
    assert not bool(improper.proper)
    assert np.isnan(improper.mean).all()
    nonfinite = hybrid.recurrence_auxiliary(
        np.zeros(1),
        np.zeros((1, 1)),
        np.asarray([[[np.nan]], [[0.0]]]),
        np.zeros((2, 1)),
    )
    assert not bool(nonfinite.proper)
    assert np.isnan(nonfinite.log_potential).all()
    inconsistent_dynamics, dynamics_ok = (
        hybrid.expected_dynamics_log_potential(
            np.zeros((2, 1)),
            np.ones((2, 1, 1)),
            np.asarray([[[2.0]]]),
            np.ones((2, 1, 1)),
            np.zeros((2, 1)),
            np.ones((2, 1, 1)),
        )
    )
    assert not bool(dynamics_ok)
    assert np.isnan(inconsistent_dynamics).all()


def test_float32_zero_pg_limit_and_scaled_spd_chain_are_finite():
    dtype = np.float32
    auxiliary = hybrid.recurrence_auxiliary(
        np.zeros(2, dtype=dtype),
        np.zeros((2, 2), dtype=dtype),
        np.zeros((2, 1, 2), dtype=dtype),
        np.zeros((2, 1), dtype=dtype),
    )
    assert np.asarray(auxiliary.omega).dtype == dtype
    np.testing.assert_array_equal(auxiliary.omega, np.float32(0.25))
    gaussian = jax.jit(hybrid.gaussian_coordinate)(
        np.zeros((4, 2), dtype=dtype),
        np.broadcast_to(np.eye(2, dtype=dtype) * 1e-4, (4, 2, 2)),
        np.full((4, 2), 0.5, dtype=dtype),
        np.asarray(
            [[[0.98, 0.01], [0.0, 0.97]], [[0.9, -0.02], [0.01, 0.95]]],
            dtype=dtype,
        ),
        np.zeros((2, 2), dtype=dtype),
        np.broadcast_to(np.eye(2, dtype=dtype) * 1e3, (2, 2, 2)),
        np.zeros((3, 2), dtype=dtype),
        np.broadcast_to(np.eye(2, dtype=dtype) * 1e-5, (3, 2, 2)),
    )
    assert bool(gaussian.proper)
    assert np.asarray(gaussian.mean).dtype == dtype
    assert np.isfinite(np.asarray(gaussian.covariance)).all()


def test_k8_t100_uses_only_chain_pair_and_block_gaussian_storage():
    time, card, dimension = 100, 8, 2
    dtype = np.float32
    model = hybrid.StructuredHybridModel(
        np.full(card, -np.log(card), dtype=dtype),
        np.zeros((time, card), dtype=dtype),
        np.zeros((time, dimension), dtype=dtype),
        np.broadcast_to(np.eye(dimension, dtype=dtype), (time, dimension, dimension)),
        np.zeros(time, dtype=dtype),
        np.broadcast_to(np.eye(dimension, dtype=dtype) * 0.4, (card, dimension, dimension)),
        np.zeros((card, dimension), dtype=dtype),
        np.broadcast_to(np.eye(dimension, dtype=dtype) * 0.8, (card, dimension, dimension)),
        np.linspace(-0.2, 0.2, card * (card - 1) * dimension, dtype=dtype).reshape(
            card, card - 1, dimension
        ),
        np.zeros((card, card - 1), dtype=dtype),
    )
    initial = hybrid.structured_hybrid_initial(model)
    result = jax.jit(hybrid.structured_hybrid_sweep)(initial, model)
    assert int(result.status) == hybrid.SWEEP_FINITE
    assert result.belief.categorical.singleton.shape == (time, card)
    assert result.belief.categorical.pair.shape == (time - 1, card, card)
    assert result.belief.gaussian.covariance.shape == (
        time,
        dimension,
        dimension,
    )
    assert result.belief.gaussian.lag_covariance.shape == (
        time - 1,
        dimension,
        dimension,
    )
    assert not any(
        getattr(value, "shape", None) == (time, card, card, card)
        for value in jax.tree.leaves(result)
    )


def test_nonfinite_belief_produces_diverged_sweep_status():
    model = _small_model()
    initial = hybrid.structured_hybrid_initial(model)
    broken_gaussian = initial.gaussian._replace(
        mean=initial.gaussian.mean.at[0, 0].set(np.nan)
    )
    result = hybrid.structured_hybrid_sweep(
        initial._replace(gaussian=broken_gaussian), model
    )
    assert int(result.status) == hybrid.SWEEP_DIVERGED
    assert np.isinf(result.residual)

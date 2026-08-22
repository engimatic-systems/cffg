"""Independent coordinate oracles for the D10 structured hybrid family."""

import itertools

import jax
import numpy as np
import pytest

from cffg import structured_hybrid as hybrid
from cffg.experiments.structured_recurrence import (
    fixed_drive_potentials,
    smooth_log_chain,
)


def _logsumexp(value):
    value = np.asarray(value)
    maximum = np.max(value)
    return maximum + np.log(np.exp(value - maximum).sum())


def _enumerate_chain(log_initial, log_unary, dynamics, transition):
    log_initial = np.asarray(log_initial)
    log_unary = np.asarray(log_unary)
    time, card = log_unary.shape
    paths = list(itertools.product(range(card), repeat=time))
    log_weight = np.empty(len(paths), dtype=np.result_type(
        log_initial, log_unary, dynamics, transition
    ))
    for row, path in enumerate(paths):
        value = log_initial[path[0]] + log_unary[0, path[0]]
        for step in range(time - 1):
            value += dynamics[step, path[step + 1]]
            value += transition[step, path[step], path[step + 1]]
            value += log_unary[step + 1, path[step + 1]]
        log_weight[row] = value
    normalizer = _logsumexp(log_weight)
    mass = np.exp(log_weight - normalizer)
    singleton = np.zeros((time, card), dtype=mass.dtype)
    pair = np.zeros((time - 1, card, card), dtype=mass.dtype)
    for path, probability in zip(paths, mass, strict=True):
        for step, state in enumerate(path):
            singleton[step, state] += probability
        for step in range(time - 1):
            pair[step, path[step], path[step + 1]] += probability
    entropy = -np.sum(np.where(mass > 0, mass * np.log(mass), 0.0))
    return singleton, pair, normalizer, entropy


@pytest.mark.parametrize("card", (2, 3))
def test_categorical_coordinate_matches_literal_path_enumeration(card):
    time = 3
    log_initial = np.log(np.linspace(1.0, 2.0, card))
    log_initial -= _logsumexp(log_initial)
    log_unary = np.log(
        np.arange(1, time * card + 1, dtype=np.float64).reshape(time, card)
    )
    dynamics = np.linspace(-0.7, 0.6, (time - 1) * card).reshape(
        time - 1, card
    )
    transition = np.linspace(
        -1.3, 0.5, (time - 1) * card * card
    ).reshape(time - 1, card, card)
    expected = _enumerate_chain(
        log_initial, log_unary, dynamics, transition
    )
    actual = hybrid.categorical_coordinate(
        log_initial, log_unary, dynamics, transition
    )
    assert bool(actual.proper)
    np.testing.assert_allclose(actual.singleton, expected[0], atol=2e-14)
    np.testing.assert_allclose(actual.pair, expected[1], atol=2e-14)
    assert float(actual.log_normalizer) == pytest.approx(expected[2], abs=2e-14)
    assert float(actual.entropy) == pytest.approx(expected[3], abs=2e-14)
    np.testing.assert_allclose(actual.pair.sum(axis=2), actual.singleton[:-1])
    np.testing.assert_allclose(actual.pair.sum(axis=1), actual.singleton[1:])

    if card == 3:
        assert np.all(np.asarray(actual.pair)[..., -1] > 0)
        assert np.all(np.asarray(actual.pair)[:, -1, :] > 0)


def _dense_gaussian_oracle(
    external_eta,
    external_precision,
    singleton,
    matrices,
    offsets,
    process_covariance,
    recurrence_eta,
    recurrence_precision,
):
    time, dimension = external_eta.shape
    precision = np.zeros((time * dimension, time * dimension))
    information = np.zeros(time * dimension)
    for step in range(time):
        index = slice(step * dimension, (step + 1) * dimension)
        precision[index, index] += external_precision[step]
        information[index] += external_eta[step]
    for step in range(1, time):
        for regime, responsibility in enumerate(singleton[step]):
            residual = np.zeros((dimension, time * dimension))
            previous = slice((step - 1) * dimension, step * dimension)
            current = slice(step * dimension, (step + 1) * dimension)
            residual[:, previous] = -matrices[regime]
            residual[:, current] = np.eye(dimension)
            weight = np.linalg.inv(process_covariance[regime])
            precision += responsibility * residual.T @ weight @ residual
            information += (
                responsibility * residual.T @ weight @ offsets[regime]
            )
    for step in range(time - 1):
        index = slice(step * dimension, (step + 1) * dimension)
        precision[index, index] += recurrence_precision[step]
        information[index] += recurrence_eta[step]
    covariance = np.linalg.inv(precision)
    mean = covariance @ information
    return mean.reshape(time, dimension), covariance, precision, information


def test_gaussian_coordinate_matches_dense_k3_d2_noncommuting_oracle():
    time, card, dimension = 3, 3, 2
    singleton = np.asarray(
        [[0.2, 0.5, 0.3], [0.6, 0.1, 0.3], [0.15, 0.7, 0.15]]
    )
    matrices = np.asarray(
        [
            [[0.7, 0.25], [-0.15, 0.55]],
            [[-0.3, 0.8], [0.45, 0.2]],
            [[0.5, -0.4], [0.3, 0.65]],
        ]
    )
    offsets = np.asarray([[0.3, -0.2], [-0.1, 0.4], [0.25, 0.15]])
    process_covariance = np.asarray(
        [
            [[0.7, 0.18], [0.18, 0.45]],
            [[0.35, -0.09], [-0.09, 0.8]],
            [[0.55, 0.21], [0.21, 0.65]],
        ]
    )
    external_precision = np.asarray(
        [
            [[1.4, 0.2], [0.2, 0.9]],
            [[0.6, -0.1], [-0.1, 0.8]],
            [[1.1, 0.25], [0.25, 1.3]],
        ]
    )
    external_eta = np.asarray([[0.3, -0.4], [0.1, 0.25], [-0.2, 0.5]])
    recurrence_precision = np.asarray(
        [[[0.25, 0.08], [0.08, 0.3]], [[0.4, -0.05], [-0.05, 0.2]]]
    )
    recurrence_eta = np.asarray([[0.12, -0.07], [-0.18, 0.09]])
    actual = hybrid.gaussian_coordinate(
        external_eta,
        external_precision,
        singleton,
        matrices,
        offsets,
        process_covariance,
        recurrence_eta,
        recurrence_precision,
    )
    expected_mean, expected_covariance, dense_precision, dense_eta = (
        _dense_gaussian_oracle(
            external_eta,
            external_precision,
            singleton,
            matrices,
            offsets,
            process_covariance,
            recurrence_eta,
            recurrence_precision,
        )
    )
    assert bool(actual.proper)
    np.testing.assert_allclose(actual.mean, expected_mean, atol=3e-14)
    for step in range(time):
        index = slice(step * dimension, (step + 1) * dimension)
        np.testing.assert_allclose(
            actual.covariance[step], expected_covariance[index, index], atol=3e-14
        )
    for step in range(1, time):
        current = slice(step * dimension, (step + 1) * dimension)
        previous = slice((step - 1) * dimension, step * dimension)
        np.testing.assert_allclose(
            actual.lag_covariance[step - 1],
            expected_covariance[current, previous],
            atol=3e-14,
        )
        expected_second = (
            expected_covariance[current, previous]
            + np.outer(expected_mean[step], expected_mean[step - 1])
        )
        np.testing.assert_allclose(
            actual.lag_second_moment[step - 1], expected_second, atol=3e-14
        )
    dense_logdet = np.linalg.slogdet(dense_precision)[1]
    expected_entropy = 0.5 * (
        time * dimension * (1 + np.log(2 * np.pi)) - dense_logdet
    )
    assert float(actual.entropy) == pytest.approx(expected_entropy, abs=3e-14)
    assembled = np.zeros_like(dense_precision)
    for step in range(time):
        index = slice(step * dimension, (step + 1) * dimension)
        assembled[index, index] = actual.precision_diagonal[step]
    for step in range(time - 1):
        left = slice(step * dimension, (step + 1) * dimension)
        right = slice((step + 1) * dimension, (step + 2) * dimension)
        assembled[right, left] = actual.precision_lower[step]
        assembled[left, right] = actual.precision_lower[step].T
    np.testing.assert_allclose(assembled, dense_precision, atol=3e-14)
    np.testing.assert_allclose(actual.information.reshape(-1), dense_eta)

    dynamics, proper = hybrid.expected_dynamics_log_potential(
        actual.mean,
        actual.covariance,
        actual.lag_covariance,
        matrices,
        offsets,
        process_covariance,
    )
    assert bool(proper)
    expected_dynamics = np.empty((time - 1, card))
    dense_mean = expected_mean.reshape(-1)
    for step in range(1, time):
        for regime in range(card):
            residual = np.zeros((dimension, time * dimension))
            previous = slice((step - 1) * dimension, step * dimension)
            current = slice(step * dimension, (step + 1) * dimension)
            residual[:, previous] = -matrices[regime]
            residual[:, current] = np.eye(dimension)
            residual_mean = residual @ dense_mean - offsets[regime]
            residual_second = (
                residual @ expected_covariance @ residual.T
                + np.outer(residual_mean, residual_mean)
            )
            weight = np.linalg.inv(process_covariance[regime])
            expected_dynamics[step - 1, regime] = (
                0.5 * np.linalg.slogdet(weight)[1]
                - 0.5 * dimension * np.log(2 * np.pi)
                - 0.5 * np.trace(weight @ residual_second)
            )
    np.testing.assert_allclose(dynamics, expected_dynamics, atol=5e-14)


@pytest.mark.parametrize("card", (2, 3))
def test_conditioned_drive_is_the_exact_d9_reduction(card):
    log_initial, log_unary, log_transition, drives, weights, offsets = (
        fixed_drive_potentials(card)
    )
    expected = smooth_log_chain(log_initial, log_unary, log_transition)
    actual = hybrid.conditioned_categorical_coordinate(
        log_initial, log_unary, drives, weights, offsets
    )
    assert bool(actual.proper)
    np.testing.assert_allclose(actual.singleton, expected.singleton, atol=2e-14)
    np.testing.assert_allclose(actual.pair, expected.consecutive, atol=2e-14)
    assert float(actual.log_normalizer) == pytest.approx(
        expected.log_normalizer, abs=2e-14
    )


def test_k1_bypasses_d9_and_reduces_to_a_gaussian_chain():
    time, dimension = 4, 2
    model = hybrid.StructuredHybridModel(
        np.zeros(1),
        np.zeros((time, 1)),
        np.asarray([[0.2, -0.1], [0.0, 0.3], [-0.2, 0.1], [0.15, -0.25]]),
        np.broadcast_to(np.eye(dimension), (time, dimension, dimension)).copy(),
        np.zeros(time),
        np.asarray([[[0.7, 0.2], [-0.1, 0.6]]]),
        np.asarray([[0.1, -0.2]]),
        np.asarray([[[0.5, 0.08], [0.08, 0.65]]]),
        np.zeros((1, 0, dimension)),
        np.zeros((1, 0)),
    )
    initial = hybrid.structured_hybrid_initial(model)
    result = hybrid.structured_hybrid_sweep(initial, model)
    assert int(result.status) == hybrid.SWEEP_FINITE
    np.testing.assert_array_equal(result.belief.categorical.singleton, 1.0)
    np.testing.assert_array_equal(result.belief.categorical.pair, 1.0)
    assert result.recurrence.omega.shape == (time - 1, 1, 0)
    assert result.recurrence_statistics.linear.shape == (1, 0, dimension + 1)
    assert np.isfinite(np.asarray(result.objective))


def test_coordinate_shapes_fail_before_algebra():
    with pytest.raises(ValueError, match="categorical coordinate requires"):
        hybrid.categorical_coordinate(
            np.zeros(2), np.zeros((3, 2)), np.zeros((2, 3)), np.zeros((2, 2, 2))
        )
    with pytest.raises(ValueError, match="Gaussian coordinate shapes disagree"):
        hybrid.gaussian_coordinate(
            np.zeros((2, 1)),
            np.zeros((2, 1, 1)),
            np.ones((3, 1)),
            np.ones((1, 1, 1)),
            np.zeros((1, 1)),
            np.ones((1, 1, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 1, 1)),
        )


def test_atomic_coordinates_are_jittable():
    categorical = jax.jit(hybrid.categorical_coordinate)(
        np.log([0.6, 0.4]),
        np.log([[0.8, 0.2], [0.25, 0.75]]),
        np.asarray([[0.2, -0.1]]),
        np.asarray([[[-0.3, -1.4], [-0.7, -0.5]]]),
    )
    assert bool(categorical.proper)
    auxiliary = jax.jit(hybrid.recurrence_auxiliary)(
        np.asarray([0.0]),
        np.asarray([[0.3]]),
        np.asarray([[[1.0]], [[-0.7]]]),
        np.asarray([[0.0], [0.2]]),
    )
    assert bool(auxiliary.proper)

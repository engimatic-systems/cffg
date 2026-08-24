from __future__ import annotations

import itertools
import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.switching_stability import (
    StabilityStatus,
    common_quadratic_certificate,
    evaluate_one_step_drift,
    evaluate_recurrence_drift,
    recurrence_drift_merit,
    recurrence_drift_residuals,
    recurrence_stability_certificate,
    recurrent_transition_probabilities,
)


def _model(regimes: int, dimension: int, dtype: np.dtype) -> tuple[np.ndarray, ...]:
    matrix = np.stack(
        [np.eye(dimension, dtype=dtype) * (0.35 + 0.04 * mode) for mode in range(regimes)]
    )
    offset = np.asarray(
        [[0.03 * (mode + axis + 1) for axis in range(dimension)] for mode in range(regimes)],
        dtype=dtype,
    )
    covariance = np.stack(
        [np.eye(dimension, dtype=dtype) * (0.01 + 0.002 * mode) for mode in range(regimes)]
    )
    metric = np.stack(
        [
            np.diag(np.linspace(1.0, 1.3 + 0.1 * mode, dimension)).astype(dtype)
            for mode in range(regimes)
        ]
    )
    return matrix, offset, covariance, metric


def _numpy_stick(state: np.ndarray, weights: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    regimes = weights.shape[0]
    result = np.empty((regimes, regimes), dtype=state.dtype)
    for source in range(regimes):
        remaining = 1.0
        for destination in range(regimes - 1):
            logit = float(weights[source, destination] @ state + offsets[source, destination])
            accept = 1.0 / (1.0 + np.exp(-logit))
            result[source, destination] = remaining * accept
            remaining *= 1.0 - accept
        result[source, -1] = remaining
    return result


@pytest.mark.parametrize("regimes", [2, 3])
def test_dense_one_step_identity_matches_independent_loop(regimes: int) -> None:
    dtype = np.dtype(np.float64)
    dimension = 2
    matrix, offset, covariance, metric = _model(regimes, dimension, dtype)
    matrix[:, 0, 1] = np.linspace(0.07, 0.13, regimes)
    matrix[:, 1, 0] = np.linspace(-0.04, 0.02, regimes)
    weights = np.arange(regimes * (regimes - 1) * dimension, dtype=dtype)
    weights = weights.reshape(regimes, regimes - 1, dimension) / 9.0 - 0.25
    recurrence_offset = np.linspace(-0.3, 0.25, regimes * (regimes - 1), dtype=dtype)
    recurrence_offset = recurrence_offset.reshape(regimes, regimes - 1)
    states = np.asarray([[0.4, -0.2], [-0.5, 0.7], [0.1, 0.8]], dtype=dtype)
    sources = np.arange(states.shape[0], dtype=np.int32) % regimes
    expected_probability = np.stack(
        [_numpy_stick(state, weights, recurrence_offset) for state in states]
    )
    probability = np.asarray(
        recurrent_transition_probabilities(states, weights, recurrence_offset)
    )
    np.testing.assert_allclose(probability, expected_probability, rtol=1e-13, atol=1e-13)

    selected = probability[np.arange(states.shape[0]), sources]
    evidence = evaluate_one_step_drift(
        states, sources, selected, matrix, offset, covariance, metric
    )
    assert evidence.status is StabilityStatus.SATISFIED
    expected_destination = np.empty((states.shape[0], regimes))
    expected_current = np.empty(states.shape[0])
    for index, (state, source) in enumerate(zip(states, sources, strict=True)):
        expected_current[index] = state @ metric[source] @ state
        for destination in range(regimes):
            mean = matrix[destination] @ state + offset[destination]
            expected_destination[index, destination] = (
                mean @ metric[destination] @ mean
                + np.trace(metric[destination] @ covariance[destination])
            )
    expected_next = np.sum(selected * expected_destination, axis=-1)
    np.testing.assert_allclose(evidence.current_value, expected_current)
    np.testing.assert_allclose(evidence.destination_value, expected_destination)
    np.testing.assert_allclose(evidence.expected_next_value, expected_next)
    np.testing.assert_allclose(evidence.drift, expected_next - expected_current)


def test_common_certificate_is_sound_with_affine_noise() -> None:
    matrix, offset, covariance, _ = _model(3, 2, np.dtype(np.float64))
    common = np.asarray([[1.3, 0.2], [0.2, 0.9]])
    evidence = common_quadratic_certificate(matrix, offset, covariance, common)
    assert evidence.status is StabilityStatus.SATISFIED
    assert evidence.foster_rate is not None
    assert evidence.additive_bound is not None
    rng = np.random.default_rng(82)
    for state, destination in zip(rng.normal(size=(200, 2)), itertools.cycle(range(3))):
        mean = matrix[destination] @ state + offset[destination]
        next_value = mean @ common @ mean + np.trace(common @ covariance[destination])
        current = state @ common @ state
        assert next_value <= evidence.foster_rate * current + evidence.additive_bound + 1e-12


def test_individually_stable_shears_are_switching_unstable() -> None:
    matrix = np.asarray(
        [
            [[0.5, 1.0], [0.0, 0.5]],
            [[0.5, 0.0], [1.0, 0.5]],
        ]
    )
    assert np.allclose(np.abs(np.linalg.eigvals(matrix)), 0.5)
    product_radius = max(abs(np.linalg.eigvals(matrix[1] @ matrix[0])))
    assert product_radius > 1.0
    lift = np.kron(matrix[1], matrix[1]) @ np.kron(matrix[0], matrix[0])
    assert max(abs(np.linalg.eigvals(lift))) > 1.0
    zeros = np.zeros((2, 2))
    covariance = np.zeros((2, 2, 2))
    evidence = common_quadratic_certificate(matrix, zeros, covariance, np.eye(2))
    assert evidence.status is StabilityStatus.VIOLATED


def _recurrence_stabilized_scalar(dtype: np.dtype) -> tuple[np.ndarray, ...]:
    matrix = np.asarray([[[0.5]], [[0.5]], [[1.2]]], dtype=dtype)
    offset = np.zeros((3, 1), dtype=dtype)
    covariance = np.full((3, 1, 1), 0.01, dtype=dtype)
    metric = np.ones((3, 1, 1), dtype=dtype)
    # The last destination needs both rejects.  In either tail, one reject
    # decays exponentially, while the first two destinations are contractive.
    weights = np.tile(np.asarray([[[-2.0], [2.0]]], dtype=dtype), (3, 1, 1))
    recurrence_offset = np.tile(np.asarray([[4.0, 4.0]], dtype=dtype), (3, 1))
    return matrix, offset, covariance, weights, recurrence_offset, metric


def test_recurrence_certificate_retains_tail_stabilization_and_tie_faces() -> None:
    matrix, offset, covariance, weights, recurrence_offset, metric = (
        _recurrence_stabilized_scalar(np.dtype(np.float64))
    )
    common = common_quadratic_certificate(matrix, offset, covariance, metric[0])
    assert common.status is StabilityStatus.VIOLATED
    evidence = recurrence_stability_certificate(
        matrix,
        offset,
        covariance,
        weights,
        recurrence_offset,
        metric,
        foster_rate=0.5,
        tail_radius=4.0,
        compact_splits=8,
    )
    assert evidence.status is StabilityStatus.SATISFIED
    assert evidence.orthant_signs is not None
    np.testing.assert_array_equal(evidence.orthant_signs[:, 0], [-1.0, 1.0])
    assert evidence.exponentially_suppressed is not None
    assert np.all(evidence.exponentially_suppressed[:, :, 2])
    assert evidence.additive_bound is not None and evidence.additive_bound > 0.0
    assert evidence.asymptotic_lyapunov_envelope > 2.0 * evidence.additive_bound
    assert evidence.compact_box_count == 8
    assert evidence.tail_additive_bound is not None
    assert evidence.compact_additive_bound is not None
    assert evidence.decay_rate is not None
    assert evidence.decay_intercept is not None
    assert np.all(evidence.mode_contraction[0, 0, :2] < 0.5)
    assert evidence.mode_contraction[0, 0, 2] == 0.5
    for cone, sign in enumerate((-1.0, 1.0)):
        for magnitude in np.linspace(4.0, 100.0, 193):
            state = np.asarray([sign * magnitude])
            probability = _numpy_stick(state, weights, recurrence_offset)
            for source in range(3):
                destination = np.asarray(
                    [
                        (matrix[mode] @ state + offset[mode])
                        @ metric[mode]
                        @ (matrix[mode] @ state + offset[mode])
                        + np.trace(metric[mode] @ covariance[mode])
                        for mode in range(3)
                    ]
                )
                residual = probability[source] @ destination - 0.5 * state @ metric[source] @ state
                assert residual <= evidence.tail_additive_bound[cone, source] + 1e-12
                rate = evidence.decay_rate[cone, source, 2]
                intercept = evidence.decay_intercept[cone, source, 2]
                assert probability[source, 2] <= np.exp(intercept - rate * magnitude) + 1e-12
    for value in np.linspace(-4.0, 4.0, 257):
        state = np.asarray([value])
        probability = _numpy_stick(state, weights, recurrence_offset)
        destination = np.asarray(
            [
                (matrix[mode] @ state + offset[mode])
                @ metric[mode]
                @ (matrix[mode] @ state + offset[mode])
                + np.trace(metric[mode] @ covariance[mode])
                for mode in range(3)
            ]
        )
        for source in range(3):
            residual = probability[source] @ destination - 0.5 * state @ metric[source] @ state
            assert residual <= evidence.compact_additive_bound + 1e-12

    # Removing one tail's decay exposes the unstable destination on that
    # orthant, including its coordinate-boundary face.
    no_positive_tail_decay = weights.copy()
    no_positive_tail_decay[:, 1, 0] = -2.0
    violated = recurrence_stability_certificate(
        matrix,
        offset,
        covariance,
        no_positive_tail_decay,
        recurrence_offset,
        metric,
        foster_rate=0.5,
        tail_radius=4.0,
        compact_splits=8,
    )
    assert violated.status is StabilityStatus.VIOLATED


def test_finite_probe_score_is_explicitly_descriptive_and_differentiable() -> None:
    matrix, offset, covariance, weights, recurrence_offset, metric = (
        _recurrence_stabilized_scalar(np.dtype(np.float64))
    )
    states = jnp.asarray([[-5.0], [-1.0], [0.0], [1.0], [5.0]])
    sources = jnp.asarray([0, 1, 2, 0, 1], dtype=jnp.int32)
    certificate = recurrence_stability_certificate(
        matrix,
        offset,
        covariance,
        weights,
        recurrence_offset,
        metric,
        foster_rate=0.8,
        tail_radius=4.0,
        compact_splits=8,
    )
    evidence = evaluate_recurrence_drift(
        states,
        sources,
        matrix,
        offset,
        covariance,
        weights,
        recurrence_offset,
        metric,
        foster_rate=0.8,
        additive_bound=certificate.additive_bound,
    )
    assert evidence.scope == "descriptive_finite_probe"
    assert evidence.status is StabilityStatus.SATISFIED

    def merit(value: jax.Array) -> jax.Array:
        changed = jnp.asarray(weights).at[0, 0, 0].set(value)
        return recurrence_drift_merit(
            jnp.asarray([[-1.0], [-0.5], [0.5], [1.0]]),
            jnp.zeros(4, dtype=jnp.int32),
            matrix,
            offset,
            covariance,
            changed,
            recurrence_offset,
            metric,
            0.2,
            0.0,
        )

    derivative = jax.grad(merit)(jnp.asarray(-2.0))
    assert np.isfinite(derivative)
    assert abs(float(derivative)) > 1e-10


def test_literal_zero_support_and_empty_support_are_distinct() -> None:
    matrix, offset, covariance, metric = _model(2, 1, np.dtype(np.float64))
    states = np.asarray([[1.0], [-1.0]])
    sources = np.asarray([0, 1], dtype=np.int32)
    literal = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    exact = evaluate_one_step_drift(
        states, sources, literal, matrix, offset, covariance, metric
    )
    assert exact.status is StabilityStatus.SATISFIED
    assert exact.expected_next_value is not None
    empty = evaluate_one_step_drift(
        states, sources, np.zeros_like(literal), matrix, offset, covariance, metric
    )
    assert empty.status is StabilityStatus.INFEASIBLE


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("nan", StabilityStatus.DIVERGED),
        ("improper_covariance", StabilityStatus.INFEASIBLE),
        ("improper_metric", StabilityStatus.INFEASIBLE),
        ("float16", StabilityStatus.INFEASIBLE),
    ],
)
def test_hostile_model_statuses(mutation: str, expected: StabilityStatus) -> None:
    matrix, offset, covariance, metric = _model(2, 2, np.dtype(np.float64))
    if mutation == "nan":
        matrix[0, 0, 0] = np.nan
    elif mutation == "improper_covariance":
        covariance[0, 0, 0] = -1.0
    elif mutation == "improper_metric":
        metric[0, 0, 0] = -1.0
    else:
        matrix = matrix.astype(np.float16)
        offset = offset.astype(np.float16)
        covariance = covariance.astype(np.float16)
        metric = metric.astype(np.float16)
    evidence = common_quadratic_certificate(matrix, offset, covariance, metric[0])
    assert evidence.status is expected


def test_nonfinite_recurrence_and_settings_diverge() -> None:
    model = list(_recurrence_stabilized_scalar(np.dtype(np.float64)))
    model[3][0, 0, 0] = np.inf
    evidence = recurrence_stability_certificate(
        *model, foster_rate=0.5, tail_radius=4.0, compact_splits=2
    )
    assert evidence.status is StabilityStatus.DIVERGED
    model = list(_recurrence_stabilized_scalar(np.dtype(np.float64)))
    evidence = recurrence_stability_certificate(
        *model, foster_rate=np.nan, tail_radius=4.0, compact_splits=2
    )
    assert evidence.status is StabilityStatus.DIVERGED


def test_float32_near_unit_boundary_uses_dtype_guard() -> None:
    with jax.enable_x64(False):
        epsilon = np.finfo(np.float32).eps
        offset = np.zeros((1, 1), dtype=np.float32)
        covariance = np.zeros((1, 1, 1), dtype=np.float32)
        metric = np.ones((1, 1), dtype=np.float32)
        stable = np.asarray([[[1.0 - 1024.0 * epsilon]]], dtype=np.float32)
        unstable = np.asarray([[[1.0 + 1024.0 * epsilon]]], dtype=np.float32)
        stable_result = common_quadratic_certificate(stable, offset, covariance, metric)
        unstable_result = common_quadratic_certificate(
            unstable, offset, covariance, metric
        )
        assert stable_result.status is StabilityStatus.SATISFIED
        assert unstable_result.status is StabilityStatus.VIOLATED


def test_float32_certificate_bounds_exact_affine_value_outward() -> None:
    dtype = np.dtype(np.float32)
    matrix = np.zeros((1, 1, 1), dtype=dtype)
    offset = np.asarray([[0.94792676]], dtype=dtype)
    covariance = np.zeros((1, 1, 1), dtype=dtype)
    weights = np.zeros((1, 0, 1), dtype=dtype)
    recurrence_offset = np.zeros((1, 0), dtype=dtype)
    metric = np.ones((1, 1, 1), dtype=dtype)
    evidence = recurrence_stability_certificate(
        matrix,
        offset,
        covariance,
        weights,
        recurrence_offset,
        metric,
        foster_rate=0.5,
        tail_radius=1.0,
        compact_splits=1,
    )
    exact = float(np.float64(offset[0, 0]) ** 2)
    assert evidence.status is StabilityStatus.SATISFIED
    assert evidence.additive_bound is not None
    assert evidence.additive_bound > exact
    common = common_quadratic_certificate(
        matrix, offset, covariance, metric[0]
    )
    assert common.status is StabilityStatus.SATISFIED
    assert common.additive_bound is not None
    assert common.additive_bound > exact


def test_underflowed_finite_stick_probability_keeps_positive_bound() -> None:
    dtype = np.dtype(np.float64)
    matrix = np.zeros((3, 1, 1), dtype=dtype)
    offset = np.asarray([[0.0], [0.0], [1.0]], dtype=dtype)
    covariance = np.zeros((3, 1, 1), dtype=dtype)
    weights = np.zeros((3, 2, 1), dtype=dtype)
    recurrence_offset = np.full((3, 2), 500.0, dtype=dtype)
    metric = np.ones((3, 1, 1), dtype=dtype)
    evidence = recurrence_stability_certificate(
        matrix,
        offset,
        covariance,
        weights,
        recurrence_offset,
        metric,
        foster_rate=0.5,
        tail_radius=1.0,
        compact_splits=1,
    )
    assert evidence.status is StabilityStatus.SATISFIED
    assert evidence.compact_additive_bound is not None
    assert evidence.compact_additive_bound > 0.0


def test_ill_conditioned_metric_requires_verified_matrix_inequality() -> None:
    source_metric = np.asarray(
        [
            [0.6442435453630332, 0.47874189248554033],
            [0.47874189248554033, 0.3557564546382765],
        ]
    )
    destination_metric = np.asarray(
        [
            [0.9948577149958107, -0.07152511383506864],
            [-0.07152511383506864, 0.005142285005263679],
        ]
    )
    matrix = np.asarray(
        [
            np.zeros((2, 2)),
            [
                [1.2433191057540423e-6, -5.5546725013935428e-8],
                [-2.6004415162469168e-7, 2.4241433323312407e-7],
            ],
        ]
    )
    offset = np.zeros((2, 2))
    covariance = np.zeros((2, 2, 2))
    weights = np.zeros((2, 1, 2))
    recurrence_offset = np.full((2, 1), -20.0)
    metric = np.stack((source_metric, destination_metric))
    foster_rate = 0.4999928568145363
    evidence = recurrence_stability_certificate(
        matrix,
        offset,
        covariance,
        weights,
        recurrence_offset,
        metric,
        foster_rate=foster_rate,
        tail_radius=1.0,
        compact_splits=1,
    )
    assert evidence.status is StabilityStatus.VIOLATED
    direction = np.asarray([0.74310701, -1.0]) * 1e5
    probability = _numpy_stick(direction, weights, recurrence_offset)[0]
    next_value = sum(
        probability[destination]
        * (matrix[destination] @ direction)
        @ metric[destination]
        @ (matrix[destination] @ direction)
        for destination in range(2)
    )
    current = direction @ source_metric @ direction
    assert next_value > foster_rate * current


def test_descriptive_residual_uses_expected_minus_rate_times_value() -> None:
    dtype = np.dtype(np.float64)
    matrix = np.asarray([[[0.8]]], dtype=dtype)
    offset = np.zeros((1, 1), dtype=dtype)
    covariance = np.zeros((1, 1, 1), dtype=dtype)
    weights = np.zeros((1, 0, 1), dtype=dtype)
    recurrence_offset = np.zeros((1, 0), dtype=dtype)
    metric = np.ones((1, 1, 1), dtype=dtype)
    evidence = evaluate_recurrence_drift(
        np.asarray([[10.0]], dtype=dtype),
        np.asarray([0], dtype=np.int32),
        matrix,
        offset,
        covariance,
        weights,
        recurrence_offset,
        metric,
        foster_rate=0.2,
        additive_bound=0.0,
    )
    assert evidence.status is StabilityStatus.VIOLATED
    np.testing.assert_allclose(evidence.residual, [44.0])
    pure_residual = recurrence_drift_residuals(
        np.asarray([[10.0]], dtype=dtype),
        np.asarray([0], dtype=np.int32),
        matrix,
        offset,
        covariance,
        weights,
        recurrence_offset,
        metric,
        0.2,
        0.0,
    )[-1]
    np.testing.assert_allclose(pure_residual, [44.0])


def test_host_evidence_preserves_float64_when_jax_x64_is_disabled() -> None:
    script = textwrap.dedent(
        """
        import numpy as np
        from cffg.switching_stability import evaluate_one_step_drift, evaluate_recurrence_drift

        dtype = np.dtype(np.float64)
        matrix = np.asarray([[[1.0000000001]]], dtype=dtype)
        offset = np.asarray([[0.0000000003]], dtype=dtype)
        covariance = np.asarray([[[0.0000000007]]], dtype=dtype)
        metric = np.ones((1, 1, 1), dtype=dtype)
        state = np.asarray([[123456789.125]], dtype=dtype)
        source = np.asarray([0], dtype=np.int32)
        probability = np.ones((1, 1), dtype=dtype)
        result = evaluate_one_step_drift(
            state, source, probability, matrix, offset, covariance, metric
        )
        mean = matrix[0] @ state[0] + offset[0]
        expected = mean @ metric[0] @ mean + np.trace(metric[0] @ covariance[0])
        assert result.expected_next_value.dtype == np.float64
        assert result.expected_next_value[0] == expected
        diagnostic = evaluate_recurrence_drift(
            state,
            source,
            matrix,
            offset,
            covariance,
            np.zeros((1, 0, 1), dtype=dtype),
            np.zeros((1, 0), dtype=dtype),
            metric,
            foster_rate=0.5,
            additive_bound=0.0,
        )
        assert diagnostic.expected_next_value.dtype == np.float64
        assert diagnostic.expected_next_value[0] == expected
        """
    )
    environment = dict(os.environ)
    environment["JAX_ENABLE_X64"] = "0"
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_k8_hundred_box_certificate_and_jit_are_polynomial() -> None:
    regimes = 8
    dimension = 2
    matrix, offset, covariance, metric = _model(regimes, dimension, np.dtype(np.float32))
    weights = np.zeros((regimes, regimes - 1, dimension), dtype=np.float32)
    recurrence_offset = np.zeros((regimes, regimes - 1), dtype=np.float32)
    evidence = recurrence_stability_certificate(
        matrix,
        offset,
        covariance,
        weights,
        recurrence_offset,
        metric,
        foster_rate=0.8,
        tail_radius=5.0,
        compact_splits=10,
    )
    assert evidence.status is StabilityStatus.SATISFIED
    assert evidence.compact_box_count == 100
    states = jnp.linspace(-2.0, 2.0, 200 * dimension, dtype=jnp.float32).reshape(200, dimension)
    sources = jnp.arange(200, dtype=jnp.int32) % regimes
    merit = jax.jit(recurrence_drift_merit)(
        states,
        sources,
        matrix,
        offset,
        covariance,
        weights,
        recurrence_offset,
        metric,
        0.8,
        evidence.additive_bound,
    )
    probability = recurrent_transition_probabilities(states, weights, recurrence_offset)
    assert probability.shape == (200, regimes, regimes)
    assert np.isfinite(merit)

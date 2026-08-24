"""Truth-free quadratic-drift authorities for recurrent switching dynamics.

Rows of a recurrence block are source regimes.  Columns of every returned
transition table are destination regimes, and destination ``j`` selects
``(A[j], b[j], Q[j])``. Finite-probe evidence is not a global certificate;
the common-quadratic and recurrence-aware authorities own distinct sufficient
global claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


class StabilityStatus(str, Enum):
    """Status of a host-side stability evaluation."""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    INFEASIBLE = "infeasible"
    DIVERGED = "diverged"


@dataclass(frozen=True)
class OneStepDriftEvidence:
    """Exact one-step quadratic-drift terms, or a loud invalid status."""

    status: StabilityStatus
    reason: str
    current_value: np.ndarray | None = None
    destination_value: np.ndarray | None = None
    expected_next_value: np.ndarray | None = None
    drift: np.ndarray | None = None


@dataclass(frozen=True)
class CommonQuadraticEvidence:
    """Arbitrary-switching common-quadratic certificate evidence."""

    status: StabilityStatus
    reason: str
    mode_contraction: np.ndarray | None = None
    maximum_contraction: float | None = None
    proof_tolerance: float | None = None
    foster_rate: float | None = None
    additive_bound: float | None = None
    asymptotic_lyapunov_envelope: float | None = None


@dataclass(frozen=True)
class RecurrenceDriftEvidence:
    """Exact drift and merit on a declared finite probe region."""

    status: StabilityStatus
    reason: str
    scope: str
    transition_probability: np.ndarray | None = None
    current_value: np.ndarray | None = None
    expected_next_value: np.ndarray | None = None
    drift: np.ndarray | None = None
    additive_bound: float | None = None
    residual: np.ndarray | None = None
    tolerance: np.ndarray | None = None
    merit: float | None = None


@dataclass(frozen=True)
class RecurrenceCertificateEvidence:
    """Rigorous orthant-tail plus compact-cover Foster evidence."""

    status: StabilityStatus
    reason: str
    foster_rate: float | None = None
    tail_radius: float | None = None
    orthant_signs: np.ndarray | None = None
    mode_contraction: np.ndarray | None = None
    exponentially_suppressed: np.ndarray | None = None
    decay_rate: np.ndarray | None = None
    decay_intercept: np.ndarray | None = None
    tail_additive_bound: np.ndarray | None = None
    compact_additive_bound: float | None = None
    additive_bound: float | None = None
    asymptotic_lyapunov_envelope: float | None = None
    compact_box_count: int | None = None


def _stick_probabilities(
    states: jax.Array,
    recurrence_weights: jax.Array,
    recurrence_offsets: jax.Array,
) -> jax.Array:
    logits = jnp.einsum("...d,ild->...il", states, recurrence_weights)
    logits = logits + recurrence_offsets
    accept = jax.nn.sigmoid(logits)
    reject = 1.0 - accept
    prefix = jnp.concatenate(
        (
            jnp.ones((*accept.shape[:-1], 1), dtype=accept.dtype),
            jnp.cumprod(reject, axis=-1),
        ),
        axis=-1,
    )
    probability = jnp.concatenate(
        (accept * prefix[..., :-1], prefix[..., -1:]), axis=-1
    )
    normalizer = jnp.sum(probability, axis=-1, keepdims=True)
    return probability / normalizer


def recurrent_transition_probabilities(
    states: jax.Array,
    recurrence_weights: jax.Array,
    recurrence_offsets: jax.Array,
) -> jax.Array:
    """Return ``p(destination | source, state)`` with source rows."""
    return _stick_probabilities(
        jnp.asarray(states),
        jnp.asarray(recurrence_weights),
        jnp.asarray(recurrence_offsets),
    )


def one_step_quadratic_drift(
    states: jax.Array,
    source_regime: jax.Array,
    transition_probability: jax.Array,
    dynamics_matrix: jax.Array,
    dynamics_offset: jax.Array,
    dynamics_covariance: jax.Array,
    lyapunov_matrix: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return current, destination, expected-next, and exact drift arrays."""
    states = jnp.asarray(states)
    source_regime = jnp.asarray(source_regime)
    transition_probability = jnp.asarray(transition_probability)
    dynamics_matrix = jnp.asarray(dynamics_matrix)
    dynamics_offset = jnp.asarray(dynamics_offset)
    dynamics_covariance = jnp.asarray(dynamics_covariance)
    lyapunov_matrix = jnp.asarray(lyapunov_matrix)
    destination_mean = jnp.einsum("kde,ne->nkd", dynamics_matrix, states)
    destination_mean = destination_mean + dynamics_offset[None, :, :]
    destination_value = jnp.einsum(
        "nkd,kde,nke->nk",
        destination_mean,
        lyapunov_matrix,
        destination_mean,
    ) + jnp.einsum(
        "kde,ked->k", lyapunov_matrix, dynamics_covariance
    )[None, :]
    source_matrix = lyapunov_matrix[source_regime]
    current_value = jnp.einsum(
        "nd,nde,ne->n", states, source_matrix, states
    )
    expected_next_value = jnp.sum(
        transition_probability * destination_value, axis=-1
    )
    drift = expected_next_value - current_value
    return current_value, destination_value, expected_next_value, drift


def _selected_recurrence_probability(
    states: jax.Array,
    source_regime: jax.Array,
    recurrence_weights: jax.Array,
    recurrence_offsets: jax.Array,
) -> jax.Array:
    all_probability = recurrent_transition_probabilities(
        states, recurrence_weights, recurrence_offsets
    )
    return all_probability[
        jnp.arange(states.shape[0], dtype=source_regime.dtype), source_regime
    ]


def recurrence_drift_residuals(
    states: jax.Array,
    source_regime: jax.Array,
    dynamics_matrix: jax.Array,
    dynamics_offset: jax.Array,
    dynamics_covariance: jax.Array,
    recurrence_weights: jax.Array,
    recurrence_offsets: jax.Array,
    lyapunov_matrix: jax.Array,
    foster_rate: float | jax.Array,
    additive_bound: float | jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return exact finite-probe drift residuals for declared Foster settings."""
    states = jnp.asarray(states)
    source_regime = jnp.asarray(source_regime)
    probability = _selected_recurrence_probability(
        states, source_regime, recurrence_weights, recurrence_offsets
    )
    current, _, expected, drift = one_step_quadratic_drift(
        states,
        source_regime,
        probability,
        dynamics_matrix,
        dynamics_offset,
        dynamics_covariance,
        lyapunov_matrix,
    )
    residual = drift + (
        1.0 - jnp.asarray(foster_rate, states.dtype)
    ) * current
    residual = residual - jnp.asarray(additive_bound, states.dtype)
    return probability, current, expected, drift, residual


def recurrence_drift_merit(
    states: jax.Array,
    source_regime: jax.Array,
    dynamics_matrix: jax.Array,
    dynamics_offset: jax.Array,
    dynamics_covariance: jax.Array,
    recurrence_weights: jax.Array,
    recurrence_offsets: jax.Array,
    lyapunov_matrix: jax.Array,
    foster_rate: float | jax.Array,
    additive_bound: float | jax.Array,
) -> jax.Array:
    """Return the scale-normalized squared positive finite-probe drift."""
    _, current, expected, _, residual = recurrence_drift_residuals(
        states,
        source_regime,
        dynamics_matrix,
        dynamics_offset,
        dynamics_covariance,
        recurrence_weights,
        recurrence_offsets,
        lyapunov_matrix,
        foster_rate,
        additive_bound,
    )
    scale = (
        1.0
        + jnp.abs(current)
        + jnp.abs(expected)
        + jnp.abs(jnp.asarray(additive_bound, states.dtype))
    )
    return jnp.mean(jnp.square(jnp.maximum(residual, 0.0) / scale))


def _dtype_tolerance(dtype: np.dtype[Any], *values: Any, count: int) -> float:
    dtype = np.dtype(dtype)
    if (
        not np.issubdtype(dtype, np.floating)
        or dtype.itemsize < np.dtype(np.float32).itemsize
    ):
        raise ValueError("stability evidence requires float32 or wider dtype")
    flattened = [
        np.asarray(value, dtype=dtype).reshape(-1) for value in values
    ]
    scale = max(
        (1.0, *(float(np.max(np.abs(value), initial=0.0)) for value in flattened))
    )
    tolerance = 64.0 * max(1, count) * float(np.finfo(dtype).eps) * scale
    if not np.isfinite(tolerance):
        raise ValueError("stability tolerance is not finite")
    return tolerance


def _proof_guard(
    dtype: np.dtype[Any],
    *,
    count: int,
    scale: Any,
) -> np.ndarray:
    declared = np.dtype(dtype)
    scale_array = np.asarray(scale, dtype=np.float64)
    return (
        64.0
        * max(1, count)
        * float(np.finfo(declared).eps)
        * np.maximum(1.0, np.abs(scale_array))
    )


def _outward_upper(
    value: Any,
    dtype: np.dtype[Any],
    *,
    count: int,
    scale: Any | None = None,
) -> np.ndarray:
    upper = np.asarray(value, dtype=np.float64)
    local_scale = np.abs(upper) if scale is None else scale
    guarded = upper + _proof_guard(dtype, count=count, scale=local_scale)
    return np.nextafter(guarded, np.inf)


def _outward_positive_difference_lower(
    left: float,
    right: float,
    dtype: np.dtype[Any],
) -> float:
    guard = float(
        _proof_guard(dtype, count=2, scale=abs(left) + abs(right))
    )
    return float(np.nextafter(left - right - guard, -np.inf))


def _covariance_upper(covariance: np.ndarray) -> np.ndarray:
    dimension = covariance.shape[-1]
    upper = covariance.astype(np.float64).copy()
    for index in range(covariance.shape[0]):
        guard = float(
            _proof_guard(
                covariance.dtype,
                count=dimension**3,
                scale=np.sum(np.abs(upper[index])),
            )
        )
        upper[index] = upper[index] + guard * np.eye(dimension)
    return upper


def _invalid_status(arrays: tuple[np.ndarray, ...]) -> tuple[StabilityStatus, str] | None:
    if any(not np.all(np.isfinite(value)) for value in arrays):
        return StabilityStatus.DIVERGED, "parameters or evidence are non-finite"
    return None


def _matrix_tolerance(matrix: np.ndarray, *, count: int) -> float:
    return _dtype_tolerance(matrix.dtype, matrix, count=count)


def _validate_model(
    dynamics_matrix: Any,
    dynamics_offset: Any,
    dynamics_covariance: Any,
    lyapunov_matrix: Any,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    tuple[StabilityStatus, str] | None,
]:
    matrix = np.asarray(dynamics_matrix)
    offset = np.asarray(dynamics_offset)
    covariance = np.asarray(dynamics_covariance)
    metric = np.asarray(lyapunov_matrix)
    if matrix.ndim != 3 or matrix.shape[0] < 1 or matrix.shape[1] != matrix.shape[2]:
        return None, (StabilityStatus.INFEASIBLE, "dynamics shape is infeasible")
    regimes, dimension, _ = matrix.shape
    if (
        offset.shape != (regimes, dimension)
        or covariance.shape != (regimes, dimension, dimension)
        or metric.shape != (regimes, dimension, dimension)
        or len({matrix.dtype, offset.dtype, covariance.dtype, metric.dtype}) != 1
    ):
        return None, (StabilityStatus.INFEASIBLE, "model shapes or dtypes differ")
    invalid = _invalid_status((matrix, offset, covariance, metric))
    if invalid is not None:
        return None, invalid
    covariance_symmetric = 0.5 * (covariance + covariance.swapaxes(-1, -2))
    metric_symmetric = 0.5 * (metric + metric.swapaxes(-1, -2))
    try:
        for index in range(regimes):
            covariance_tolerance = _matrix_tolerance(
                covariance[index], count=dimension**3
            )
            metric_tolerance = _matrix_tolerance(
                metric[index], count=dimension**3
            )
            if (
                np.max(np.abs(covariance[index] - covariance_symmetric[index]))
                > covariance_tolerance
                or np.max(np.abs(metric[index] - metric_symmetric[index]))
                > metric_tolerance
                or np.min(np.linalg.eigvalsh(covariance_symmetric[index]))
                < -covariance_tolerance
                or np.min(np.linalg.eigvalsh(metric_symmetric[index]))
                <= metric_tolerance
            ):
                return None, (
                    StabilityStatus.INFEASIBLE,
                    "covariance or Lyapunov matrix is improper",
                )
    except ValueError as error:
        return None, (StabilityStatus.INFEASIBLE, str(error))
    except np.linalg.LinAlgError as error:
        return None, (
            StabilityStatus.DIVERGED,
            "matrix validation failed: " + str(error),
        )
    arrays = (matrix, offset, covariance_symmetric, metric_symmetric)
    return arrays, None


def evaluate_one_step_drift(
    states: Any,
    source_regime: Any,
    transition_probability: Any,
    dynamics_matrix: Any,
    dynamics_offset: Any,
    dynamics_covariance: Any,
    lyapunov_matrix: Any,
) -> OneStepDriftEvidence:
    """Validate and evaluate the exact one-step drift identity."""
    model, error = _validate_model(
        dynamics_matrix,
        dynamics_offset,
        dynamics_covariance,
        lyapunov_matrix,
    )
    if error is not None:
        return OneStepDriftEvidence(*error)
    matrix, offset, covariance, metric = model
    state = np.asarray(states)
    source = np.asarray(source_regime)
    probability = np.asarray(transition_probability)
    regimes, dimension, _ = matrix.shape
    if (
        state.ndim != 2
        or not state.shape[0]
        or state.shape[1] != dimension
        or source.shape != (state.shape[0],)
        or not np.issubdtype(source.dtype, np.integer)
        or probability.shape != (state.shape[0], regimes)
        or state.dtype != matrix.dtype
        or probability.dtype != matrix.dtype
        or np.any(source < 0)
        or np.any(source >= regimes)
    ):
        return OneStepDriftEvidence(
            StabilityStatus.INFEASIBLE, "probe shapes, regimes, or dtypes differ"
        )
    invalid = _invalid_status((state, probability))
    if invalid is not None:
        return OneStepDriftEvidence(*invalid)
    try:
        tolerance = _dtype_tolerance(
            matrix.dtype, probability, count=regimes
        )
    except ValueError as exception:
        return OneStepDriftEvidence(StabilityStatus.INFEASIBLE, str(exception))
    row_sum = np.sum(probability, axis=-1)
    if np.any(probability < 0) or np.any(np.abs(row_sum - 1.0) > tolerance):
        return OneStepDriftEvidence(
            StabilityStatus.INFEASIBLE,
            "transition support is empty or probability is improper",
        )
    destination_mean = np.einsum("kde,ne->nkd", matrix, state)
    destination_mean = destination_mean + offset[None, :, :]
    destination_value = np.einsum(
        "nkd,kde,nke->nk", destination_mean, metric, destination_mean
    ) + np.einsum("kde,ked->k", metric, covariance)[None, :]
    current_value = np.einsum(
        "nd,nde,ne->n", state, metric[source], state
    )
    expected_next_value = np.sum(probability * destination_value, axis=-1)
    values = (
        current_value,
        destination_value,
        expected_next_value,
        expected_next_value - current_value,
    )
    invalid = _invalid_status(values)
    if invalid is not None:
        return OneStepDriftEvidence(*invalid)
    return OneStepDriftEvidence(
        StabilityStatus.SATISFIED,
        "exact one-step identity evaluated",
        *values,
    )


def common_quadratic_certificate(
    dynamics_matrix: Any,
    dynamics_offset: Any,
    dynamics_covariance: Any,
    lyapunov_matrix: Any,
) -> CommonQuadraticEvidence:
    """Certify a common quadratic Foster bound under arbitrary switching."""
    matrix = np.asarray(dynamics_matrix)
    common = np.asarray(lyapunov_matrix)
    if matrix.ndim != 3 or common.shape != matrix.shape[1:]:
        return CommonQuadraticEvidence(
            StabilityStatus.INFEASIBLE, "common Lyapunov shape differs"
        )
    repeated = np.broadcast_to(common, matrix.shape).copy()
    model, error = _validate_model(
        matrix, dynamics_offset, dynamics_covariance, repeated
    )
    if error is not None:
        return CommonQuadraticEvidence(*error)
    matrix, offset, covariance, metric = model
    covariance_upper = _covariance_upper(covariance)
    common = metric[0]
    try:
        contraction = _mode_contractions(matrix, metric)[0]
    except np.linalg.LinAlgError as exception:
        return CommonQuadraticEvidence(
            StabilityStatus.DIVERGED,
            "common contraction decomposition failed: " + str(exception),
        )
    maximum = float(np.max(contraction))
    try:
        tolerance = _dtype_tolerance(
            matrix.dtype, contraction, count=matrix.shape[-1] ** 3
        )
    except ValueError as exception:
        return CommonQuadraticEvidence(StabilityStatus.INFEASIBLE, str(exception))
    if maximum + tolerance >= 1.0:
        return CommonQuadraticEvidence(
            StabilityStatus.VIOLATED,
            "at least one mode lacks strict common contraction",
            contraction,
            maximum,
            tolerance,
        )
    if maximum == 0.0:
        foster_rate = 0.0
        offset_multiplier = 1.0
    else:
        foster_rate = float(
            _outward_upper(
                np.sqrt(maximum),
                matrix.dtype,
                count=matrix.shape[-1],
                scale=maximum,
            )
        )
        if foster_rate >= 1.0:
            return CommonQuadraticEvidence(
                StabilityStatus.VIOLATED,
                "outward-rounded common contraction has no strict rate below one",
                contraction,
                maximum,
                tolerance,
            )
        contraction_gap = _outward_positive_difference_lower(
            foster_rate, maximum, matrix.dtype
        )
        if contraction_gap <= 0.0:
            return CommonQuadraticEvidence(
                StabilityStatus.VIOLATED,
                "outward-rounded common contraction has no positive Young gap",
                contraction,
                maximum,
                tolerance,
            )
        offset_multiplier = float(
            _outward_upper(
                foster_rate / contraction_gap,
                matrix.dtype,
                count=2,
                scale=foster_rate / contraction_gap,
            )
        )
    proof_offset = offset.astype(np.float64)
    proof_common = common.astype(np.float64)
    proof_covariance = covariance_upper
    offset_terms = np.einsum(
        "kd,de,ke->k", proof_offset, proof_common, proof_offset
    )
    noise_terms = np.einsum("de,ked->k", proof_common, proof_covariance)
    additive_terms = offset_multiplier * offset_terms + noise_terms
    additive_scale = (
        abs(offset_multiplier)
        * np.einsum(
            "kd,de,ke->k",
            np.abs(proof_offset),
            np.abs(proof_common),
            np.abs(proof_offset),
        )
        + np.einsum("de,ked->k", np.abs(proof_common), np.abs(proof_covariance))
    )
    additive = float(
        np.max(
            _outward_upper(
                additive_terms,
                matrix.dtype,
                count=matrix.shape[-1] ** 3,
                scale=additive_scale,
            )
        )
    )
    envelope_denominator = _outward_positive_difference_lower(
        1.0, foster_rate, matrix.dtype
    )
    if envelope_denominator <= 0.0:
        return CommonQuadraticEvidence(
            StabilityStatus.VIOLATED,
            "outward-rounded rate has no positive asymptotic denominator",
            contraction,
            maximum,
            tolerance,
            foster_rate,
            additive,
        )
    asymptotic_envelope = float(
        _outward_upper(
            additive / envelope_denominator,
            matrix.dtype,
            count=2,
            scale=additive / envelope_denominator,
        )
    )
    if not np.all(np.isfinite((foster_rate, additive, asymptotic_envelope))):
        return CommonQuadraticEvidence(
            StabilityStatus.DIVERGED, "common Foster bound is non-finite"
        )
    return CommonQuadraticEvidence(
        StabilityStatus.SATISFIED,
        "common quadratic Foster bound holds under arbitrary switching",
        contraction,
        maximum,
        tolerance,
        foster_rate,
        additive,
        asymptotic_envelope,
    )


def _mode_contractions(
    dynamics_matrix: np.ndarray,
    lyapunov_matrix: np.ndarray,
) -> np.ndarray:
    regimes = dynamics_matrix.shape[0]
    declared_dtype = dynamics_matrix.dtype
    proof_matrix = dynamics_matrix.astype(np.float64)
    proof_metric = lyapunov_matrix.astype(np.float64)
    roots = np.linalg.cholesky(proof_metric)
    contraction = np.empty((regimes, regimes), dtype=np.float64)
    for source in range(regimes):
        for destination in range(regimes):
            numerator = roots[destination].T @ proof_matrix[destination]
            transformed = np.linalg.solve(roots[source], numerator.T).T
            singular = np.linalg.svd(transformed, compute_uv=False)[0]
            raw = singular * singular
            condition = float(np.linalg.cond(roots[source]))
            scale = (
                float(np.sum(np.abs(transformed))) ** 2
                * max(1.0, condition)
            )
            contraction[source, destination] = (
                0.0
                if np.all(transformed == 0.0)
                else _outward_upper(
                    raw,
                    declared_dtype,
                    count=dynamics_matrix.shape[-1] ** 3,
                    scale=scale,
                )
            )
    return contraction


def recurrence_stability_certificate(
    dynamics_matrix: Any,
    dynamics_offset: Any,
    dynamics_covariance: Any,
    recurrence_weights: Any,
    recurrence_offsets: Any,
    lyapunov_matrix: Any,
    *,
    foster_rate: float,
    tail_radius: float,
    compact_splits: int,
    maximum_tail_cones: int = 4096,
) -> RecurrenceCertificateEvidence:
    """Certify a global recurrence-aware Foster bound."""
    from cffg.switching_stability_certificate import (
        recurrence_stability_certificate as certify,
    )

    return certify(
        dynamics_matrix,
        dynamics_offset,
        dynamics_covariance,
        recurrence_weights,
        recurrence_offsets,
        lyapunov_matrix,
        foster_rate=foster_rate,
        tail_radius=tail_radius,
        compact_splits=compact_splits,
        maximum_tail_cones=maximum_tail_cones,
    )
def evaluate_recurrence_drift(
    states: Any,
    source_regime: Any,
    dynamics_matrix: Any,
    dynamics_offset: Any,
    dynamics_covariance: Any,
    recurrence_weights: Any,
    recurrence_offsets: Any,
    lyapunov_matrix: Any,
    *,
    foster_rate: float,
    additive_bound: float,
) -> RecurrenceDriftEvidence:
    """Evaluate the recurrence-aware merit on declared finite probes."""
    from cffg.switching_stability_diagnostics import (
        evaluate_recurrence_drift as evaluate,
    )

    return evaluate(
        states,
        source_regime,
        dynamics_matrix,
        dynamics_offset,
        dynamics_covariance,
        recurrence_weights,
        recurrence_offsets,
        lyapunov_matrix,
        foster_rate=foster_rate,
        additive_bound=additive_bound,
    )


__all__ = [
    "CommonQuadraticEvidence",
    "OneStepDriftEvidence",
    "RecurrenceCertificateEvidence",
    "RecurrenceDriftEvidence",
    "StabilityStatus",
    "common_quadratic_certificate",
    "evaluate_one_step_drift",
    "evaluate_recurrence_drift",
    "one_step_quadratic_drift",
    "recurrence_drift_merit",
    "recurrence_drift_residuals",
    "recurrence_stability_certificate",
    "recurrent_transition_probabilities",
]

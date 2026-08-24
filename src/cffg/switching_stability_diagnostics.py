"""Validated descriptive diagnostics for switching-stability authorities."""

from __future__ import annotations

from typing import Any

import numpy as np

from cffg.switching_stability import (
    RecurrenceDriftEvidence,
    StabilityStatus,
    _dtype_tolerance,
    _invalid_status,
    _validate_model,
    evaluate_one_step_drift,
)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0.0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _transition_probabilities(
    states: np.ndarray,
    weights: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    logits = np.einsum("nd,ild->nil", states, weights) + offsets[None, :, :]
    accept = _sigmoid(logits)
    reject = 1.0 - accept
    prefix = np.concatenate(
        (
            np.ones((*accept.shape[:-1], 1), dtype=accept.dtype),
            np.cumprod(reject, axis=-1),
        ),
        axis=-1,
    )
    return np.concatenate(
        (accept * prefix[..., :-1], prefix[..., -1:]), axis=-1
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
    model, error = _validate_model(
        dynamics_matrix,
        dynamics_offset,
        dynamics_covariance,
        lyapunov_matrix,
    )
    if error is not None:
        return RecurrenceDriftEvidence(*error, scope="descriptive_finite_probe")
    matrix, offset, covariance, metric = model
    state = np.asarray(states)
    source = np.asarray(source_regime)
    weights = np.asarray(recurrence_weights)
    recurrence_offset = np.asarray(recurrence_offsets)
    regimes, dimension, _ = matrix.shape
    if not np.all(np.isfinite((foster_rate, additive_bound))):
        return RecurrenceDriftEvidence(
            StabilityStatus.DIVERGED,
            "recurrence settings are non-finite",
            "descriptive_finite_probe",
        )
    if (
        state.ndim != 2
        or not state.shape[0]
        or state.shape[1] != dimension
        or state.dtype != matrix.dtype
        or source.shape != (state.shape[0],)
        or not np.issubdtype(source.dtype, np.integer)
        or weights.shape != (regimes, regimes - 1, dimension)
        or recurrence_offset.shape != (regimes, regimes - 1)
        or weights.dtype != matrix.dtype
        or recurrence_offset.dtype != matrix.dtype
        or np.any(source < 0)
        or np.any(source >= regimes)
        or not 0.0 <= foster_rate < 1.0
        or additive_bound < 0.0
    ):
        return RecurrenceDriftEvidence(
            StabilityStatus.INFEASIBLE,
            "recurrence probes, parameters, or rate are infeasible",
            "descriptive_finite_probe",
        )
    invalid = _invalid_status((state, weights, recurrence_offset))
    if invalid is not None:
        return RecurrenceDriftEvidence(*invalid, scope="descriptive_finite_probe")
    all_probability = _transition_probabilities(state, weights, recurrence_offset)
    probability = all_probability[np.arange(state.shape[0]), source]
    one_step = evaluate_one_step_drift(
        state,
        source,
        probability,
        matrix,
        offset,
        covariance,
        metric,
    )
    if one_step.status is not StabilityStatus.SATISFIED:
        return RecurrenceDriftEvidence(
            one_step.status, one_step.reason, "descriptive_finite_probe"
        )
    current = one_step.current_value
    expected = one_step.expected_next_value
    drift = one_step.drift
    residual = expected - foster_rate * current - additive_bound
    arrays = (probability, current, expected, drift, residual)
    invalid = _invalid_status(arrays)
    if invalid is not None:
        return RecurrenceDriftEvidence(*invalid, scope="descriptive_finite_probe")
    probability, current, expected, drift, residual = arrays
    try:
        tolerance = np.asarray(
            [
                _dtype_tolerance(
                    matrix.dtype,
                    current[index],
                    expected[index],
                    additive_bound,
                    count=regimes * dimension * dimension,
                )
                for index in range(state.shape[0])
            ]
        )
    except ValueError as exception:
        return RecurrenceDriftEvidence(
            StabilityStatus.INFEASIBLE,
            str(exception),
            "descriptive_finite_probe",
        )
    scale = (
        1.0 + np.abs(current) + np.abs(expected) + abs(float(additive_bound))
    )
    merit = float(np.mean(np.square(np.maximum(residual, 0.0) / scale)))
    if not np.isfinite(merit):
        return RecurrenceDriftEvidence(
            StabilityStatus.DIVERGED,
            "recurrence-aware merit is non-finite",
            "descriptive_finite_probe",
        )
    status = (
        StabilityStatus.SATISFIED
        if np.all(residual <= tolerance)
        else StabilityStatus.VIOLATED
    )
    reason = (
        "all declared probe drift inequalities hold"
        if status is StabilityStatus.SATISFIED
        else "at least one declared probe drift inequality is violated"
    )
    return RecurrenceDriftEvidence(
        status,
        reason,
        "descriptive_finite_probe",
        probability,
        current,
        expected,
        drift,
        float(additive_bound),
        residual,
        tolerance,
        merit,
    )


__all__ = ["evaluate_recurrence_drift"]

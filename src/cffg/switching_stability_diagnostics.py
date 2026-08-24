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
    recurrence_drift_merit,
    recurrence_drift_residuals,
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
    probability, current, expected, drift, residual = recurrence_drift_residuals(
        state,
        source,
        matrix,
        offset,
        covariance,
        weights,
        recurrence_offset,
        metric,
        foster_rate,
        additive_bound,
    )
    arrays = tuple(
        np.asarray(value)
        for value in (probability, current, expected, drift, residual)
    )
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
    merit = float(
        recurrence_drift_merit(
            state,
            source,
            matrix,
            offset,
            covariance,
            weights,
            recurrence_offset,
            metric,
            foster_rate,
            additive_bound,
        )
    )
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

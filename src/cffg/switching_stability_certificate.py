"""Outward-rounded recurrence-aware global stability certificate."""

from __future__ import annotations

from typing import Any

import numpy as np

from cffg.switching_stability import (
    RecurrenceCertificateEvidence,
    StabilityStatus,
    _covariance_upper,
    _dtype_tolerance,
    _invalid_status,
    _mode_contractions,
    _outward_upper,
    _outward_positive_difference_lower,
    _proof_guard,
    _validate_model,
)


def _orthant_signs(dimension: int, dtype: np.dtype[Any]) -> np.ndarray:
    count = 1 << dimension
    bits = np.arange(count, dtype=np.uint64)[:, None]
    axes = np.arange(dimension, dtype=np.uint64)[None, :]
    positive = ((bits >> axes) & 1).astype(bool)
    return np.where(positive, 1.0, -1.0).astype(dtype)


def _suppression_events(
    destination: int,
    signed_weight: np.ndarray,
    recurrence_offset: np.ndarray,
    dtype: np.dtype[Any],
) -> tuple[tuple[float, float], ...]:
    events: list[tuple[float, float]] = []
    if destination < signed_weight.shape[0]:
        accept = signed_weight[destination]
        tolerance = _dtype_tolerance(
            dtype,
            accept,
            recurrence_offset[destination],
            count=accept.shape[0] + 1,
        )
        rate = float(np.min(-accept)) - tolerance
        if rate > tolerance:
            events.append(
                (rate, float(recurrence_offset[destination]) + tolerance)
            )
    for gate in range(destination):
        reject = signed_weight[gate]
        tolerance = _dtype_tolerance(
            dtype,
            reject,
            recurrence_offset[gate],
            count=reject.shape[0] + 1,
        )
        rate = float(np.min(reject)) - tolerance
        if rate > tolerance:
            events.append((rate, float(-recurrence_offset[gate]) + tolerance))
    return tuple(events)


def _exponential_polynomial_supremum(
    intercept: float,
    decay_rate: float,
    radius: float,
    coefficients: tuple[float, float, float],
    dtype: np.dtype[Any],
) -> float:
    quadratic, linear, constant = coefficients
    transition = float(
        _outward_upper(
            intercept / decay_rate,
            dtype,
            count=2,
            scale=abs(intercept / decay_rate),
        )
    )

    def monomial(power: int) -> float:
        saturated = (
            float(np.power(np.float64(transition), power))
            if transition >= radius
            else 0.0
        )
        exponential_start = max(radius, transition)
        stationary = (
            0.0
            if power == 0
            else float(
                _outward_upper(
                    power / decay_rate,
                    dtype,
                    count=1,
                    scale=power / decay_rate,
                )
            )
        )
        point = max(exponential_start, stationary)
        exponent_scale = abs(intercept) + abs(decay_rate * point)
        exponent = min(
            0.0,
            float(
                _outward_upper(
                    intercept - decay_rate * point,
                    dtype,
                    count=2,
                    scale=exponent_scale,
                )
            ),
        )
        exponential = np.exp(exponent) * np.power(np.float64(point), power)
        return float(
            _outward_upper(
                max(float(saturated), float(exponential)),
                dtype,
                count=4,
                scale=max(abs(float(saturated)), abs(float(exponential))),
            )
        )

    terms = np.asarray(
        (
            quadratic * monomial(2),
            linear * monomial(1),
            constant * monomial(0),
        )
    )
    return float(
        _outward_upper(
            np.sum(terms), dtype, count=12, scale=np.sum(np.abs(terms))
        )
    )


def _destination_polynomial_bounds(
    dynamics_matrix: np.ndarray,
    dynamics_offset: np.ndarray,
    dynamics_covariance: np.ndarray,
    lyapunov_matrix: np.ndarray,
) -> np.ndarray:
    regimes = dynamics_matrix.shape[0]
    declared_dtype = dynamics_matrix.dtype
    proof_matrix = dynamics_matrix.astype(np.float64)
    proof_offset = dynamics_offset.astype(np.float64)
    proof_covariance = dynamics_covariance.astype(np.float64)
    proof_metric = lyapunov_matrix.astype(np.float64)
    coefficients = np.empty((regimes, 3), dtype=np.float64)
    for destination in range(regimes):
        metric_maximum = float(
            _outward_upper(
                np.max(np.linalg.eigvalsh(proof_metric[destination])),
                declared_dtype,
                count=dynamics_matrix.shape[-1] ** 3,
                scale=np.sum(np.abs(proof_metric[destination])),
            )
        )
        matrix_norm = float(
            _outward_upper(
                np.linalg.svd(proof_matrix[destination], compute_uv=False)[0],
                declared_dtype,
                count=dynamics_matrix.shape[-1] ** 3,
                scale=np.sum(np.abs(proof_matrix[destination])),
            )
        )
        offset_norm = float(
            _outward_upper(
                np.linalg.norm(proof_offset[destination]),
                declared_dtype,
                count=dynamics_matrix.shape[-1],
                scale=np.sum(np.abs(proof_offset[destination])),
            )
        )
        noise_terms = (
            proof_metric[destination] * proof_covariance[destination].T
        )
        noise = float(
            _outward_upper(
                np.sum(noise_terms),
                declared_dtype,
                count=dynamics_matrix.shape[-1] ** 2,
                scale=np.sum(np.abs(noise_terms)),
            )
        )
        raw = np.asarray(
            (
                metric_maximum * matrix_norm**2,
                2.0 * metric_maximum * matrix_norm * offset_norm,
                metric_maximum * offset_norm**2 + noise,
            )
        )
        coefficients[destination] = _outward_upper(
            raw,
            declared_dtype,
            count=4,
            scale=np.abs(raw),
        )
    return coefficients


def _safe_mode_additive_bounds(
    contraction: np.ndarray,
    foster_rate: float,
    dynamics_offset: np.ndarray,
    dynamics_covariance: np.ndarray,
    lyapunov_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    regimes = contraction.shape[0]
    safe = np.zeros((regimes, regimes), dtype=bool)
    additive = np.full((regimes, regimes), np.inf, dtype=np.float64)
    declared_dtype = dynamics_offset.dtype
    proof_offset = dynamics_offset.astype(np.float64)
    proof_covariance = dynamics_covariance.astype(np.float64)
    proof_metric = lyapunov_matrix.astype(np.float64)
    for source in range(regimes):
        for destination in range(regimes):
            alpha = float(contraction[source, destination])
            tolerance = _dtype_tolerance(
                declared_dtype,
                alpha,
                count=dynamics_offset.shape[-1] ** 3,
            )
            contraction_gap = _outward_positive_difference_lower(
                foster_rate, alpha, declared_dtype
            )
            if contraction_gap <= tolerance:
                continue
            safe[source, destination] = True
            offset_terms = (
                proof_offset[destination, :, None]
                * proof_metric[destination]
                * proof_offset[destination, None, :]
            )
            offset_energy = float(
                _outward_upper(
                    np.sum(offset_terms),
                    declared_dtype,
                    count=dynamics_offset.shape[-1] ** 2,
                    scale=np.sum(np.abs(offset_terms)),
                )
            )
            noise_terms = (
                proof_metric[destination] * proof_covariance[destination].T
            )
            noise = float(
                _outward_upper(
                    np.sum(noise_terms),
                    declared_dtype,
                    count=dynamics_offset.shape[-1] ** 2,
                    scale=np.sum(np.abs(noise_terms)),
                )
            )
            if alpha == 0.0:
                raw = offset_energy + noise
            else:
                offset_multiplier = float(
                    _outward_upper(
                        foster_rate / contraction_gap,
                        declared_dtype,
                        count=2,
                        scale=foster_rate / contraction_gap,
                    )
                )
                raw = offset_multiplier * offset_energy + noise
            additive[source, destination] = _outward_upper(
                raw,
                declared_dtype,
                count=6,
                scale=abs(raw),
            )
    return safe, additive


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0.0
    output = np.empty_like(value)
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _compact_probability_upper(
    lower: np.ndarray,
    upper: np.ndarray,
    weights: np.ndarray,
    offsets: np.ndarray,
    declared_dtype: np.dtype[Any],
) -> np.ndarray:
    positive = weights >= 0.0
    logit_lower = np.sum(np.where(positive, weights * lower, weights * upper), axis=-1)
    logit_upper = np.sum(np.where(positive, weights * upper, weights * lower), axis=-1)
    logit_lower = logit_lower + offsets
    logit_upper = logit_upper + offsets
    logit_scale = (
        np.sum(
            np.maximum(np.abs(weights * lower), np.abs(weights * upper)),
            axis=-1,
        )
        + np.abs(offsets)
    )
    logit_guard = _proof_guard(
        declared_dtype, count=weights.shape[-1] + 1, scale=logit_scale
    )
    logit_lower = np.nextafter(logit_lower - logit_guard, -np.inf)
    logit_upper = np.nextafter(logit_upper + logit_guard, np.inf)
    accept_upper = np.minimum(
        1.0,
        _outward_upper(
            _sigmoid(logit_upper), declared_dtype, count=8, scale=1.0
        ),
    )
    reject_upper = np.minimum(
        1.0,
        _outward_upper(
            _sigmoid(-logit_lower), declared_dtype, count=8, scale=1.0
        ),
    )
    prefix = np.ones((weights.shape[0], weights.shape[1] + 1), dtype=np.float64)
    for gate in range(weights.shape[1]):
        prefix[:, gate + 1] = np.minimum(
            1.0,
            _outward_upper(
                prefix[:, gate] * reject_upper[:, gate],
                declared_dtype,
                count=2,
                scale=1.0,
            ),
        )
    accepted = np.minimum(
        1.0,
        _outward_upper(
            accept_upper * prefix[:, :-1],
            declared_dtype,
            count=2,
            scale=1.0,
        ),
    )
    return np.concatenate((accepted, prefix[:, -1:]), axis=-1)


def _compact_additive_bound(
    radius: float,
    splits: int,
    dynamics_matrix: np.ndarray,
    dynamics_offset: np.ndarray,
    dynamics_covariance: np.ndarray,
    recurrence_weights: np.ndarray,
    recurrence_offsets: np.ndarray,
    lyapunov_matrix: np.ndarray,
) -> tuple[float, int]:
    dimension = dynamics_matrix.shape[-1]
    declared_dtype = dynamics_matrix.dtype
    proof_matrix = dynamics_matrix.astype(np.float64)
    proof_offset = dynamics_offset.astype(np.float64)
    proof_covariance = dynamics_covariance.astype(np.float64)
    proof_weights = recurrence_weights.astype(np.float64)
    proof_recurrence_offset = recurrence_offsets.astype(np.float64)
    proof_metric = lyapunov_matrix.astype(np.float64)
    edges = np.linspace(-radius, radius, splits + 1, dtype=np.float64)
    maximum = 0.0
    box_count = splits**dimension
    for flat_index in range(box_count):
        coordinate = np.unravel_index(flat_index, (splits,) * dimension)
        lower = np.asarray([edges[index] for index in coordinate])
        upper = np.asarray([edges[index + 1] for index in coordinate])
        probability_upper = _compact_probability_upper(
            lower,
            upper,
            proof_weights,
            proof_recurrence_offset,
            declared_dtype,
        )
        destination_upper = np.empty(dynamics_matrix.shape[0], dtype=np.float64)
        for destination in range(dynamics_matrix.shape[0]):
            row_positive = proof_matrix[destination] >= 0.0
            mean_lower = np.sum(
                np.where(
                    row_positive,
                    proof_matrix[destination] * lower,
                    proof_matrix[destination] * upper,
                ),
                axis=-1,
            ) + proof_offset[destination]
            mean_upper = np.sum(
                np.where(
                    row_positive,
                    proof_matrix[destination] * upper,
                    proof_matrix[destination] * lower,
                ),
                axis=-1,
            ) + proof_offset[destination]
            mean_scale = (
                np.sum(
                    np.maximum(
                        np.abs(proof_matrix[destination] * lower),
                        np.abs(proof_matrix[destination] * upper),
                    ),
                    axis=-1,
                )
                + np.abs(proof_offset[destination])
            )
            mean_guard = _proof_guard(
                declared_dtype, count=dimension + 1, scale=mean_scale
            )
            mean_lower = np.nextafter(mean_lower - mean_guard, -np.inf)
            mean_upper = np.nextafter(mean_upper + mean_guard, np.inf)
            maximum_absolute = np.maximum(np.abs(mean_lower), np.abs(mean_upper))
            metric_maximum = float(
                _outward_upper(
                    np.max(np.linalg.eigvalsh(proof_metric[destination])),
                    declared_dtype,
                    count=dimension**3,
                    scale=np.sum(np.abs(proof_metric[destination])),
                )
            )
            noise_terms = (
                proof_metric[destination] * proof_covariance[destination].T
            )
            noise = float(
                _outward_upper(
                    np.sum(noise_terms),
                    declared_dtype,
                    count=dimension**2,
                    scale=np.sum(np.abs(noise_terms)),
                )
            )
            mean_square = float(
                _outward_upper(
                    maximum_absolute @ maximum_absolute,
                    declared_dtype,
                    count=dimension,
                    scale=np.sum(np.square(maximum_absolute)),
                )
            )
            raw_destination = metric_maximum * mean_square + noise
            destination_upper[destination] = _outward_upper(
                raw_destination,
                declared_dtype,
                count=3,
                scale=abs(raw_destination),
            )
        source_upper = _outward_upper(
            probability_upper @ destination_upper,
            declared_dtype,
            count=dynamics_matrix.shape[0],
            scale=np.sum(
                probability_upper * destination_upper[None, :], axis=-1
            ),
        )
        maximum = max(maximum, float(np.max(source_upper)))
    return float(
        _outward_upper(
            maximum, declared_dtype, count=1, scale=maximum
        )
    ), box_count


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
    """Certify a global recurrence-aware Foster bound.

    Closed coordinate orthants cover every tail direction, including their tie
    faces.  A destination that is not quadratically contractive must have one
    stick event whose probability decays exponentially throughout the entire
    orthant.  A deterministic interval cover bounds the remaining compact box.
    """
    model, error = _validate_model(
        dynamics_matrix,
        dynamics_offset,
        dynamics_covariance,
        lyapunov_matrix,
    )
    if error is not None:
        return RecurrenceCertificateEvidence(*error)
    matrix, offset, covariance, metric = model
    covariance_upper = _covariance_upper(covariance)
    weights = np.asarray(recurrence_weights)
    recurrence_offset = np.asarray(recurrence_offsets)
    regimes, dimension, _ = matrix.shape
    if not np.all(np.isfinite((foster_rate, tail_radius))):
        return RecurrenceCertificateEvidence(
            StabilityStatus.DIVERGED, "certificate settings are non-finite"
        )
    if (
        weights.shape != (regimes, regimes - 1, dimension)
        or recurrence_offset.shape != (regimes, regimes - 1)
        or weights.dtype != matrix.dtype
        or recurrence_offset.dtype != matrix.dtype
        or not 0.0 < foster_rate < 1.0
        or tail_radius <= 0.0
        or isinstance(compact_splits, bool)
        or not isinstance(compact_splits, int)
        or compact_splits < 1
        or isinstance(maximum_tail_cones, bool)
        or not isinstance(maximum_tail_cones, int)
        or maximum_tail_cones < 1
        or (1 << dimension) > maximum_tail_cones
        or compact_splits**dimension > 1_000_000
    ):
        return RecurrenceCertificateEvidence(
            StabilityStatus.INFEASIBLE,
            "recurrence shapes, dtypes, or certificate settings are infeasible",
        )
    invalid = _invalid_status((weights, recurrence_offset))
    if invalid is not None:
        return RecurrenceCertificateEvidence(*invalid)
    try:
        contraction = _mode_contractions(matrix, metric)
    except ValueError as exception:
        return RecurrenceCertificateEvidence(
            StabilityStatus.INFEASIBLE, str(exception)
        )
    except np.linalg.LinAlgError as exception:
        return RecurrenceCertificateEvidence(
            StabilityStatus.DIVERGED,
            "contraction decomposition failed: " + str(exception),
        )
    signs = _orthant_signs(dimension, matrix.dtype)
    safe, safe_additive = _safe_mode_additive_bounds(
        contraction,
        foster_rate,
        offset,
        covariance_upper,
        metric,
    )
    shape = (signs.shape[0], regimes, regimes)
    suppressed = np.zeros(shape, dtype=bool)
    decay_rate = np.zeros(shape, dtype=np.float64)
    decay_intercept = np.zeros(shape, dtype=np.float64)
    tail_additive = np.zeros((signs.shape[0], regimes), dtype=np.float64)
    try:
        polynomial = _destination_polynomial_bounds(
            matrix, offset, covariance_upper, metric
        )
    except np.linalg.LinAlgError as exception:
        return RecurrenceCertificateEvidence(
            StabilityStatus.DIVERGED,
            "tail decomposition failed: " + str(exception),
        )
    uncovered: list[tuple[int, int, int]] = []
    for cone, sign in enumerate(signs):
        for source in range(regimes):
            safe_bound = 0.0
            suppressed_terms: list[float] = []
            for destination in range(regimes):
                if safe[source, destination]:
                    safe_bound = max(
                        safe_bound, float(safe_additive[source, destination])
                    )
                    continue
                events = _suppression_events(
                    destination,
                    weights[source] * sign[None, :],
                    recurrence_offset[source],
                    matrix.dtype,
                )
                if not events:
                    uncovered.append((cone, source, destination))
                    continue
                candidates = [
                    (
                        _exponential_polynomial_supremum(
                            intercept,
                            rate,
                            tail_radius,
                            tuple(float(value) for value in polynomial[destination]),
                            matrix.dtype,
                        ),
                        rate,
                        intercept,
                    )
                    for rate, intercept in events
                ]
                bound, rate, intercept = min(candidates)
                suppressed[cone, source, destination] = True
                decay_rate[cone, source, destination] = rate
                decay_intercept[cone, source, destination] = intercept
                suppressed_terms.append(bound)
            raw_tail = safe_bound + float(np.sum(suppressed_terms))
            tail_additive[cone, source] = _outward_upper(
                raw_tail,
                matrix.dtype,
                count=regimes + 1,
                scale=safe_bound + float(np.sum(np.abs(suppressed_terms))),
            )
    if uncovered:
        return RecurrenceCertificateEvidence(
            StabilityStatus.VIOLATED,
            f"{len(uncovered)} tail source/cone destinations are neither "
            "contractive nor exponentially suppressed",
            foster_rate,
            tail_radius,
            signs,
            np.broadcast_to(contraction, shape).copy(),
            suppressed,
            decay_rate,
            decay_intercept,
            tail_additive,
        )
    try:
        compact_additive, box_count = _compact_additive_bound(
            tail_radius,
            compact_splits,
            matrix,
            offset,
            covariance_upper,
            weights,
            recurrence_offset,
            metric,
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exception:
        return RecurrenceCertificateEvidence(
            StabilityStatus.DIVERGED, f"compact verification failed: {exception}"
        )
    additive = float(
        _outward_upper(
            max(float(np.max(tail_additive)), compact_additive),
            matrix.dtype,
            count=1,
            scale=max(float(np.max(tail_additive)), compact_additive),
        )
    )
    envelope_denominator = _outward_positive_difference_lower(
        1.0, foster_rate, matrix.dtype
    )
    if envelope_denominator <= 0.0:
        return RecurrenceCertificateEvidence(
            StabilityStatus.VIOLATED,
            "outward-rounded rate has no positive asymptotic denominator",
        )
    asymptotic_envelope = float(
        _outward_upper(
            additive / envelope_denominator,
            matrix.dtype,
            count=2,
            scale=additive / envelope_denominator,
        )
    )
    if (
        not np.all(np.isfinite(tail_additive))
        or not np.isfinite(compact_additive)
        or not np.isfinite(additive)
        or not np.isfinite(asymptotic_envelope)
    ):
        return RecurrenceCertificateEvidence(
            StabilityStatus.DIVERGED, "certificate evidence is non-finite"
        )
    return RecurrenceCertificateEvidence(
        StabilityStatus.SATISFIED,
        "global Foster bound certified by all orthant tails and compact cover",
        foster_rate,
        tail_radius,
        signs,
        np.broadcast_to(contraction, shape).copy(),
        suppressed,
        decay_rate,
        decay_intercept,
        tail_additive,
        compact_additive,
        additive,
        asymptotic_envelope,
        box_count,
    )




__all__ = ["recurrence_stability_certificate"]

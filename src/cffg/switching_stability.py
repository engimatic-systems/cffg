"""Truth-free quadratic-drift authorities for recurrent switching dynamics.

Rows of a recurrence block are source regimes.  Columns of every returned
transition table are destination regimes, and destination ``j`` selects
``(A[j], b[j], Q[j])``.  Finite-probe evidence is deliberately not a global
stability certificate; :func:`common_quadratic_certificate` owns that stronger,
more conservative claim.
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
    moment_bound: float | None = None


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
    lyapunov_moment_bound: float | None = None
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
    residual = drift + jnp.asarray(foster_rate, states.dtype) * current
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
    values = tuple(
        np.asarray(value)
        for value in one_step_quadratic_drift(
            state, source, probability, matrix, offset, covariance, metric
        )
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
    common = metric[0]
    root = np.linalg.cholesky(common)
    transformed = np.stack(
        [
            np.linalg.solve(root, (root.T @ value).T).T
            for value in matrix
        ]
    )
    contraction = np.square(np.linalg.svd(transformed, compute_uv=False)[:, 0])
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
        foster_rate = float(np.sqrt(maximum))
        young_factor = foster_rate / maximum - 1.0
        offset_multiplier = 1.0 + 1.0 / young_factor
    offset_energy = np.einsum("kd,de,ke->k", offset, common, offset)
    noise_energy = np.einsum("de,ked->k", common, covariance)
    additive = float(np.max(offset_multiplier * offset_energy + noise_energy))
    moment_bound = additive / (1.0 - foster_rate)
    if not np.all(np.isfinite((foster_rate, additive, moment_bound))):
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
        moment_bound,
    )


def _mode_contractions(
    dynamics_matrix: np.ndarray,
    lyapunov_matrix: np.ndarray,
) -> np.ndarray:
    regimes = dynamics_matrix.shape[0]
    roots = np.linalg.cholesky(lyapunov_matrix)
    contraction = np.empty((regimes, regimes), dtype=dynamics_matrix.dtype)
    for source in range(regimes):
        for destination in range(regimes):
            numerator = roots[destination].T @ dynamics_matrix[destination]
            transformed = np.linalg.solve(roots[source], numerator.T).T
            singular = np.linalg.svd(transformed, compute_uv=False)[0]
            contraction[source, destination] = singular * singular
    return contraction


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
    tolerance: float,
) -> tuple[tuple[float, float], ...]:
    events: list[tuple[float, float]] = []
    if destination < signed_weight.shape[0]:
        accept = signed_weight[destination]
        rate = float(np.min(-accept))
        if rate > tolerance:
            events.append((rate, float(recurrence_offset[destination])))
    for gate in range(destination):
        reject = signed_weight[gate]
        rate = float(np.min(reject))
        if rate > tolerance:
            events.append((rate, float(-recurrence_offset[gate])))
    return tuple(events)


def _exponential_polynomial_supremum(
    intercept: float,
    decay_rate: float,
    radius: float,
    coefficients: tuple[float, float, float],
) -> float:
    quadratic, linear, constant = coefficients
    transition = intercept / decay_rate

    def monomial(power: int) -> float:
        saturated = transition**power if transition >= radius else 0.0
        exponential_start = max(radius, transition)
        stationary = 0.0 if power == 0 else power / decay_rate
        point = max(exponential_start, stationary)
        exponent = min(0.0, intercept - decay_rate * point)
        exponential = np.exp(exponent) * point**power
        return max(float(saturated), float(exponential))

    return quadratic * monomial(2) + linear * monomial(1) + constant * monomial(0)


def _destination_polynomial_bounds(
    dynamics_matrix: np.ndarray,
    dynamics_offset: np.ndarray,
    dynamics_covariance: np.ndarray,
    lyapunov_matrix: np.ndarray,
) -> np.ndarray:
    regimes = dynamics_matrix.shape[0]
    coefficients = np.empty((regimes, 3), dtype=np.float64)
    for destination in range(regimes):
        metric_maximum = float(np.max(np.linalg.eigvalsh(lyapunov_matrix[destination])))
        matrix_norm = float(np.linalg.svd(dynamics_matrix[destination], compute_uv=False)[0])
        offset_norm = float(np.linalg.norm(dynamics_offset[destination]))
        noise = float(
            np.trace(lyapunov_matrix[destination] @ dynamics_covariance[destination])
        )
        coefficients[destination] = (
            metric_maximum * matrix_norm**2,
            2.0 * metric_maximum * matrix_norm * offset_norm,
            metric_maximum * offset_norm**2 + noise,
        )
    return coefficients


def _safe_mode_additive_bounds(
    contraction: np.ndarray,
    foster_rate: float,
    dynamics_offset: np.ndarray,
    dynamics_covariance: np.ndarray,
    lyapunov_matrix: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    safe = contraction + tolerance < foster_rate
    regimes = contraction.shape[0]
    additive = np.full((regimes, regimes), np.inf, dtype=np.float64)
    for source in range(regimes):
        for destination in range(regimes):
            alpha = float(contraction[source, destination])
            if not safe[source, destination]:
                continue
            offset_energy = float(
                dynamics_offset[destination]
                @ lyapunov_matrix[destination]
                @ dynamics_offset[destination]
            )
            noise = float(
                np.trace(
                    lyapunov_matrix[destination]
                    @ dynamics_covariance[destination]
                )
            )
            if alpha == 0.0:
                additive[source, destination] = offset_energy + noise
            else:
                young_factor = foster_rate / alpha - 1.0
                additive[source, destination] = (
                    (1.0 + 1.0 / young_factor) * offset_energy + noise
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
) -> np.ndarray:
    positive = weights >= 0.0
    logit_lower = np.sum(np.where(positive, weights * lower, weights * upper), axis=-1)
    logit_upper = np.sum(np.where(positive, weights * upper, weights * lower), axis=-1)
    logit_lower = logit_lower + offsets
    logit_upper = logit_upper + offsets
    accept_upper = _sigmoid(logit_upper)
    reject_upper = _sigmoid(-logit_lower)
    prefix = np.concatenate(
        (np.ones((weights.shape[0], 1), dtype=weights.dtype), np.cumprod(reject_upper, axis=-1)),
        axis=-1,
    )
    return np.concatenate(
        (accept_upper * prefix[:, :-1], prefix[:, -1:]), axis=-1
    )


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
    edges = np.linspace(-radius, radius, splits + 1, dtype=dynamics_matrix.dtype)
    maximum = 0.0
    box_count = splits**dimension
    for flat_index in range(box_count):
        coordinate = np.unravel_index(flat_index, (splits,) * dimension)
        lower = np.asarray([edges[index] for index in coordinate])
        upper = np.asarray([edges[index + 1] for index in coordinate])
        probability_upper = _compact_probability_upper(
            lower, upper, recurrence_weights, recurrence_offsets
        )
        destination_upper = np.empty(dynamics_matrix.shape[0], dtype=np.float64)
        for destination in range(dynamics_matrix.shape[0]):
            row_positive = dynamics_matrix[destination] >= 0.0
            mean_lower = np.sum(
                np.where(
                    row_positive,
                    dynamics_matrix[destination] * lower,
                    dynamics_matrix[destination] * upper,
                ),
                axis=-1,
            ) + dynamics_offset[destination]
            mean_upper = np.sum(
                np.where(
                    row_positive,
                    dynamics_matrix[destination] * upper,
                    dynamics_matrix[destination] * lower,
                ),
                axis=-1,
            ) + dynamics_offset[destination]
            maximum_absolute = np.maximum(np.abs(mean_lower), np.abs(mean_upper))
            metric_maximum = float(
                np.max(np.linalg.eigvalsh(lyapunov_matrix[destination]))
            )
            noise = float(
                np.trace(
                    lyapunov_matrix[destination]
                    @ dynamics_covariance[destination]
                )
            )
            destination_upper[destination] = (
                metric_maximum * float(maximum_absolute @ maximum_absolute) + noise
            )
        maximum = max(
            maximum,
            float(np.max(probability_upper @ destination_upper)),
        )
    return maximum, box_count


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
        recurrence_tolerance = _dtype_tolerance(
            matrix.dtype,
            weights,
            recurrence_offset,
            count=max(1, dimension),
        )
        contraction = _mode_contractions(matrix, metric)
        contraction_tolerance = _dtype_tolerance(
            matrix.dtype, contraction, count=dimension**3
        )
    except (ValueError, np.linalg.LinAlgError) as exception:
        return RecurrenceCertificateEvidence(
            StabilityStatus.INFEASIBLE, str(exception)
        )
    signs = _orthant_signs(dimension, matrix.dtype)
    safe, safe_additive = _safe_mode_additive_bounds(
        contraction,
        foster_rate,
        offset,
        covariance,
        metric,
        contraction_tolerance,
    )
    shape = (signs.shape[0], regimes, regimes)
    suppressed = np.zeros(shape, dtype=bool)
    decay_rate = np.zeros(shape, dtype=matrix.dtype)
    decay_intercept = np.zeros(shape, dtype=matrix.dtype)
    tail_additive = np.zeros((signs.shape[0], regimes), dtype=np.float64)
    polynomial = _destination_polynomial_bounds(matrix, offset, covariance, metric)
    uncovered: list[tuple[int, int, int]] = []
    for cone, sign in enumerate(signs):
        for source in range(regimes):
            safe_bound = 0.0
            suppressed_bound = 0.0
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
                    recurrence_tolerance,
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
                suppressed_bound += bound
            tail_additive[cone, source] = safe_bound + suppressed_bound
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
            covariance,
            weights,
            recurrence_offset,
            metric,
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exception:
        return RecurrenceCertificateEvidence(
            StabilityStatus.DIVERGED, f"compact verification failed: {exception}"
        )
    additive = max(float(np.max(tail_additive)), compact_additive)
    moment_bound = additive / (1.0 - foster_rate)
    if (
        not np.all(np.isfinite(tail_additive))
        or not np.isfinite(compact_additive)
        or not np.isfinite(additive)
        or not np.isfinite(moment_bound)
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
        moment_bound,
        box_count,
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

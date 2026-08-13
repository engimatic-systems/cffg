"""Observation-only initialization for recurrent switching state-space models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import numpy as np


@dataclass(frozen=True)
class RSLDSInitialization:
    """Point parameters and fixed continuous-state prior for the first run."""

    dynamics_matrix: np.ndarray
    dynamics_offset: np.ndarray
    dynamics_covariance: np.ndarray
    recurrence_weights: np.ndarray
    recurrence_offsets: np.ndarray
    emission_matrix: np.ndarray
    emission_offset: np.ndarray
    emission_covariance: np.ndarray
    initial_state_mean: np.ndarray
    initial_state_covariance: np.ndarray
    initial_regime_probability: np.ndarray

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(value).copy()
            for name, value in vars(self).items()
        }


class RSLDSInitializer(Protocol):
    """Observation-only initializer used before an rSLDS engine is compiled."""

    def __call__(
        self,
        observations: np.ndarray,
        *,
        regimes: int,
        latent_dimension: int,
        key: jax.Array,
    ) -> RSLDSInitialization: ...


def _validate_observations(observations: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations)
    if observations.ndim != 2 or observations.shape[0] < 2 or observations.shape[1] < 1:
        raise ValueError("observations must have shape [T,p] with T>=2 and p>=1")
    if not np.issubdtype(observations.dtype, np.inexact):
        raise TypeError("observations must use a floating dtype")
    if not np.all(np.isfinite(observations)):
        raise ValueError("observations must be finite")
    return observations


def _covariance(
    rows: np.ndarray,
    *,
    reference_scale: float | None = None,
) -> np.ndarray:
    rows = np.asarray(rows)
    dimension = rows.shape[1]
    centered = rows - np.mean(rows, axis=0)
    covariance = centered.T @ centered / max(rows.shape[0], 1)
    covariance = 0.5 * (covariance + covariance.T)
    dtype = covariance.dtype
    if reference_scale is None:
        reference_scale = float(np.max(np.abs(covariance)))
    scale = max(float(reference_scale), float(np.finfo(dtype).tiny))
    # The dtype-scaled diagonal protects invertibility of a rank-deficient sample.
    guard = 16.0 * dimension * np.finfo(dtype).eps * scale
    return covariance + np.asarray(guard, dtype=dtype) * np.eye(
        dimension, dtype=dtype
    )


def _canonicalize_components(components: np.ndarray) -> np.ndarray:
    components = components.copy()
    for row in range(components.shape[0]):
        pivot = int(np.argmax(np.abs(components[row])))
        if components[row, pivot] < 0:
            components[row] *= -1
    return components


def _repair_empty_clusters(
    labels: np.ndarray,
    distances: np.ndarray,
    regimes: int,
) -> np.ndarray:
    labels = labels.copy()
    counts = np.bincount(labels, minlength=regimes)
    for empty in np.flatnonzero(counts == 0):
        assigned_distance = distances[np.arange(labels.size), labels]
        candidates = np.argsort(assigned_distance)[::-1]
        moved = next(index for index in candidates if counts[labels[index]] > 1)
        source = labels[moved]
        labels[moved] = empty
        counts[source] -= 1
        counts[empty] += 1
    return labels


def _transition_kmeans(
    states: np.ndarray,
    regimes: int,
    key: jax.Array,
    iterations: int,
) -> np.ndarray:
    features = np.concatenate((states[:-1], states[1:]), axis=1)
    centered = features - np.mean(features, axis=0)
    scale = np.std(centered, axis=0)
    dtype = features.dtype
    threshold = np.sqrt(np.finfo(dtype).eps) * max(
        float(np.max(np.abs(centered))), float(np.finfo(dtype).tiny)
    )
    # Constant coordinates carry no clustering information and use unit scale.
    scale = np.where(scale > threshold, scale, np.asarray(1.0, dtype=dtype))
    normalized = centered / scale
    try:
        indices = np.asarray(
            jax.random.choice(
                key,
                normalized.shape[0],
                shape=(regimes,),
                replace=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError("key must be an explicit JAX PRNG key") from error
    centers = normalized[indices].copy()
    previous = None
    for _ in range(iterations):
        distances = np.sum(
            np.square(normalized[:, None, :] - centers[None, :, :]), axis=2
        )
        labels = _repair_empty_clusters(
            np.argmin(distances, axis=1), distances, regimes
        )
        if previous is not None and np.array_equal(labels, previous):
            break
        previous = labels.copy()
        centers = np.stack(
            [np.mean(normalized[labels == regime], axis=0) for regime in range(regimes)]
        )
    return labels


@dataclass(frozen=True)
class PCAARHMMInitializer:
    """PCA coordinates plus transition-pair clustering and local AR fits.

    PCA fixes an observation-derived latent coordinate system. K-means on
    consecutive latent pairs assigns regime labels, after which regularized
    local autoregressions initialize the switching dynamics. The emission
    covariance uses the probabilistic-PCA isotropic residual variance, because
    the raw PCA residual covariance is singular in the retained directions.
    The explicit JAX key selects the initial cluster centers and is the only
    random input.
    """

    kmeans_iterations: int = 100
    regression_prior_strength: float = 1.0
    transition_pseudocount: float = 1.0

    def __post_init__(self) -> None:
        if self.kmeans_iterations <= 0:
            raise ValueError("kmeans_iterations must be positive")
        if (
            not np.isfinite(self.regression_prior_strength)
            or self.regression_prior_strength <= 0
        ):
            raise ValueError("regression_prior_strength must be finite and positive")
        if (
            not np.isfinite(self.transition_pseudocount)
            or self.transition_pseudocount <= 0
        ):
            raise ValueError("transition_pseudocount must be finite and positive")

    def __call__(
        self,
        observations: np.ndarray,
        *,
        regimes: int,
        latent_dimension: int,
        key: jax.Array,
    ) -> RSLDSInitialization:
        observations = _validate_observations(observations)
        steps, observation_dimension = observations.shape
        if regimes < 2:
            raise ValueError("regimes must be at least two")
        if not 1 <= latent_dimension <= min(steps, observation_dimension):
            raise ValueError(
                "latent_dimension must be positive and no larger than the data rank"
            )
        if steps - 1 < regimes:
            raise ValueError("initializer needs at least one transition per regime")

        observation_mean = np.mean(observations, axis=0)
        centered = observations - observation_mean
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        components = _canonicalize_components(right[:latent_dimension])
        states = centered @ components.T
        labels_tail = _transition_kmeans(
            states, regimes, key, self.kmeans_iterations
        )
        labels = np.concatenate((labels_tail[:1], labels_tail))

        dimension = latent_dimension
        augmented_identity = np.concatenate(
            (
                np.eye(dimension, dtype=states.dtype),
                np.zeros((dimension, 1), dtype=states.dtype),
            ),
            axis=1,
        )
        dynamics_matrix = np.empty((regimes, dimension, dimension), dtype=states.dtype)
        dynamics_offset = np.empty((regimes, dimension), dtype=states.dtype)
        dynamics_covariance = np.empty(
            (regimes, dimension, dimension), dtype=states.dtype
        )
        state_scale = max(
            float(np.max(np.abs(_covariance(states)))),
            float(np.finfo(states.dtype).tiny),
        )
        for regime in range(regimes):
            selected = labels_tail == regime
            design = np.concatenate(
                (
                    states[:-1][selected],
                    np.ones((int(np.sum(selected)), 1), dtype=states.dtype),
                ),
                axis=1,
            )
            targets = states[1:][selected]
            strength = np.asarray(self.regression_prior_strength, dtype=states.dtype)
            precision = design.T @ design + strength * np.eye(
                dimension + 1, dtype=states.dtype
            )
            natural = targets.T @ design + strength * augmented_identity
            coefficients = np.linalg.solve(precision.T, natural.T).T
            dynamics_matrix[regime] = coefficients[:, :-1]
            dynamics_offset[regime] = coefficients[:, -1]
            residuals = targets - design @ coefficients.T
            dynamics_covariance[regime] = _covariance(
                residuals, reference_scale=state_scale
            )

        counts = np.full(
            (regimes, regimes),
            self.transition_pseudocount,
            dtype=states.dtype,
        )
        np.add.at(counts, (labels[:-1], labels[1:]), 1.0)
        transition = counts / np.sum(counts, axis=1, keepdims=True)
        recurrence_offsets = np.empty((regimes, regimes - 1), dtype=states.dtype)
        for source in range(regimes):
            for stick in range(regimes - 1):
                remaining = np.sum(transition[source, stick:])
                accept = transition[source, stick] / remaining
                recurrence_offsets[source, stick] = np.log(accept) - np.log1p(-accept)

        initial_counts = np.full(
            regimes, self.transition_pseudocount, dtype=states.dtype
        )
        initial_counts[labels[0]] += 1.0
        emission_residuals = observations - (
            states @ components + observation_mean
        )
        observation_scale = max(
            float(np.max(np.abs(_covariance(observations)))),
            float(np.finfo(observations.dtype).tiny),
        )
        discarded_dimensions = observation_dimension - latent_dimension
        residual_variance = float(
            np.sum(np.square(emission_residuals))
            / (steps * max(discarded_dimensions, 1))
        )
        # This dtype-scaled guard protects PPCA when no discarded axis exists.
        variance_guard = (
            16.0
            * observation_dimension
            * np.finfo(observations.dtype).eps
            * observation_scale
        )
        emission_covariance = np.asarray(
            max(residual_variance, variance_guard), dtype=observations.dtype
        ) * np.eye(observation_dimension, dtype=observations.dtype)
        return RSLDSInitialization(
            dynamics_matrix=dynamics_matrix,
            dynamics_offset=dynamics_offset,
            dynamics_covariance=dynamics_covariance,
            recurrence_weights=np.zeros(
                (regimes, regimes - 1, dimension), dtype=states.dtype
            ),
            recurrence_offsets=recurrence_offsets,
            emission_matrix=components.T,
            emission_offset=observation_mean,
            emission_covariance=emission_covariance,
            initial_state_mean=states[0],
            initial_state_covariance=_covariance(states),
            initial_regime_probability=initial_counts / np.sum(initial_counts),
        )


def _validate_initialization(
    initialization: RSLDSInitialization,
    *,
    regimes: int,
    latent_dimension: int,
    observation_dimension: int,
) -> RSLDSInitialization:
    if not isinstance(initialization, RSLDSInitialization):
        raise TypeError("initializer must return RSLDSInitialization")
    arrays = initialization.arrays()
    expected = {
        "dynamics_matrix": (regimes, latent_dimension, latent_dimension),
        "dynamics_offset": (regimes, latent_dimension),
        "dynamics_covariance": (regimes, latent_dimension, latent_dimension),
        "recurrence_weights": (regimes, regimes - 1, latent_dimension),
        "recurrence_offsets": (regimes, regimes - 1),
        "emission_matrix": (observation_dimension, latent_dimension),
        "emission_offset": (observation_dimension,),
        "emission_covariance": (observation_dimension, observation_dimension),
        "initial_state_mean": (latent_dimension,),
        "initial_state_covariance": (latent_dimension, latent_dimension),
        "initial_regime_probability": (regimes,),
    }
    actual = {name: value.shape for name, value in arrays.items()}
    if actual != expected:
        raise ValueError(f"initializer shapes {actual} != {expected}")
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("initializer parameters must be finite")
    for name in (
        "dynamics_covariance",
        "emission_covariance",
        "initial_state_covariance",
    ):
        try:
            np.linalg.cholesky(arrays[name])
        except np.linalg.LinAlgError:
            raise ValueError(f"initializer {name} must be positive definite") from None
    probability = arrays["initial_regime_probability"]
    if np.any(probability < 0) or np.sum(probability) <= 0:
        raise ValueError("initial regime probabilities must be nonnegative with positive mass")
    arrays["initial_regime_probability"] = probability / np.sum(probability)
    return RSLDSInitialization(**arrays)

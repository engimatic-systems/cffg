"""Canonical runtime and functional views for observed-only rSLDS fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cffg.learn import Learnables
from cffg.models.coordinate_fit import CoordinateResidual, coordinate_residual
from cffg.models.rslds import (
    INITIAL_REGIME_LEARNABLE,
)

if TYPE_CHECKING:
    from cffg.models.rslds_init import RSLDSInitialization


def legacy_snapshot(
    learning: Learnables,
    points: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Preserve the historical posterior-plus-reporting residual view."""
    return {
        **{
            f"posterior/{name}": np.asarray(value)
            for name, value in learning.state_dict().items()
        },
        **{f"point/{name}": np.asarray(value) for name, value in points.items()},
    }


def legacy_fixed_point_residual(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
) -> float:
    """Reproduce the historical scalar used by the current status contract."""
    if before.keys() != after.keys():
        raise ValueError("fixed-point snapshots disagree on parameter families")
    residual = 0.0
    for name in before:
        left = np.asarray(before[name])
        right = np.asarray(after[name])
        if left.shape != right.shape:
            raise ValueError(f"fixed-point array {name!r} changed shape")
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            return float("inf")
        scale = max(
            1.0,
            float(np.max(np.abs(left), initial=0.0)),
            float(np.max(np.abs(right), initial=0.0)),
        )
        residual = max(
            residual,
            float(np.max(np.abs(right - left), initial=0.0)) / scale,
        )
    return residual


def _normalized_probability_from_log_potential(statistic: np.ndarray) -> np.ndarray:
    statistic = np.asarray(statistic)
    if statistic.ndim != 1 or not np.issubdtype(statistic.dtype, np.inexact):
        raise ValueError("initial regime log potential must be one-dimensional")
    if np.any(np.isnan(statistic)) or np.any(np.isposinf(statistic)):
        raise ValueError("initial regime log potential is invalid")
    finite = np.isfinite(statistic)
    if not np.any(finite):
        raise ValueError("initial regime log potential has empty support")
    shifted = np.where(finite, statistic - np.max(statistic[finite]), -np.inf)
    probability = np.exp(shifted)
    return probability / np.sum(probability)


def _normalized_probability(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability)
    if (
        probability.ndim != 1
        or not np.issubdtype(probability.dtype, np.inexact)
        or not np.all(np.isfinite(probability))
        or np.any(probability < 0.0)
        or np.sum(probability) <= 0.0
    ):
        raise ValueError("initial regime probability is invalid")
    return probability / np.sum(probability)


def _runtime_blocks(
    points: dict[str, np.ndarray],
    initial_probability: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        "initial": {"regime_probability": initial_probability.copy()},
        "dynamics": {
            "matrix": np.asarray(points["dynamics_matrix"]).copy(),
            "offset": np.asarray(points["dynamics_offset"]).copy(),
            "covariance": np.asarray(points["dynamics_covariance"]).copy(),
        },
        "recurrence": {
            "weights": np.asarray(points["recurrence_weights"]).copy(),
            "offsets": np.asarray(points["recurrence_offsets"]).copy(),
        },
        "emission": {
            "matrix": np.asarray(points["emission_matrix"]).copy(),
            "offset": np.asarray(points["emission_offset"]).copy(),
            "covariance": np.asarray(points["emission_covariance"]).copy(),
        },
    }


def initial_runtime_blocks(
    points: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Represent the observation-only initializer as installed model values."""
    return _runtime_blocks(
        points,
        _normalized_probability(points["initial_regime_probability"]),
    )


def learned_runtime_blocks(
    learning: Learnables,
    points: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Represent a learned coordinate using the next solve's Dirichlet potential."""
    initial_statistic = learning.rollover(INITIAL_REGIME_LEARNABLE)
    return _runtime_blocks(
        points,
        _normalized_probability_from_log_potential(initial_statistic),
    )


def observation_probe_states(
    observations: np.ndarray,
    initialization: "RSLDSInitialization",
    *,
    maximum_states: int = 32,
) -> np.ndarray:
    """Construct immutable latent probes from observations and their initializer."""
    observations = np.asarray(observations)
    if maximum_states <= 0:
        raise ValueError("maximum probe states must be positive")
    centered = observations - np.asarray(initialization.emission_offset)
    inverse = np.linalg.pinv(np.asarray(initialization.emission_matrix))
    states = centered @ inverse.T
    if not np.all(np.isfinite(states)):
        raise ValueError("observation-derived transition probes are non-finite")
    indices = np.linspace(
        0,
        len(states) - 1,
        num=min(maximum_states, len(states)),
        dtype=int,
    )
    return states[np.unique(indices)].copy()


def transition_probabilities(
    states: np.ndarray,
    weights: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Evaluate every source row's sequential transition probabilities."""
    states = np.asarray(states)
    weights = np.asarray(weights)
    offsets = np.asarray(offsets)
    if (
        states.ndim != 2
        or weights.ndim != 3
        or offsets.shape != weights.shape[:2]
        or states.shape[1] != weights.shape[2]
        or weights.shape[0] != weights.shape[1] + 1
    ):
        raise ValueError("transition probe shapes are inconsistent")
    if not all(np.all(np.isfinite(value)) for value in (states, weights, offsets)):
        raise ValueError("transition probe inputs must be finite")
    logits = np.einsum("skd,pd->psk", weights, states) + offsets[None, :, :]
    accepts = np.exp(-np.logaddexp(0.0, -logits))
    probabilities = np.empty(
        (states.shape[0], weights.shape[0], weights.shape[0]),
        dtype=np.result_type(states, weights, offsets),
    )
    remaining = np.ones(accepts.shape[:2], dtype=probabilities.dtype)
    for gate in range(weights.shape[1]):
        probabilities[:, :, gate] = remaining * accepts[:, :, gate]
        remaining *= 1.0 - accepts[:, :, gate]
    probabilities[:, :, -1] = remaining
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    return probabilities


def transition_residual(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    probe_states: np.ndarray,
) -> CoordinateResidual:
    """Compare destination probabilities on one fixed truth-free probe."""
    before_probability = transition_probabilities(
        probe_states,
        before["recurrence_weights"],
        before["recurrence_offsets"],
    )
    after_probability = transition_probabilities(
        probe_states,
        after["recurrence_weights"],
        after["recurrence_offsets"],
    )
    return coordinate_residual(
        {"transition": {"probability": before_probability}},
        {"transition": {"probability": after_probability}},
    )

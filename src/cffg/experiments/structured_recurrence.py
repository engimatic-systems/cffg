"""Independent oracle for the structured categorical recurrence objective.

Production uses the D9 structured pair coordinate. This NumPy authority stays
independent of that JAX implementation: it smooths arbitrary log pair
potentials exactly on a chain and checks their junction-tree free energy.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np

from cffg.experiments.recurrence_smoothing import (
    integrated_transition,
    jj_log_transition,
    probability_tolerance,
    recurrence_statistics,
    stick_transition,
)


@dataclasses.dataclass(frozen=True)
class StructuredChain:
    singleton: np.ndarray
    consecutive: np.ndarray
    forward: np.ndarray
    backward: np.ndarray
    to_out: np.ndarray
    to_in: np.ndarray
    log_normalizer: float


def _logsumexp(value, axis=None):
    value = np.asarray(value)
    maximum = np.max(value, axis=axis, keepdims=True)
    valid = np.isfinite(maximum)
    safe_maximum = np.where(valid, maximum, 0.0)
    shifted = np.where(valid, value - safe_maximum, -np.inf)
    total = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    safe_total = np.where(total > 0, total, 1.0)
    result = np.where(
        valid & (total > 0), safe_maximum + np.log(safe_total), -np.inf
    )
    if axis is None:
        return np.asarray(result).reshape(())
    return np.squeeze(result, axis=axis)


def _log_normalize(value, *, axis=-1, name="belief"):
    normalizer = _logsumexp(value, axis=axis)
    if not np.isfinite(normalizer).all():
        raise ValueError(f"{name} has empty support")
    return value - np.expand_dims(normalizer, axis=axis)


def _validate_log_inputs(log_initial, log_evidence, log_transition):
    log_initial = np.asarray(log_initial)
    log_evidence = np.asarray(log_evidence)
    log_transition = np.asarray(log_transition)
    dtype = np.result_type(log_initial, log_evidence, log_transition)
    if not np.issubdtype(dtype, np.floating):
        dtype = np.dtype(np.float64)
    log_initial = log_initial.astype(dtype, copy=False)
    log_evidence = log_evidence.astype(dtype, copy=False)
    log_transition = log_transition.astype(dtype, copy=False)
    if log_initial.ndim != 1 or log_evidence.ndim != 2 or log_transition.ndim != 3:
        raise ValueError("log potentials require [K], [T,K], and [T-1,K,K]")
    time, card = log_evidence.shape
    if card < 2 or log_initial.shape != (card,) or log_transition.shape != (time - 1, card, card):
        raise ValueError("log-potential shapes disagree")
    for name, value in (
        ("log initial", log_initial),
        ("log evidence", log_evidence),
        ("log transition", log_transition),
    ):
        if np.isnan(value).any() or np.isposinf(value).any():
            raise ValueError(f"{name} must be finite or negative infinity")
    return log_initial, log_evidence, log_transition


def smooth_log_chain(log_initial, log_evidence, log_transition) -> StructuredChain:
    """Normalized O(T K^2) forward/backward smoothing in log space."""
    log_initial, log_evidence, log_transition = _validate_log_inputs(
        log_initial, log_evidence, log_transition
    )
    time, card = log_evidence.shape
    forward = np.empty((time, card), dtype=log_transition.dtype)
    backward = np.zeros((time, card), dtype=log_transition.dtype)
    scales = np.empty(time, dtype=log_transition.dtype)

    raw = log_initial + log_evidence[0]
    scales[0] = _logsumexp(raw)
    if not np.isfinite(scales[0]):
        raise ValueError("structured chain has empty support at step 0")
    forward[0] = raw - scales[0]
    for step in range(time - 1):
        predicted = _logsumexp(forward[step, :, None] + log_transition[step], axis=0)
        raw = predicted + log_evidence[step + 1]
        scales[step + 1] = _logsumexp(raw)
        if not np.isfinite(scales[step + 1]):
            raise ValueError(f"structured chain has empty support at step {step + 1}")
        forward[step + 1] = raw - scales[step + 1]

    for step in range(time - 2, -1, -1):
        raw = _logsumexp(
            log_transition[step]
            + log_evidence[step + 1][None, :]
            + backward[step + 1][None, :],
            axis=1,
        )
        backward[step] = _log_normalize(raw, name=f"backward step {step}")

    singleton = np.exp(
        np.stack(
            [
                _log_normalize(
                    forward[step] + backward[step],
                    name=f"singleton step {step}",
                )
                for step in range(time)
            ]
        )
    )
    consecutive = np.empty((time - 1, card, card), dtype=log_transition.dtype)
    to_out = np.empty((time - 1, card), dtype=log_transition.dtype)
    to_in = np.empty((time - 1, card), dtype=log_transition.dtype)
    for step in range(time - 1):
        pair_log = (
            forward[step, :, None]
            + log_transition[step]
            + log_evidence[step + 1][None, :]
            + backward[step + 1][None, :]
        )
        consecutive[step] = np.exp(
            _log_normalize(pair_log.reshape(-1), name=f"pair step {step}").reshape(card, card)
        )
        to_out[step] = np.exp(
            _log_normalize(
                _logsumexp(forward[step, :, None] + log_transition[step], axis=0),
                name=f"message to out step {step}",
            )
        )
        to_in[step] = np.exp(
            _log_normalize(
                _logsumexp(
                    log_transition[step]
                    + log_evidence[step + 1][None, :]
                    + backward[step + 1][None, :],
                    axis=1,
                ),
                name=f"message to in step {step}",
            )
        )
    return StructuredChain(
        singleton,
        consecutive,
        forward,
        backward,
        to_out,
        to_in,
        float(scales.sum()),
    )


def enumerate_log_chain(log_initial, log_evidence, log_transition) -> StructuredChain:
    """Literal path authority for compact K^T cases."""
    log_initial, log_evidence, log_transition = _validate_log_inputs(
        log_initial, log_evidence, log_transition
    )
    time, card = log_evidence.shape
    if card**time > 1_000_000:
        raise ValueError("path enumeration is restricted to at most 1,000,000 paths")
    paths = list(itertools.product(range(card), repeat=time))
    log_weight = np.empty(len(paths), dtype=log_transition.dtype)
    for row, path in enumerate(paths):
        value = log_initial[path[0]] + log_evidence[0, path[0]]
        for step in range(time - 1):
            value += log_transition[step, path[step], path[step + 1]]
            value += log_evidence[step + 1, path[step + 1]]
        log_weight[row] = value
    log_normalizer = _logsumexp(log_weight)
    if not np.isfinite(log_normalizer):
        raise ValueError("enumerated structured chain has empty support")
    probability = np.exp(log_weight - log_normalizer)
    singleton = np.zeros((time, card), dtype=probability.dtype)
    consecutive = np.zeros((time - 1, card, card), dtype=probability.dtype)
    for path, mass in zip(paths, probability, strict=True):
        for step, state in enumerate(path):
            singleton[step, state] += mass
        for step in range(time - 1):
            consecutive[step, path[step], path[step + 1]] += mass
    empty = np.empty((time - 1, card), dtype=probability.dtype)
    return StructuredChain(
        singleton,
        consecutive,
        np.empty_like(singleton),
        np.empty_like(singleton),
        empty,
        empty.copy(),
        float(log_normalizer),
    )


def _entropy(probability):
    probability = np.asarray(probability)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(
            probability > 0, probability * np.log(probability), 0.0
        )
    return float(-np.sum(terms))


def chain_entropy(posterior: StructuredChain) -> float:
    """Junction-tree entropy from pair and separator marginals."""
    pair_entropy = sum(_entropy(pair) for pair in posterior.consecutive)
    separator_entropy = sum(
        _entropy(posterior.singleton[step])
        for step in range(1, len(posterior.singleton) - 1)
    )
    return pair_entropy - separator_entropy


def structured_free_energy(
    posterior: StructuredChain,
    log_initial,
    log_evidence,
    log_transition,
) -> float:
    """Chain VFE using the exact pair/separator entropy identity."""
    log_initial, log_evidence, log_transition = _validate_log_inputs(
        log_initial, log_evidence, log_transition
    )
    expected = np.einsum("i,i->", posterior.singleton[0], log_initial)
    expected += np.einsum("ti,ti->", posterior.singleton, log_evidence)
    expected += np.einsum("tij,tij->", posterior.consecutive, log_transition)
    return float(-chain_entropy(posterior) - expected)


def transition_bethe_energy(posterior: StructuredChain, log_transition) -> float:
    """Time-local pair regions plus categorical separator corrections."""
    log_transition = np.asarray(log_transition)
    node_terms = sum(
        -_entropy(posterior.consecutive[step])
        - np.einsum("ij,ij->", posterior.consecutive[step], log_transition[step])
        for step in range(len(log_transition))
    )
    edge_corrections = sum(
        _entropy(posterior.singleton[step])
        for step in range(1, len(posterior.singleton) - 1)
    )
    return float(node_terms + edge_corrections)


def local_pair_from_messages(left, right, log_transition):
    """One factor belief from its two incoming sum-product cavities."""
    pair_log = np.asarray(left)[:, None] + np.asarray(log_transition) + np.asarray(right)[None, :]
    pair_log = _log_normalize(pair_log.reshape(-1), name="factor-local pair")
    card = np.asarray(left).shape[0]
    return np.exp(pair_log.reshape(card, card))


def structured_recurrence_statistics(
    posterior: StructuredChain,
    means,
    covariances,
    weights,
    offsets,
):
    """Use the same consecutive-state beliefs for all PG learning rows."""
    return tuple(
        recurrence_statistics(
            posterior.consecutive[step],
            means[step],
            covariances[step],
            weights,
            offsets,
        )
        for step in range(len(posterior.consecutive))
    )


def registered_case(card: int, *, dtype=np.float64):
    if card == 2:
        initial = [0.62, 0.38]
        evidence = [[0.9, 0.1], [0.12, 0.88], [0.82, 0.18], [0.15, 0.85], [0.7, 0.3]]
        drives = [[-0.7], [0.25], [0.8], [-0.35]]
        weights = [[[1.35]], [[-1.1]]]
        offsets = [[-0.2], [0.35]]
    elif card == 3:
        initial = [0.5, 0.3, 0.2]
        evidence = [[0.8, 0.15, 0.05], [0.08, 0.84, 0.08], [0.1, 0.15, 0.75], [0.72, 0.18, 0.1]]
        drives = [[-0.4], [0.65], [0.15]]
        weights = [[[1.0], [-0.7]], [[-1.2], [0.8]], [[0.55], [1.1]]]
        offsets = [[-0.3, 0.15], [0.25, -0.45], [-0.1, 0.4]]
    else:
        raise ValueError("registered cases are K=2 and K=3")
    return tuple(
        np.asarray(value, dtype=dtype)
        for value in (initial, evidence, drives, weights, offsets)
    )


def fixed_drive_potentials(card: int, *, dtype=np.float64):
    initial, evidence, drives, weights, offsets = registered_case(card, dtype=dtype)
    transition = np.stack(
        [stick_transition(drive, weights, offsets) for drive in drives]
    )
    return (
        np.log(initial),
        np.log(evidence),
        np.log(transition),
        drives,
        weights,
        offsets,
    )


def uncertain_drive_potentials():
    initial, evidence, drives, weights, offsets = registered_case(2)
    means = drives + np.asarray([[0.15], [-0.1], [0.2], [0.05]])
    covariances = np.asarray([[[0.9]], [[0.55]], [[1.1]], [[0.7]]])
    exact = np.stack(
        [
            integrated_transition(
                means[step], covariances[step], weights, offsets
            )
            for step in range(len(drives))
        ]
    )
    bounded = np.stack(
        [
            jj_log_transition(
                means[step], covariances[step], weights, offsets
            )
            for step in range(len(drives))
        ]
    )
    return np.log(initial), np.log(evidence), np.log(exact), bounded


__all__ = [
    "StructuredChain",
    "chain_entropy",
    "enumerate_log_chain",
    "fixed_drive_potentials",
    "local_pair_from_messages",
    "probability_tolerance",
    "registered_case",
    "smooth_log_chain",
    "structured_free_energy",
    "structured_recurrence_statistics",
    "transition_bethe_energy",
    "uncertain_drive_potentials",
]

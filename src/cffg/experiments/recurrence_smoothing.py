"""Exact categorical-chain oracle for recurrence-link diagnostics.

This module is deliberately outside the runtime inference path.  It provides
an independent NumPy authority for compact recurrent chains and a thin model
builder that exposes the existing CFFG recurrence messages and learning
readout for comparison.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.hermite import hermgauss

from cffg import recurrence
from cffg.compiler import NEG, compile_template
from cffg.core.ir import (
    BlobRef,
    FactorSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.core.serialize import to_dict
from cffg.engine import Engine, RunStatus


@dataclasses.dataclass(frozen=True)
class ChainPosterior:
    singleton: np.ndarray
    consecutive: np.ndarray
    log_normalizer: float


@dataclasses.dataclass(frozen=True)
class ChainMessages:
    to_out: np.ndarray
    to_in: np.ndarray


def probability_tolerance(dtype, cardinality: int) -> float:
    """Named comparison tolerance for normalized categorical quantities."""
    return float(1024 * cardinality * np.finfo(np.dtype(dtype)).eps)


def _probability_rows(value, name: str) -> np.ndarray:
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.floating):
        value = value.astype(np.float64)
    if value.ndim < 1 or value.shape[-1] < 2:
        raise ValueError(f"{name} must have a categorical final axis")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    if np.any(value < 0):
        raise ValueError(f"{name} must be nonnegative")
    sums = value.sum(axis=-1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError(f"{name} must have nonempty support")
    return value / sums


def _chain_inputs(initial, evidence, transitions):
    initial = _probability_rows(initial, "initial")
    evidence = _probability_rows(evidence, "evidence")
    transitions = np.asarray(transitions)
    if not np.issubdtype(transitions.dtype, np.floating):
        transitions = transitions.astype(np.float64)
    if transitions.ndim != 3 or transitions.shape[1] != transitions.shape[2]:
        raise ValueError("transitions must have shape [T-1,K,K]")
    if evidence.ndim != 2 or initial.ndim != 1:
        raise ValueError("initial/evidence must have shapes [K] and [T,K]")
    time, card = evidence.shape
    if initial.shape != (card,) or transitions.shape != (time - 1, card, card):
        raise ValueError("initial, evidence, and transition shapes disagree")
    if not np.isfinite(transitions).all():
        raise ValueError("transitions must be finite")
    if np.any(transitions < 0):
        raise ValueError("transitions must be nonnegative")
    row_sums = transitions.sum(axis=-1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("each transition row must have nonempty support")
    return initial, evidence, transitions / row_sums


def forward_backward(initial, evidence, transitions) -> tuple[ChainPosterior, ChainMessages]:
    """O(T K^2) normalized smoothing and sum-product factor messages."""
    initial, evidence, transitions = _chain_inputs(initial, evidence, transitions)
    time, card = evidence.shape
    dtype = np.result_type(initial, evidence, transitions)
    alpha = np.empty((time, card), dtype=dtype)
    beta = np.ones((time, card), dtype=dtype)
    scales = np.empty(time, dtype=dtype)
    unnormalized = initial * evidence[0]
    scales[0] = unnormalized.sum()
    if scales[0] <= 0:
        raise ValueError("initial evidence has empty support")
    alpha[0] = unnormalized / scales[0]
    for step in range(time - 1):
        unnormalized = (alpha[step] @ transitions[step]) * evidence[step + 1]
        scales[step + 1] = unnormalized.sum()
        if scales[step + 1] <= 0:
            raise ValueError(f"posterior support is empty at step {step + 1}")
        alpha[step + 1] = unnormalized / scales[step + 1]
    for step in range(time - 2, -1, -1):
        beta[step] = transitions[step] @ (evidence[step + 1] * beta[step + 1])
        scale = beta[step].sum()
        if scale <= 0:
            raise ValueError(f"backward support is empty at step {step}")
        beta[step] /= scale
    singleton = alpha * beta
    singleton /= singleton.sum(axis=-1, keepdims=True)
    consecutive = np.empty((time - 1, card, card), dtype=dtype)
    to_out = np.empty((time - 1, card), dtype=dtype)
    to_in = np.empty((time - 1, card), dtype=dtype)
    for step in range(time - 1):
        right = evidence[step + 1] * beta[step + 1]
        pair = alpha[step, :, None] * transitions[step] * right[None, :]
        consecutive[step] = pair / pair.sum()
        to_out[step] = alpha[step] @ transitions[step]
        to_out[step] /= to_out[step].sum()
        to_in[step] = transitions[step] @ right
        to_in[step] /= to_in[step].sum()
    return (
        ChainPosterior(singleton, consecutive, float(np.log(scales).sum())),
        ChainMessages(to_out, to_in),
    )


def enumerate_paths(initial, evidence, transitions) -> ChainPosterior:
    """Literal K^T authority, restricted to compact diagnostic cases."""
    initial, evidence, transitions = _chain_inputs(initial, evidence, transitions)
    time, card = evidence.shape
    if card**time > 1_000_000:
        raise ValueError("path enumeration is restricted to at most 1,000,000 paths")
    weights = []
    paths = []
    for path in itertools.product(range(card), repeat=time):
        weight = initial[path[0]] * evidence[0, path[0]]
        for step in range(time - 1):
            weight *= transitions[step, path[step], path[step + 1]]
            weight *= evidence[step + 1, path[step + 1]]
        paths.append(path)
        weights.append(weight)
    weights = np.asarray(weights, dtype=np.result_type(initial, evidence, transitions))
    normalizer = weights.sum()
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("enumerated posterior has invalid normalization")
    probability = weights / normalizer
    singleton = np.zeros((time, card), dtype=weights.dtype)
    consecutive = np.zeros((time - 1, card, card), dtype=weights.dtype)
    for path, mass in zip(paths, probability, strict=True):
        for step, state in enumerate(path):
            singleton[step, state] += mass
        for step in range(time - 1):
            consecutive[step, path[step], path[step + 1]] += mass
    return ChainPosterior(singleton, consecutive, float(np.log(normalizer)))


def stick_transition(drive, weights, offsets) -> np.ndarray:
    drive = np.asarray(drive)
    weights = np.asarray(weights)
    offsets = np.asarray(offsets)
    if drive.ndim != 1 or weights.ndim != 3:
        raise ValueError("drive/weights must have shapes [D] and [K,K-1,D]")
    card, sticks, dimension = weights.shape
    if drive.shape != (dimension,) or offsets.shape != (card, sticks) or sticks != card - 1:
        raise ValueError("recurrence parameter shapes disagree")
    if not (np.isfinite(drive).all() and np.isfinite(weights).all() and np.isfinite(offsets).all()):
        raise ValueError("recurrence parameters and drive must be finite")
    logits = np.einsum("ild,d->il", weights, drive) + offsets
    accepted = np.exp(-np.logaddexp(0.0, -logits))
    transition = np.empty((card, card), dtype=np.result_type(drive, weights, offsets))
    remaining = np.ones(card, dtype=transition.dtype)
    for stick in range(sticks):
        transition[:, stick] = remaining * accepted[:, stick]
        remaining *= 1.0 - accepted[:, stick]
    transition[:, -1] = remaining
    return transition


def jj_log_transition(mean, covariance, weights, offsets) -> np.ndarray:
    """Independent JJ expected log-potential for one Gaussian drive."""
    mean = np.asarray(mean)
    covariance = np.asarray(covariance)
    weights = np.asarray(weights)
    offsets = np.asarray(offsets)
    card, sticks, dimension = weights.shape
    if mean.shape != (dimension,) or covariance.shape != (dimension, dimension):
        raise ValueError("drive moment shapes disagree")
    if offsets.shape != (card, sticks) or sticks != card - 1:
        raise ValueError("recurrence parameter shapes disagree")
    values = (mean, covariance, weights, offsets)
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("JJ inputs must be finite")
    logit_mean = np.einsum("ild,d->il", weights, mean) + offsets
    variance = np.einsum("ild,de,ile->il", weights, covariance, weights)
    tolerance = 32 * dimension * np.finfo(np.result_type(*values)).eps
    if np.any(variance < -tolerance):
        raise ValueError("drive covariance produces negative logit variance")
    xi = np.sqrt(np.maximum(np.square(logit_mean) + variance, 0.0))
    common = -np.logaddexp(0.0, -xi) - 0.5 * xi
    success = common + 0.5 * logit_mean
    failure = common - 0.5 * logit_mean
    prefix = np.concatenate(
        [np.zeros((card, 1), dtype=common.dtype), np.cumsum(failure, axis=-1)],
        axis=-1,
    )
    return np.concatenate([prefix[:, :-1] + success, prefix[:, -1:]], axis=-1)


def integrated_transition(mean, covariance, weights, offsets, order: int = 80) -> np.ndarray:
    """Gauss-Hermite integration for the scalar uncertain-drive oracle."""
    mean = np.asarray(mean)
    covariance = np.asarray(covariance)
    if mean.shape != (1,) or covariance.shape != (1, 1):
        raise ValueError("the independent quadrature oracle supports D=1")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all() or covariance[0, 0] < 0:
        raise ValueError("uncertain drive moments must be finite with nonnegative variance")
    nodes, weights_gh = hermgauss(order)
    points = mean[0] + np.sqrt(2.0 * covariance[0, 0]) * nodes
    values = np.stack(
        [stick_transition(np.asarray([point]), weights, offsets) for point in points]
    )
    return np.einsum("n,nij->ij", weights_gh / np.sqrt(np.pi), values)


def mean_field_chain(initial, evidence, log_potentials, *, max_iterations=10_000):
    """Coordinate-optimal fully factorized categorical chain posterior."""
    initial, evidence, _ = _chain_inputs(initial, evidence, np.exp(log_potentials))
    log_potentials = np.asarray(log_potentials)
    if not np.isfinite(log_potentials).all():
        raise ValueError("log potentials must be finite")
    time, card = evidence.shape
    dtype = np.result_type(initial, evidence, log_potentials)
    tiny = np.finfo(dtype).tiny
    unary = np.log(np.maximum(evidence, tiny))
    unary[0] += np.log(np.maximum(initial, tiny))
    q = _probability_rows(np.ones_like(evidence), "initial mean field")
    tolerance = probability_tolerance(dtype, card)
    for iteration in range(1, max_iterations + 1):
        before = q.copy()
        for step in range(time):
            score = unary[step].copy()
            if step:
                score += q[step - 1] @ log_potentials[step - 1]
            if step + 1 < time:
                score += log_potentials[step] @ q[step + 1]
            score -= score.max()
            q[step] = np.exp(score)
            q[step] /= q[step].sum()
        if np.max(np.abs(q - before)) <= tolerance:
            return q, iteration
    raise RuntimeError("mean-field categorical chain exhausted its iteration budget")


def recurrence_statistics(responsibility, mean, covariance, weights, offsets):
    """Independent PG learning statistics from a joint responsibility."""
    responsibility = _probability_rows(np.asarray(responsibility).reshape(1, -1), "responsibility").reshape(
        np.asarray(responsibility).shape
    )
    card = responsibility.shape[0]
    if responsibility.shape != (card, card):
        raise ValueError("responsibility must have shape [K,K]")
    mean = np.asarray(mean)
    covariance = np.asarray(covariance)
    dimension = mean.shape[0]
    feature_mean = np.concatenate([mean, np.ones(1, dtype=mean.dtype)])
    feature_second = np.empty((dimension + 1, dimension + 1), dtype=np.result_type(mean, covariance))
    feature_second[:-1, :-1] = covariance + np.outer(mean, mean)
    feature_second[:-1, -1] = mean
    feature_second[-1, :-1] = mean
    feature_second[-1, -1] = 1.0
    coefficients = np.concatenate([weights, offsets[..., None]], axis=-1)
    xi = np.sqrt(np.maximum(np.einsum("ilp,pq,ilq->il", coefficients, feature_second, coefficients), 0.0))
    threshold = np.sqrt(np.finfo(xi.dtype).eps)
    omega = np.where(
        xi > threshold,
        np.tanh(0.5 * xi) / (2.0 * np.where(xi > threshold, xi, 1.0)),
        0.25 - np.square(xi) / 48.0 + np.square(np.square(xi)) / 480.0,
    )
    at_risk = np.flip(np.cumsum(np.flip(responsibility, axis=1), axis=1), axis=1)[:, :-1]
    kappa = responsibility[:, :-1] - 0.5 * at_risk
    return {
        "omega": omega,
        "exposure": at_risk,
        "linear": kappa[..., None] * feature_mean,
        "quadratic": at_risk[..., None, None] * omega[..., None, None] * feature_second,
    }


def _blob(name: str) -> BlobRef:
    return BlobRef(name, f"blobs/{name}.npy")


def recurrence_chain_template(initial, evidence, drives, weights, offsets, *, dtype):
    """Build a CFFG chain whose Gaussian drives are tightly observed."""
    initial = np.asarray(initial, dtype=dtype)
    evidence = np.asarray(evidence, dtype=dtype)
    drives = np.asarray(drives, dtype=dtype)
    weights = np.asarray(weights, dtype=dtype)
    offsets = np.asarray(offsets, dtype=dtype)
    time, card = evidence.shape
    dimension = drives.shape[1]
    template = Template(name=f"recurrence-smoothing-k{card}-t{time}")
    arrays: dict[str, np.ndarray] = {
        "weights": weights,
        "offsets": offsets,
        "observation_matrix": np.eye(dimension, dtype=dtype),
        "observation_offset": np.zeros(dimension, dtype=dtype),
        "observation_covariance": np.eye(dimension, dtype=dtype)
        * np.asarray(64 * np.finfo(np.dtype(dtype)).eps, dtype=dtype),
    }
    for name in arrays:
        template.blobs[name] = _blob(name)
    for step in range(time):
        if step == 0:
            edge = "z0"
            template.variables[edge] = VariableSpec(edge, cardinality=card)
            statistic = initial * evidence[0]
        elif step == time - 1:
            edge = f"z{step}"
            template.variables[edge] = VariableSpec(edge, cardinality=card)
            statistic = evidence[step]
        else:
            left, right, observed = (f"z{step}_{role}" for role in ("left", "right", "observed"))
            for edge_name in (left, right, observed):
                template.variables[edge_name] = VariableSpec(edge_name, cardinality=card)
            template.factors[f"equal{step}"] = FactorSpec(
                f"equal{step}", "equality", edges={"a": left, "b": right, "c": observed}
            )
            edge = observed
            statistic = evidence[step]
        statistic = statistic / statistic.sum()
        name = f"state_probability_{step}"
        arrays[name] = np.asarray(statistic, dtype=dtype)
        template.blobs[name] = _blob(name)
        template.factors[f"state{step}"] = FactorSpec(
            f"state{step}", "categorical", edges={"out": edge}, param_blobs={"statistic": name}
        )
    for step in range(time - 1):
        in_edge = "z0" if step == 0 else f"z{step}_right"
        out_edge = f"z{step + 1}" if step + 1 == time - 1 else f"z{step + 1}_left"
        drive_edge, observation_edge = f"drive{step}", f"observed_drive{step}"
        template.variables[drive_edge] = VariableSpec(drive_edge, dim=dimension)
        template.variables[observation_edge] = VariableSpec(observation_edge, dim=dimension)
        template.factors[f"switch{step}"] = FactorSpec(
            f"switch{step}",
            "recurrence_link",
            edges={"out": out_edge, "in": in_edge, "drive": drive_edge},
            param_blobs={"weights": "weights", "offsets": "offsets"},
        )
        template.factors[f"observe_drive{step}"] = FactorSpec(
            f"observe_drive{step}",
            "gaussian_linear",
            edges={"out": observation_edge, "in": drive_edge},
            param_blobs={
                "matrix": "observation_matrix",
                "offset": "observation_offset",
                "cov": "observation_covariance",
            },
        )
        template.ports[f"drive{step}"] = PortSpec(f"drive{step}", observation_edge, PortRole.DATA)
    return template, arrays, {f"drive{step}": drives[step] for step in range(time - 1)}


def _normalized_log_message(row):
    row = np.asarray(row)
    support = row != np.asarray(NEG, dtype=row.dtype)
    if not np.any(support) or not np.isfinite(row[support]).all():
        raise ValueError("categorical message is improper")
    shifted = np.where(support, row - np.max(row[support]), -np.inf)
    probability = np.where(support, np.exp(shifted), 0.0)
    return probability / probability.sum()


def run_cffg_chain(initial, evidence, drives, weights, offsets, *, dtype=np.float64):
    """Run one compact engine chain and expose every comparison surface."""
    template, arrays, observations = recurrence_chain_template(
        initial, evidence, drives, weights, offsets, dtype=dtype
    )
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.bind_data(engine.init_state(), observations)
    state = jax.tree_util.tree_map(
        lambda value: value.astype(jnp.dtype(dtype)) if jnp.issubdtype(value.dtype, jnp.inexact) else value,
        state,
    )
    tolerance = probability_tolerance(dtype, np.asarray(initial).shape[0])
    result = engine.run(state, 1000, tol=tolerance / 8)
    if result.status is not RunStatus.CONVERGED:
        raise RuntimeError(f"CFFG recurrence chain ended {result.status.value}")
    state = result.state
    marginal_rows = engine.marginals(state)
    time = np.asarray(evidence).shape[0]
    singleton = np.stack(
        [marginal_rows["z0" if step == 0 else f"z{step}" if step == time - 1 else f"z{step}_left"] for step in range(time)]
    )
    beliefs = engine.gaussian_node_stats(state)
    learning = engine.recurrence_weight_stats(state)
    factor_local = []
    factor_messages_out, factor_messages_in = [], []
    coordinate_out, coordinate_in = [], []
    product_stats = []
    actual_drive_mean, actual_drive_covariance = [], []
    for group in engine.arena.recurrence_groups:
        card = group.cards[0]
        for member_index, name in enumerate(group.members):
            row = beliefs[name]
            transition = stick_transition(row["drive_mean"], weights, offsets)
            cavities = []
            own_messages = []
            for role in range(2):
                slots = list(group.d_gather_both[member_index, role])
                own_slot = int(group.d_scatter[member_index, role])
                cavity_slot = next(int(slot) for slot in slots if int(slot) != own_slot)
                cavity = state.messages[engine.arena.message_indices(cavity_slot, card)]
                own = state.messages[engine.arena.message_indices(own_slot, card)]
                cavities.append(_normalized_log_message(cavity))
                own_messages.append(_normalized_log_message(own))
            local = cavities[1][:, None] * transition * cavities[0][None, :]
            factor_local.append(local / local.sum())
            factor_messages_out.append(own_messages[0])
            factor_messages_in.append(own_messages[1])
            score = jj_log_transition(
                row["drive_mean"], row["drive_covariance"], weights, offsets
            )
            out_score = row["in_probability"] @ score
            in_score = score @ row["out_probability"]
            coordinate_out.append(_normalized_log_message(out_score))
            coordinate_in.append(_normalized_log_message(in_score))
            product_joint = np.outer(row["in_probability"], row["out_probability"])
            product_stats.append(
                recurrence_statistics(
                    product_joint,
                    row["drive_mean"],
                    row["drive_covariance"],
                    weights,
                    offsets,
                )
            )
            actual_drive_mean.append(row["drive_mean"])
            actual_drive_covariance.append(row["drive_covariance"])
    order = np.argsort([int(name.removeprefix("switch")) for name in sum((list(group.members) for group in engine.arena.recurrence_groups), [])])
    def ordered(values):
        return np.asarray(values)[order]
    return {
        "template": template,
        "status": result.status.value,
        "iterations": result.iterations,
        "singleton": singleton,
        "factor_local": ordered(factor_local),
        "messages": ChainMessages(ordered(factor_messages_out), ordered(factor_messages_in)),
        "coordinate_messages": ChainMessages(ordered(coordinate_out), ordered(coordinate_in)),
        "learning": [learning[f"switch{step}"] for step in range(time - 1)],
        "independent_product_learning": [product_stats[index] for index in order],
        "drive_mean": ordered(actual_drive_mean),
        "drive_covariance": ordered(actual_drive_covariance),
    }


def _case(card: int, *, dtype=np.float64):
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
        raise ValueError("registered compact cases are K=2 and K=3")
    return tuple(np.asarray(value, dtype=dtype) for value in (initial, evidence, drives, weights, offsets))


def _max_error(left, right):
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _stats_error(left, right):
    return max(_max_error(left[key], right[key]) for key in ("omega", "exposure", "linear", "quadratic"))


def diagnose_case(card: int, *, dtype=np.float64):
    initial, evidence, drives, weights, offsets = _case(card, dtype=dtype)
    transitions = np.stack([stick_transition(drive, weights, offsets) for drive in drives])
    exact, exact_messages = forward_backward(initial, evidence, transitions)
    enumerated = enumerate_paths(initial, evidence, transitions)
    cffg = run_cffg_chain(initial, evidence, drives, weights, offsets, dtype=dtype)
    structured_learning = [
        recurrence_statistics(
            exact.consecutive[step],
            cffg["drive_mean"][step],
            cffg["drive_covariance"][step],
            weights,
            offsets,
        )
        for step in range(len(drives))
    ]
    readout_error = max(
        _stats_error(cffg["learning"][step], cffg["independent_product_learning"][step])
        for step in range(len(drives))
    )
    structured_error = max(
        _stats_error(cffg["learning"][step], structured_learning[step])
        for step in range(len(drives))
    )
    product = cffg["singleton"][:-1, :, None] * cffg["singleton"][1:, None, :]
    return {
        "cardinality": card,
        "time": evidence.shape[0],
        "dtype": np.dtype(dtype).name,
        "tolerance": probability_tolerance(dtype, card),
        "enumerator_vs_forward_backward": max(
            _max_error(enumerated.singleton, exact.singleton),
            _max_error(enumerated.consecutive, exact.consecutive),
        ),
        "cffg_status": cffg["status"],
        "cffg_iterations": cffg["iterations"],
        "local_coordinate_message_error": max(
            _max_error(cffg["messages"].to_out, cffg["coordinate_messages"].to_out),
            _max_error(cffg["messages"].to_in, cffg["coordinate_messages"].to_in),
        ),
        "sum_product_message_error": max(
            _max_error(cffg["messages"].to_out, exact_messages.to_out),
            _max_error(cffg["messages"].to_in, exact_messages.to_in),
        ),
        "singleton_error": _max_error(cffg["singleton"], exact.singleton),
        "factor_local_joint_error": _max_error(cffg["factor_local"], exact.consecutive),
        "product_joint_error": _max_error(product, exact.consecutive),
        "product_joint_correlated_gap": float(
            np.max(np.abs(exact.consecutive - exact.singleton[:-1, :, None] * exact.singleton[1:, None, :]))
        ),
        "learning_readout_error": readout_error,
        "learning_structured_responsibility_error": structured_error,
        "drive_clamp_error": _max_error(cffg["drive_mean"], drives),
        "surfaces": {
            "exact_singleton": exact.singleton.tolist(),
            "exact_consecutive": exact.consecutive.tolist(),
            "exact_sum_product_to_out": exact_messages.to_out.tolist(),
            "exact_sum_product_to_in": exact_messages.to_in.tolist(),
            "cffg_singleton": cffg["singleton"].tolist(),
            "cffg_factor_local": cffg["factor_local"].tolist(),
            "cffg_product_consecutive": product.tolist(),
            "cffg_to_out": cffg["messages"].to_out.tolist(),
            "cffg_to_in": cffg["messages"].to_in.tolist(),
            "independent_coordinate_to_out": cffg["coordinate_messages"].to_out.tolist(),
            "independent_coordinate_to_in": cffg["coordinate_messages"].to_in.tolist(),
        },
        "template": to_dict(cffg["template"]),
        "parameters": {
            "initial": initial.tolist(),
            "evidence": evidence.tolist(),
            "drives": drives.tolist(),
            "weights": weights.tolist(),
            "offsets": offsets.tolist(),
        },
    }


def diagnose_uncertain_drive():
    dtype = np.float64
    initial, evidence, drives, weights, offsets = _case(2, dtype=dtype)
    means = drives + np.asarray([[0.15], [-0.1], [0.2], [0.05]])
    covariances = np.asarray([[[0.9]], [[0.55]], [[1.1]], [[0.7]]])
    exact_potentials = np.stack(
        [integrated_transition(means[step], covariances[step], weights, offsets) for step in range(len(drives))]
    )
    jj_log = np.stack(
        [jj_log_transition(means[step], covariances[step], weights, offsets) for step in range(len(drives))]
    )
    exact, _ = forward_backward(initial, evidence, exact_potentials)
    jj_structured, _ = forward_backward(initial, evidence, np.exp(jj_log))
    factorized, iterations = mean_field_chain(initial, evidence, jj_log)
    factorized_pair = factorized[:-1, :, None] * factorized[1:, None, :]
    return {
        "quadrature_order": 80,
        "dtype": np.dtype(dtype).name,
        "tolerance": probability_tolerance(dtype, 2),
        "pg_jj_singleton_error": _max_error(jj_structured.singleton, exact.singleton),
        "pg_jj_pair_error": _max_error(jj_structured.consecutive, exact.consecutive),
        "categorical_factorization_singleton_error": _max_error(factorized, jj_structured.singleton),
        "categorical_factorization_pair_error": _max_error(factorized_pair, jj_structured.consecutive),
        "mean_field_iterations": iterations,
        "means": means.tolist(),
        "covariances": covariances.tolist(),
    }


def diagnose_hostile_inputs():
    initial = np.asarray([1.0, 0.0])
    evidence = np.asarray([[1.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
    transitions = np.asarray(
        [[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]
    )
    deterministic, _ = forward_backward(initial, evidence, transitions)

    card, time = 8, 100
    scale_initial = np.arange(1, card + 1, dtype=np.float64)
    scale_evidence = 1.0 + (
        np.arange(time)[:, None] + 2 * np.arange(card)[None, :]
    ) % 11
    row = 1.0 + (
        np.arange(card)[:, None] + 3 * np.arange(card)[None, :]
    ) % 13
    scale_transitions = np.broadcast_to(row, (time - 1, card, card)).copy()
    scaled, _ = forward_backward(scale_initial, scale_evidence, scale_transitions)

    weights = np.asarray([[[0.5]], [[-0.4]]])
    offsets = np.zeros((2, 1))
    vacuous = recurrence.update(
        jnp.full((1, 2, 2), -np.log(2.0)),
        jnp.full((1, 2, 2), -np.log(2.0)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.asarray(weights[None]),
        jnp.asarray(offsets[None]),
    )

    rejected = []
    invalid_calls = (
        lambda: forward_backward([np.nan, 1.0], [[1.0, 1.0], [1.0, 1.0]], [[[0.5, 0.5], [0.5, 0.5]]]),
        lambda: stick_transition([np.inf], weights, offsets),
        lambda: integrated_transition([0.0], [[-1.0]], weights, offsets),
    )
    for call in invalid_calls:
        try:
            call()
        except ValueError as error:
            rejected.append(str(error))
    return {
        "exact_zero_and_deterministic_support": deterministic.singleton.tolist(),
        "vacuous_message_status": int(vacuous.status[0]),
        "vacuous_message_to_out": np.asarray(vacuous.to_out[0]).tolist(),
        "nonfinite_or_improper_rejections": rejected,
        "scale_case": {
            "cardinality": card,
            "time": time,
            "algorithm": "forward_backward_O(T*K^2)",
            "singleton_sum_error": float(np.max(np.abs(scaled.singleton.sum(axis=1) - 1.0))),
            "pair_sum_error": float(np.max(np.abs(scaled.consecutive.sum(axis=(1, 2)) - 1.0))),
            "path_enumeration_used": False,
        },
    }


def diagnostic_report(repo: Path) -> dict:
    cases = [diagnose_case(2), diagnose_case(3), diagnose_case(2, dtype=np.float32)]
    local_failed = any(
        case["local_coordinate_message_error"] > case["tolerance"] for case in cases
    )
    readout_failed = any(
        case["learning_readout_error"] > case["tolerance"] for case in cases
    )
    if local_failed:
        ownership = "local_categorical_message_defect"
    elif readout_failed:
        ownership = "learning_readout_defect"
    else:
        ownership = "variational_family_limitation"
    code_identity = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    source = Path(__file__).read_bytes()
    return {
        "schema_version": 1,
        "ticket": "cff-nbo9",
        "code_identity": code_identity,
        "oracle_sha256": hashlib.sha256(source).hexdigest(),
        "numerical_mode": {"jax_enable_x64": bool(jax.config.jax_enable_x64)},
        "tolerance_rule": "1024 * K * finfo(dtype).eps",
        "replay_command": "uv run python tools/recurrence_smoothing_diagnostic.py --output artifacts/cff-nbo9-exact-smoothing.json",
        "cases": cases,
        "uncertain_drive": diagnose_uncertain_drive(),
        "hostile_inputs": diagnose_hostile_inputs(),
        "ownership": {
            "decision": ownership,
            "falsifiers": {
                "local_message": "local_coordinate_message_error exceeds the named dtype tolerance",
                "learning_readout": "learning_readout_error exceeds the named dtype tolerance",
                "variational_family": "both implementation checks pass while exact pair/product and smoothing gaps exceed tolerance",
            },
            "production_change": (
                "A structured consecutive-regime belief would change the recurrence fixed point and BFE; "
                "it requires a new derived region/counting-number treatment. Preserving the current "
                "mean-field fixed point and BFE cannot make product responsibilities equal the exact pair posterior."
            ),
        },
    }


def write_report(output: Path, repo: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(diagnostic_report(repo), indent=2, sort_keys=True) + "\n")

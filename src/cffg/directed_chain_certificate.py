"""Host certificate for a structurally recognized Taylor Gaussian chain.

A flooding sweep reads *only* the supplied cavities. Its local factors and
interior-edge corrections give the frozen Bethe ledger even away from a root.
Repeating that calculation on the raw successor reselects every Taylor center.
Only the evaluation of the physical mean and its Jacobian is supplied by the
directed executor.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


class ChainCertificate(NamedTuple):
    residual: float
    free_energy: float
    successor_free_energy: float
    passed: bool


_LOG2PI = math.log(2 * math.pi)


def _transpose(matrix):
    return matrix.swapaxes(-1, -2)


def _symmetric(matrix):
    return (matrix + _transpose(matrix)) * 0.5


def _finite(value):
    if not np.isfinite(value).all():
        raise FloatingPointError("nonfinite Gaussian chain calculation")


def _check_symmetry(matrix):
    _finite(matrix)
    info = np.finfo(matrix.dtype)
    scale = np.maximum(np.abs(matrix).max(axis=(-2, -1)), info.tiny)
    # Same relative symmetry test as gauss._symmetry_ok, before symmetrizing.
    tolerance = 16 * matrix.shape[-1] * info.eps * scale
    if np.any(np.abs(matrix - _transpose(matrix)).max(axis=(-2, -1)) > tolerance):
        raise np.linalg.LinAlgError("asymmetric Gaussian matrix")


def _pd_logdet(matrix):
    _check_symmetry(matrix)
    symmetric = _symmetric(matrix)
    chol = np.linalg.cholesky(symmetric)
    diagonal = chol.diagonal(axis1=-2, axis2=-1)
    sign, logdet = np.linalg.slogdet(symmetric)
    _finite(chol)
    _finite(logdet)
    if np.any(sign <= 0) or np.any(diagonal <= 0):
        raise np.linalg.LinAlgError("improper Gaussian matrix")
    # LAPACK and XLA can round the final pivot differently after cancellation.
    # Require it to be resolved relative to its own diagonal in the state dtype.
    variance = symmetric.diagonal(axis1=-2, axis2=-1)
    info = np.finfo(matrix.dtype)
    if (np.any(variance < info.tiny)
            or np.any(diagonal <= np.sqrt(variance) * math.sqrt(8 * matrix.shape[-1] * info.eps))):
        raise np.linalg.LinAlgError("unresolved Gaussian pivot")
    return logdet


def _apply(matrix, vector):
    return (matrix @ vector[..., None])[..., 0]


def _solve(matrix, vector):
    return np.linalg.solve(matrix, vector[..., None])[..., 0]


def _inverse(matrix):
    return _symmetric(np.linalg.inv(matrix))


def _negative_entropy(precision):
    return 0.5 * (_pd_logdet(precision) - precision.shape[-1] * (1 + _LOG2PI))


def _normal_cost(residual, residual_covariance, precision, logdet_precision):
    """Expectation of minus log N(residual; 0, W^-1), including its constant."""
    quadratic = (residual * _apply(precision, residual)).sum(axis=-1)
    trace = np.einsum("...ij,...ji->...", precision, residual_covariance)
    return 0.5 * (residual.shape[-1] * _LOG2PI - logdet_precision + quadratic + trace)


def _prior(incoming, params):
    ((eta, precision),) = incoming
    mean, w = params["m"], params["W"]
    logdet_w = _pd_logdet(w)
    site = _apply(w, mean), w
    belief = precision + w
    energy = _negative_entropy(belief)
    residual = _solve(belief, eta + site[0]) - mean
    energy += _normal_cost(residual, _inverse(belief), w, logdet_w)
    return (site,), energy


def _equality(incoming):
    # Exclude each recipient before summing; total-minus-self loses weak legs.
    outgoing = []
    for i in range(len(incoming)):
        others = incoming[:i] + incoming[i + 1:]
        outgoing.append(tuple(np.sum(parts, axis=0) for parts in zip(*others, strict=True)))
    belief_precision = sum(p for _, p in incoming)
    return outgoing, _negative_entropy(belief_precision)


def _prediction(mean, covariance, matrix, offset, noise):
    predicted_covariance = noise + matrix @ covariance @ _transpose(matrix)
    _pd_logdet(predicted_covariance)
    predicted_precision = _inverse(predicted_covariance)
    prediction = _apply(matrix, mean) + offset
    return _apply(predicted_precision, prediction), predicted_precision


def _conditioned(incoming, params, observation, mask):
    _, (eta, precision) = incoming
    matrix, offset, w = params["A"], params["b"], params["W"]
    logdet_w = _pd_logdet(w)
    _pd_logdet(precision)
    forward = _prediction(_solve(precision, eta), _inverse(precision), matrix, offset, _inverse(w))
    back_precision = (_transpose(matrix) @ w @ matrix) * mask[:, None, None]
    back_eta = _apply(_transpose(matrix), _apply(w, observation - offset)) * mask[:, None]
    belief = precision + back_precision
    energy = _negative_entropy(belief)
    residual = observation - _apply(matrix, _solve(belief, eta + back_eta)) - offset
    covariance = matrix @ _inverse(belief) @ _transpose(matrix)
    energy += mask * _normal_cost(residual, covariance, w, logdet_w)
    return (forward, (back_eta, back_precision)), energy


def _transition(incoming, params, members, linearize):
    (eo, lo), (ei, li) = incoming
    _pd_logdet(li)
    _check_symmetry(lo)
    _pd_logdet(params["W"])
    # The registered binding first solves its packed W for the noise. D13
    # tests that unmodified noise before forming its symmetric inverse.
    noise = np.linalg.inv(params["W"])
    logdet_noise = _pd_logdet(noise)
    w = _inverse(noise)
    center = _solve(li, ei)
    evaluations = [linearize(member, x, theta) for member, x, theta in
                   zip(members, center, params["theta"], strict=True)]
    matrix, value = (np.stack(part) for part in zip(*evaluations, strict=True))
    if matrix.shape != (len(members), eo.shape[-1], ei.shape[-1]) or value.shape != eo.shape:
        raise FloatingPointError("Taylor evaluator returned an incompatible shape")
    if matrix.dtype != ei.dtype or value.dtype != ei.dtype:
        raise FloatingPointError("Taylor evaluator changed the chain arithmetic dtype")
    _finite(matrix)
    _finite(value)
    offset = value - _apply(matrix, center)
    forward = _prediction(center, _inverse(li), matrix, offset, noise)

    # Check the actual joint canonical block, including cancellation at the
    # state dtype. Positivity of a covariance-form prediction is insufficient.
    atw = _transpose(matrix) @ w
    system = w + lo
    _pd_logdet(system)
    joint = np.concatenate((
        np.concatenate((li + atw @ matrix, -atw), axis=-1),
        np.concatenate((-w @ matrix, system), axis=-1),
    ), axis=-2)
    energy = _negative_entropy(joint)
    joint_eta = np.concatenate((ei - _apply(atw, offset), eo + _apply(w, offset)), axis=-1)
    mean = _solve(joint, joint_eta)
    covariance = _inverse(joint)
    di, do = ei.shape[-1], eo.shape[-1]
    _pd_logdet(covariance[:, :di, :di])
    _pd_logdet(covariance[:, di:, di:])
    identity = np.broadcast_to(np.eye(do, dtype=ei.dtype), w.shape)
    transform = np.concatenate((-matrix, identity), axis=-1)
    residual = _apply(transform, mean) - offset
    residual_covariance = transform @ covariance @ _transpose(transform)
    energy += _normal_cost(residual, residual_covariance, w, -logdet_noise)

    # Eliminate the output coordinate directly. Subtracting the input cavity
    # from its marginal can destroy the whole backward site in float32.
    effective = _symmetric(w @ np.linalg.solve(system, lo))
    back_precision = _symmetric(_transpose(matrix) @ effective @ matrix)
    back_eta = _apply(atw, _solve(system, eo - _apply(lo, offset)))
    return (forward, (back_eta, back_precision)), energy


def _sweep(arena, eta, precision, obs, mask, params, linearize):
    next_eta, next_precision = eta.copy(), precision.copy()
    energy = eta.dtype.type(0)
    for group in arena.gauss_groups:
        gather, scatter = np.asarray(group.gather), np.asarray(group.scatter)
        incoming = [
            (eta[arena.g_eta_indices(slots, dim)], precision[arena.g_lam_indices(slots, dim)])
            for slots, dim in zip(gather.T, group.dims, strict=True)
        ]
        values = {key: params[path] for key, path in group.parameter_paths}
        if group.type == "equality" and group.kernel is None:
            outgoing, energies = _equality(incoming)
        elif group.type == "gaussian":
            outgoing, energies = _prior(incoming, values)
        elif group.type == "gaussian_linear_obs" and group.observed_role == 0:
            rows = np.asarray(group.obs_rows)
            observation = obs[arena.g_obs_indices(rows, group.observed_dim)]
            outgoing, energies = _conditioned(incoming, values, observation, mask[rows])
        elif (group.type == "gaussian_nonlinear" and group.observed_role is None
              and group.kernel.binding.objective == "linearization_frozen_gaussian_bethe"):
            outgoing, energies = _transition(incoming, values, group.members, linearize)
        else:
            raise FloatingPointError(f"unsupported directed-chain row {group.name!r}")
        for slot, dim, (e, p) in zip(scatter.T, group.dims, outgoing, strict=True):
            next_eta[arena.g_eta_indices(slot, dim)] = e
            next_precision[arena.g_lam_indices(slot, dim)] = p
        energy += energies.sum()
    for dim, pairs in arena.gauss_interior:
        left, right = np.asarray(pairs).T
        belief = precision[arena.g_lam_indices(left, dim)] + precision[arena.g_lam_indices(right, dim)]
        energy -= _negative_entropy(belief).sum()
    for value in (next_eta, next_precision, energy):
        _finite(value)
    return next_eta, next_precision, float(energy)


def certify(arena, state, linearize, evidence_cost, tolerance) -> ChainCertificate:
    """Check the raw synchronous message root and both frozen Bethe ledgers.

    The caller establishes the directed topology and built-in factor identities.
    This numeric path supports a proper prior, Gaussian equalities, unobserved
    Taylor transitions and linear conditioned outputs with binary DATA masks.
    External slots must be vacuous. Invalid inputs, unresolved positive-definite
    matrices or unsupported rows raise FloatingPointError or LinAlgError for
    fallback. A finite disagreement returns the measured values and passed=False.
    """
    if arena.n_slots or arena.n_efslots or not arena.gauss_groups or arena.semiring != "sum":
        raise FloatingPointError("certificate requires a sum-product Gaussian chain")
    if not math.isfinite(evidence_cost) or not math.isfinite(tolerance) or tolerance < 0:
        raise FloatingPointError("invalid evidence cost or tolerance")
    eta, precision, obs, mask = map(np.asarray, (state.g_eta, state.g_lam, state.g_obs, state.g_mask))
    params = {key: np.asarray(array) for key, array in state.params.items()}
    for array in (eta, precision, obs, mask, *params.values()):
        if eta.dtype not in (np.float32, np.float64) or array.dtype != eta.dtype:
            raise FloatingPointError("certificate requires uniform float32 or float64 arithmetic")
        _finite(array)
    if not np.all((mask == 0) | (mask == 1)):
        raise FloatingPointError("Gaussian DATA mask must be binary")
    for edge in arena.edges.values():
        if not edge.interior and (eta[arena.g_eta_indices(edge.slot1, edge.dim)].any()
                                  or precision[arena.g_lam_indices(edge.slot1, edge.dim)].any()):
            raise FloatingPointError("nonvacuous external Gaussian message")

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        args = obs, mask, params, linearize
        mapped_eta, mapped_precision, energy = _sweep(arena, eta, precision, *args)
        _, _, successor_energy = _sweep(arena, mapped_eta, mapped_precision, *args)
        residual = max(float(np.max(np.abs(mapped - current) / (1 + np.abs(current))))
                       for mapped, current in ((mapped_eta, eta), (mapped_precision, precision)))
    stable = abs(successor_energy - energy) <= tolerance * (1 + abs(energy))
    discrepancy = max(abs(energy - evidence_cost), abs(successor_energy - evidence_cost))
    matches = discrepancy <= tolerance * (1 + abs(evidence_cost))
    return ChainCertificate(residual, energy, successor_energy, bool(residual <= tolerance and stable and matches))

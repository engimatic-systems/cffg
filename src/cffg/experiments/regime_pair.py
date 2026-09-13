"""Executable D11 equations, deliberately unbound to production execution.

Each recurrence region keeps a Gaussian conditional on its two regimes.
Only first and second moments match the Gaussian boundary (Part I §5.4).
This is a constrained Bethe candidate, not a Gaussian trajectory posterior.
It fails to convey source/state association supplied through separate external
boundaries; D11 records that falsifier and withholds production authorization.
"""

from dataclasses import dataclass

import numpy as np


def logsumexp(value, axis=None):
    value = np.asarray(value)
    maximum = np.max(value, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(maximum), maximum, 0)
    total = np.exp(value - safe).sum(axis=axis, keepdims=True)
    with np.errstate(divide="ignore"):
        result = safe + np.log(total)
    return result.reshape(()) if axis is None else np.squeeze(result, axis=axis)


def omega(c):
    """PG mean, with the removable singularity evaluated to dtype accuracy."""
    c = np.asarray(c)
    small = c <= np.sqrt(np.finfo(c.dtype).eps)
    return np.where(small, 0.25 - c * c / 48, np.tanh(c / 2) / (2 * np.where(small, 1, c)))


def gaussian_moments(eta, precision):
    """Reject an unresolvable precision; never regularize an improper belief."""
    dtype = precision.dtype
    dim = len(eta)
    eigenvalues = np.linalg.eigvalsh(precision)
    scale = np.max(np.abs(eigenvalues))
    # Relative eigenvalue resolution protects the positive-definite integral.
    threshold = 32 * dim * np.finfo(dtype).eps * scale
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= threshold:
        raise ValueError("improper or numerically singular tilted Gaussian")
    chol = np.linalg.cholesky(precision)
    covariance = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(dim, dtype=dtype)))
    mean = covariance @ eta
    log_mass = (dim * np.log(2 * np.pi) - 2 * np.log(np.diag(chol)).sum() + eta @ mean) / 2
    return mean, covariance, log_mass


def quadratic_bound(weights, offsets, c):
    """Prefix sums of at-risk sticks; O(K² d²), without a K³ score tensor."""
    card, _, dim = weights.shape
    linear = np.zeros((card, card, dim), dtype=weights.dtype)
    precision = np.zeros((card, card, dim, dim), dtype=weights.dtype)
    constant = np.zeros((card, card), dtype=weights.dtype)
    pg = omega(c)
    for i in range(card):
        h = np.zeros(dim, dtype=weights.dtype)
        jmat = np.zeros((dim, dim), dtype=weights.dtype)
        a = weights.dtype.type(0)
        for j in range(card):
            if j < card - 1:
                r, bias, w = weights[i, j], offsets[i, j], pg[i, j]
                h = h - (0.5 + w * bias) * r
                jmat = jmat + w * np.outer(r, r)
                a = a - np.logaddexp(c[i, j] / 2, -c[i, j] / 2)
                a = a + w * (c[i, j] ** 2 - bias ** 2) / 2 - bias / 2
                linear[i, j], constant[i, j] = h + r, a + bias
            else:
                linear[i, j], constant[i, j] = h, a
            precision[i, j] = jmat
    return linear, precision, constant


@dataclass(frozen=True)
class PairRegion:
    probability: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    drive_mean: np.ndarray
    drive_covariance: np.ndarray
    to_source: np.ndarray
    to_destination: np.ndarray
    to_drive_eta: np.ndarray
    to_drive_precision: np.ndarray
    log_normalizer: float
    free_energy: float
    c: np.ndarray
    status: str
    iterations: int


def _validate(weights, offsets, c, left, right, eta, precision):
    values = [np.asarray(x) for x in (weights, offsets, c, left, right, eta, precision)]
    dtype = np.result_type(*values)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("coordinates must use float32 or float64")
    weights, offsets, c, left, right, eta, precision = [x.astype(dtype) for x in values]
    if weights.ndim != 3:
        raise ValueError("weights require [K,K-1,d]")
    card, sticks, dim = weights.shape
    if card < 2 or sticks != card - 1 or dim < 1 or any((
        offsets.shape != (card, sticks), c.shape != (card, sticks),
        left.shape != (card,), right.shape != (card,), eta.shape != (dim,),
        precision.shape != (dim, dim),
    )):
        raise ValueError("coordinate shapes disagree")
    for value in (weights, offsets, c, eta, precision):
        if not np.isfinite(value).all():
            raise ValueError("non-finite coordinate")
    if (c < 0).any():
        raise ValueError("auxiliary c must be nonnegative")
    if not np.array_equal(precision, precision.T):
        raise ValueError("cavity precision must be symmetric")
    for value in (left, right):
        if np.isnan(value).any() or np.isposinf(value).any():
            raise ValueError("categorical cavity must be finite or negative infinity")
    if not np.isfinite(left).any() or not np.isfinite(right).any():
        raise ValueError("empty supported component")
    return weights, offsets, c, left, right, eta, precision


def local_objective(probability, mean, covariance, weights, offsets, c, left, right, eta, precision):
    """The complete cavity-inclusive local F, including Gaussian constants."""
    h, jmat, constant = quadratic_bound(weights, offsets, c)
    value = probability.dtype.type(0)
    dim = mean.shape[-1]
    for i, j in np.argwhere(probability > 0):
        m, s, p = mean[i, j], covariance[i, j], probability[i, j]
        sign, logdet = np.linalg.slogdet(s)
        if sign <= 0:
            raise ValueError("improper region covariance")
        second = s + np.outer(m, m)
        entropy = (dim * (1 + np.log(2 * np.pi)) + logdet) / 2
        energy = -constant[i, j] - (h[i, j] + eta) @ m
        energy += np.sum((jmat[i, j] + precision) * second) / 2
        value += p * (np.log(p) - entropy + energy - left[i] - right[j])
    if not np.isfinite(value):
        raise ValueError("non-finite local free energy")
    return float(value)


def tilted_region(weights, offsets, c, left, right, eta, precision):
    """Fixed-auxiliary stationary region and all three outgoing messages.

    Categorical messages are normalized on the active cavity face. Values
    outside that face are set to log zero; changing support needs a fresh solve.
    The Gaussian quotient can have negative precision: only tilted beliefs
    must be proper. A quotient is never silently replaced with a vacuous one.
    """
    weights, offsets, c, left, right, eta, precision = _validate(
        weights, offsets, c, left, right, eta, precision
    )
    card, _, dim = weights.shape
    h, jmat, constant = quadratic_bound(weights, offsets, c)
    score = np.full((card, card), -np.inf, dtype=weights.dtype)
    mean = np.zeros((card, card, dim), dtype=weights.dtype)
    covariance = np.zeros((card, card, dim, dim), dtype=weights.dtype)
    for i, j in np.argwhere(np.isfinite(left[:, None] + right[None, :])):
        mean[i, j], covariance[i, j], mass = gaussian_moments(eta + h[i, j], precision + jmat[i, j])
        score[i, j] = mass + constant[i, j]
    log_mass = score + left[:, None] + right[None, :]
    normalizer = logsumexp(log_mass)
    if not np.isfinite(normalizer):
        raise ValueError("non-finite normalizer")
    p = np.exp(log_mass - normalizer)
    m = np.einsum("ij,ijd->d", p, mean)
    centered = mean - m
    s = np.einsum("ij,ijab->ab", p, covariance) + np.einsum("ij,ija,ijb->ab", p, centered, centered)
    _, inv_s, _ = gaussian_moments(np.zeros(dim, dtype=weights.dtype), s)
    to_source = logsumexp(score + right[None, :], axis=1)
    to_destination = logsumexp(score + left[:, None], axis=0)
    to_source -= logsumexp(to_source)
    to_destination -= logsumexp(to_destination)
    energy = local_objective(p, mean, covariance, weights, offsets, c, left, right, eta, precision)
    return PairRegion(
        probability=p, mean=mean, covariance=covariance,
        drive_mean=m, drive_covariance=s,
        to_source=to_source, to_destination=to_destination,
        to_drive_eta=inv_s @ m - eta, to_drive_precision=inv_s - precision,
        log_normalizer=float(normalizer), free_energy=energy,
        c=c.copy(), status="fixed_auxiliary", iterations=0,
    )


def auxiliary_target(region, weights, offsets):
    """Exposure-weighted E[psi²], tying the JJ coordinate across destinations."""
    p, m, s = region.probability, region.mean, region.covariance
    card = len(p)
    target = np.zeros_like(offsets)
    for i in range(card):
        mass = p.dtype.type(0)
        first = np.zeros(m.shape[-1], dtype=p.dtype)
        second = np.zeros(s.shape[-2:], dtype=p.dtype)
        for j in range(card - 1, -1, -1):
            mass += p[i, j]
            first += p[i, j] * m[i, j]
            second += p[i, j] * (s[i, j] + np.outer(m[i, j], m[i, j]))
            if j < card - 1 and mass > 0:
                r, bias = weights[i, j], offsets[i, j]
                value = r @ second @ r + 2 * bias * (r @ first) + mass * bias ** 2
                # Roundoff in a nonnegative second moment is bounded relatively.
                scale = abs(r @ second @ r) + abs(2 * bias * (r @ first)) + mass * bias ** 2
                if value < -32 * np.finfo(p.dtype).eps * scale:
                    raise ValueError("negative squared-logit moment")
                target[i, j] = np.sqrt(max(value, 0) / mass)
    if not np.isfinite(target).all():
        raise ValueError("non-finite auxiliary residual")
    return target


def solve_region(weights, offsets, left, right, eta, precision, *, max_iterations=256):
    """Exact block minimizations of local F; no Newton/globalization assumption.

    Budget exhaustion is not convergence. This local descent says nothing
    about convergence of a graph-wide expectation-consistent message sweep.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    c = np.ones_like(offsets)
    previous = None
    tolerance = 64 * np.finfo(c.dtype).eps ** 0.75
    for iteration in range(1, max_iterations + 1):
        region = tilted_region(weights, offsets, c, left, right, eta, precision)
        target = auxiliary_target(region, weights, offsets)
        residual = np.max(abs(target - c) / (1 + abs(c)))
        delta = np.inf if previous is None else abs(region.free_energy - previous) / (1 + abs(previous))
        if previous is not None and region.free_energy > previous + tolerance * (1 + abs(previous)):
            raise ValueError("diverged: local free energy increased")
        if np.isfinite(delta) and max(residual, delta) <= tolerance:
            return PairRegion(**{**vars(region), "status": "converged", "iterations": iteration})
        previous, c = region.free_energy, target
    return PairRegion(**{**vars(region), "status": "budget_exhausted", "iterations": max_iterations})


def learning_statistics(region):
    """Pair-conditioned augmented-feature increments, in the D11 coordinate."""
    p, m, s = region.probability, region.mean, region.covariance
    card, _, dim = m.shape
    first = np.zeros((card, card - 1, dim + 1), dtype=p.dtype)
    second = np.zeros((card, card - 1, dim + 1, dim + 1), dtype=p.dtype)
    pg = omega(region.c)
    for i in range(card):
        tail_h = np.zeros(dim + 1, dtype=p.dtype)
        tail_m = np.zeros((dim + 1, dim + 1), dtype=p.dtype)
        for j in range(card - 1, -1, -1):
            h = np.append(m[i, j], p.dtype.type(1))
            moment = np.outer(h, h)
            moment[:-1, :-1] += s[i, j]
            tail_h += p[i, j] * h
            tail_m += p[i, j] * moment
            if j < card - 1:
                first[i, j] = p[i, j] * h - tail_h / 2
                second[i, j] = pg[i, j] * tail_m
    return first, second

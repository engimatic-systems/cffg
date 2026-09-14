"""Independent tiny float64 D12 EC mathematics, including squared auxiliaries.

Only NumPy/SciPy are used. ``evaluate`` is independent of iteration: every
message, conditional moment, learning increment and certificate is evaluated
at the supplied coordinate. Fixed origins are metadata. Dense blocks are
per region; the separate tiny solver may form a dense coordinate Jacobian.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


SPEC_PATH = Path(__file__).parent / "fixtures/hybrid-joint-reference-spec.json"
EPS = np.finfo(np.float64).eps
TAU = 64 * EPS**0.75
ARITHMETIC = 512 * 4 * EPS
LOG_2PI = np.log(2 * np.pi)


class ReferenceFailure(ValueError):
    """Invalid inputs or an unrepresentable active reference belief."""


@dataclass
class Natural:
    constant: np.ndarray
    linear: np.ndarray
    precision: np.ndarray

    def __add__(self, other):
        return Natural(*(a + b for a, b in zip(self.arrays(), other.arrays())))

    def arrays(self):
        return self.constant, self.linear, self.precision


@dataclass
class Belief:
    probability: np.ndarray
    log_probability: np.ndarray
    displacement: np.ndarray
    covariance: np.ndarray
    origin: np.ndarray
    active: np.ndarray
    log_normalizer: float

    @property
    def mean(self):
        return np.where(self.active[..., None], self.displacement + self.origin, 0)

    @property
    def second(self):
        m = self.mean
        return self.covariance + m[..., :, None] * m[..., None, :]


@dataclass
class Model:
    sites: list[Natural]
    origins: np.ndarray
    weights: np.ndarray
    offsets: np.ndarray
    matrix: np.ndarray
    bias: np.ndarray
    noise: np.ndarray

    @property
    def shape(self):
        return self.origins.shape

    @property
    def support(self):
        return np.array([np.isfinite(s.constant) for s in self.sites])

    @property
    def auxiliary_support(self):
        t, k, _ = self.shape
        return np.array([self.support[n, :, None] &
                         np.array([self.support[n + 1, l:].any()
                                   for l in range(k - 1)])[None, :]
                         for n in range(t - 1)])


def specification():
    return json.loads(SPEC_PATH.read_text())


def load_case(name):
    """Load copied scalar fixtures; no test module or production imports."""
    f = specification()["cases"][name]
    a, h, j = (np.asarray(f[key], dtype=float) for key in
               ("site_constant", "site_linear", "site_precision"))
    origin = np.asarray(f["origins"], dtype=float)[..., None]
    c = origin[..., 0]
    sites = [Natural(aa + hh * cc - jj * cc**2 / 2,
                     (hh - jj * cc)[:, None], jj[:, None, None])
             for aa, hh, jj, cc in zip(a, h, j, c)]
    return Model(sites, origin, np.asarray(f["weights"])[..., None, None],
                 np.asarray(f["offsets"])[..., None],
                 np.asarray(f["matrix"])[..., None, None],
                 np.asarray(f["bias"])[..., None],
                 np.asarray(f["noise"])[..., None, None])


def _finite(value, name):
    if not np.isfinite(value).all():
        raise ReferenceFailure(f"nonfinite {name}")


def validate(model, xi):
    t, k, d = model.shape
    if not 2 <= t <= 3 or k < 2 or d < 1:
        raise ReferenceFailure("reference requires T=2 or T=3, K>=2, d>=1")
    shapes = {"weights": (t - 1, k, k - 1, d), "offsets": (t - 1, k, k - 1),
              "matrix": (t - 1, k, d, d), "bias": (t - 1, k, d),
              "noise": (t - 1, k, d, d), "origins": (t, k, d)}
    for name, shape in shapes.items():
        value = getattr(model, name)
        if value.shape != shape:
            raise ReferenceFailure(f"invalid {name} shape")
        _finite(value, name)
    if len(model.sites) != t:
        raise ReferenceFailure("invalid site count")
    for site in model.sites:
        if tuple(x.shape for x in site.arrays()) != ((k,), (k, d), (k, d, d)):
            raise ReferenceFailure("invalid site shape")
        if np.isnan(site.constant).any() or np.isposinf(site.constant).any():
            raise ReferenceFailure("invalid site constant")
        if not np.isfinite(site.constant).any():
            raise ReferenceFailure("empty support")
        _finite(site.linear, "site linear")
        _finite(site.precision, "site precision")
        if not np.array_equal(site.precision, site.precision.swapaxes(-1, -2)):
            raise ReferenceFailure("nonsymmetric site precision")
    if xi.shape != (t - 1, k, k - 1) or not np.isfinite(xi).all() or (xi < 0).any():
        raise ReferenceFailure("invalid squared auxiliary")
    if np.any(xi[~model.auxiliary_support] != 0):
        raise ReferenceFailure("inactive squared auxiliary must be zero")
    for q in model.noise.reshape(-1, d, d):
        _inverse(q)


def _inverse(matrix):
    if not np.array_equal(matrix, matrix.T):
        raise ReferenceFailure("nonsymmetric precision or covariance")
    try:
        chol = np.linalg.cholesky(matrix)
        inverse = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(len(chol))))
    except np.linalg.LinAlgError as error:
        raise ReferenceFailure("improper active Gaussian block") from error
    inverse = (inverse + inverse.T) / 2
    _finite(inverse, "Gaussian inverse")
    return inverse, 2 * np.log(np.diag(chol)).sum()


def omega(xi):
    xi = np.asarray(xi, dtype=float)
    c = np.sqrt(xi)
    answer = np.full_like(c, .25)
    np.divide(np.tanh(c / 2), 2 * c, out=answer, where=c != 0)
    return answer


def factor(model, time, xi):
    """D12.2 expanded at frozen origins, from original recurrence/channel data."""
    _, k, d = model.shape
    a = np.zeros((k, k))
    h = np.zeros((k, k, 2 * d))
    precision = np.zeros((k, k, 2 * d, 2 * d))
    for i in range(k):
        source_origin = model.origins[time, i]
        for j in range(k):
            for l in range(min(j + 1, k - 1)):
                w = model.weights[time, i, l]
                o = model.offsets[time, i, l] + w @ source_origin
                pg = omega(xi[i, l])
                c = np.sqrt(xi[i, l])
                sign = .5 if j == l else -.5
                a[i, j] += sign * o - np.logaddexp(c / 2, -c / 2) + pg * (xi[i, l] - o * o) / 2
                h[i, j, :d] += (sign - pg * o) * w
                precision[i, j, :d, :d] += pg * np.outer(w, w)
            transform = model.matrix[time, j]
            offset = transform @ source_origin + model.bias[time, j] - model.origins[time + 1, j]
            qinv, qlogdet = _inverse(model.noise[time, j])
            channel = np.concatenate((-transform, np.eye(d)), axis=1)
            precision[i, j] += channel.T @ qinv @ channel
            h[i, j] += channel.T @ qinv @ offset
            a[i, j] -= (d * LOG_2PI + qlogdet + offset @ qinv @ offset) / 2
    precision = (precision + precision.swapaxes(-1, -2)) / 2
    return Natural(a, h, precision)


def lift(left, right):
    k, d = left.linear.shape
    h = np.zeros((k, k, 2 * d))
    p = np.zeros((k, k, 2 * d, 2 * d))
    h[..., :d] = left.linear[:, None]
    h[..., d:] = right.linear[None, :]
    p[..., :d, :d] = left.precision[:, None]
    p[..., d:, d:] = right.precision[None, :]
    return Natural(left.constant[:, None] + right.constant[None, :], h, p)


def normalize(natural, origin):
    active = np.isfinite(natural.constant)
    if not active.any():
        raise ReferenceFailure("empty support")
    if np.isnan(natural.constant).any() or np.isposinf(natural.constant).any():
        raise ReferenceFailure("nonfinite natural constant")
    _finite(natural.linear, "natural linear")
    _finite(natural.precision, "natural precision")
    mean = np.zeros_like(natural.linear)
    covariance = np.zeros_like(natural.precision)
    score = np.full_like(natural.constant, -np.inf)
    d = mean.shape[-1]
    for index in zip(*np.nonzero(active)):
        covariance[index], determinant = _inverse(natural.precision[index])
        mean[index] = covariance[index] @ natural.linear[index]
        score[index] = natural.constant[index] + (d * LOG_2PI - determinant + natural.linear[index] @ mean[index]) / 2
    z = float(logsumexp(score))
    _finite(z, "normalizer")
    log_p = score - z
    probability = np.exp(log_p)
    if np.any(probability[active] < np.finfo(float).tiny):
        raise ReferenceFailure("tiny positive active probability underflow")
    _finite(mean, "conditional mean")
    _finite(covariance, "conditional covariance")
    return Belief(probability, log_p, mean, covariance, origin, active, z)


def project(region, side, origin):
    """Project conditional mixtures without dividing by small label masses."""
    k, d = origin.shape
    block = slice(side * d, (side + 1) * d)
    active = region.active.any(axis=1 - side)
    log_p = logsumexp(region.log_probability, axis=1 - side)
    mean = np.zeros((k, d))
    covariance = np.zeros((k, d, d))
    for i in np.flatnonzero(active):
        indices = [(i, j) if side == 0 else (j, i) for j in range(k)]
        indices = [ij for ij in indices if region.active[ij]]
        conditional = np.exp(np.array([region.log_probability[ij] for ij in indices]) - log_p[i])
        means = np.array([region.displacement[ij][block] for ij in indices])
        mean[i] = conditional @ means
        for weight, ij, m in zip(conditional, indices, means):
            delta = m - mean[i]
            covariance[i] += weight * (region.covariance[ij][block, block] + np.outer(delta, delta))
        covariance[i] = (covariance[i] + covariance[i].T) / 2
        _inverse(covariance[i])
    return Belief(np.exp(log_p), log_p, mean, covariance, origin, active, 0.)


def quotient(belief, cavity):
    """Complete normalized natural quotient; only a common constant is gauge."""
    a = np.zeros_like(cavity.constant)
    h = np.zeros_like(cavity.linear)
    p = np.zeros_like(cavity.precision)
    d = h.shape[-1]
    for i in np.flatnonzero(belief.active):
        inverse, determinant = _inverse(belief.covariance[i])
        m = belief.displacement[i]
        p[i] = inverse - cavity.precision[i]
        h[i] = inverse @ m - cavity.linear[i]
        a[i] = belief.log_probability[i] - (d * LOG_2PI + determinant + m @ inverse @ m) / 2 - cavity.constant[i]
    a[belief.active] -= a[np.flatnonzero(belief.active)[0]]
    return Natural(a, h, p)


def entropy(belief):
    d = belief.displacement.shape[-1]
    result = 0.
    for index in zip(*np.nonzero(belief.active)):
        _, determinant = _inverse(belief.covariance[index])
        result += belief.probability[index] * ((d * (1 + LOG_2PI) + determinant) / 2 - belief.log_probability[index])
    return float(result)


def expectation(belief, natural):
    answer = 0.
    for index in zip(*np.nonzero(belief.active)):
        m, v = belief.displacement[index], belief.covariance[index]
        answer += belief.probability[index] * (natural.constant[index] + natural.linear[index] @ m - (np.sum(natural.precision[index] * v) + m @ natural.precision[index] @ m) / 2)
    return float(answer)


def statistics(model, time, region, xi):
    """D12.8 targets and D11.1 centered/physical augmented learning moments."""
    _, k, d = model.shape
    target, exposure = np.zeros((k, k - 1)), np.zeros((k, k - 1))
    linear = np.zeros((k, k - 1, d + 1))
    precision = np.zeros((k, k - 1, d + 1, d + 1))
    pg = omega(xi)
    for i in range(k):
        for l in range(k - 1):
            indices = [j for j in range(l, k) if region.active[i, j]]
            if not indices:
                continue
            logs = np.array([region.log_probability[i, j] for j in indices])
            z = logsumexp(logs)
            conditional = np.exp(logs - z)
            exposure[i, l] = np.exp(z)
            theta = np.append(model.weights[time, i, l], model.offsets[time, i, l] + model.weights[time, i, l] @ model.origins[time, i])
            for cond, j in zip(conditional, indices):
                h = np.append(region.displacement[i, j, :d], 1.)
                moment = np.outer(h, h)
                moment[:d, :d] += region.covariance[i, j, :d, :d]
                logit_mean = theta @ h
                square = theta[:d] @ region.covariance[i, j, :d, :d] @ theta[:d] + logit_mean**2
                if np.any(theta != 0) and square < np.finfo(float).tiny:
                    raise ReferenceFailure("tiny positive auxiliary target underflow")
                target[i, l] += cond * square
                mass = region.probability[i, j]
                linear[i, l] += (.5 if j == l else -.5) * mass * h
                precision[i, l] += pg[i, l] * mass * moment
    physical_linear, physical_precision = np.zeros_like(linear), np.zeros_like(precision)
    for i in range(k):
        transform = np.eye(d + 1)
        transform[:d, d] = model.origins[time, i]
        physical_linear[i] = linear[i] @ transform.T
        physical_precision[i] = transform @ precision[i] @ transform.T
    return dict(target=target, exposure=exposure, centered_linear=linear,
                centered_precision=precision, linear=physical_linear, precision=physical_precision)


def zero_messages(model):
    t, k, d = model.shape
    return [[Natural(np.zeros(k), np.zeros((k, d)), np.zeros((k, d, d)))
             for _ in range(2)] for _ in range(t - 1)]


def cavities(model, messages, time):
    t = model.shape[0]
    left, right = model.sites[time], model.sites[time + 1]
    if time:
        left = left + messages[time - 1][1]
    if time < t - 2:
        right = right + messages[time + 1][0]
    return left, right


def tilted(model, messages, xi):
    factors, regions, projections, targets = [], [], [], []
    for time in range(model.shape[0] - 1):
        f = factor(model, time, xi[time])
        left, right = cavities(model, messages, time)
        k, d = left.linear.shape
        origins = np.empty((k, k, 2 * d))
        origins[..., :d] = model.origins[time, :, None]
        origins[..., d:] = model.origins[time + 1, None, :]
        region = normalize(f + lift(left, right), origins)
        projected = [project(region, side, model.origins[time + side]) for side in range(2)]
        factors.append(f)
        regions.append(region)
        projections.append(projected)
        targets.append([quotient(p, c) for p, c in zip(projected, (left, right))])
    return factors, regions, projections, targets


def _scaled_difference(a, b):
    return (a - b) / (1 + abs(b))


def evaluate(model, messages, xi):
    """Recompute D12.4/.7/.8/.9 and all arrays at one held coordinate.

    Energy stability needs a previous accepted evaluation and is added by the
    solver. Unweighted conditional residuals are retained for each incidence.
    Invalid or improper active beliefs raise ReferenceFailure, never success.
    """
    validate(model, xi)
    t, k, d = model.shape
    if len(messages) != t - 1 or any(len(bank) != 2 for bank in messages):
        raise ReferenceFailure("invalid message count")
    for time, bank in enumerate(messages):
        for side, message in enumerate(bank):
            if tuple(x.shape for x in message.arrays()) != ((k,), (k, d), (k, d, d)):
                raise ReferenceFailure("invalid message shape")
            for x in message.arrays():
                _finite(x, "message")
            support = model.support[time + side]
            if message.constant[np.flatnonzero(support)[0]] != 0:
                raise ReferenceFailure("message common gauge is not fixed")
            if any(np.any(x[~support] != 0) for x in message.arrays()):
                raise ReferenceFailure("inactive message coordinates must be zero")
    factors, regions, projections, outgoing = tilted(model, messages, xi)
    separators = []
    for time, site in enumerate(model.sites):
        total = site
        if time:
            total = total + messages[time - 1][1]
        if time < t - 1:
            total = total + messages[time][0]
        separators.append(normalize(total, model.origins[time]))
    for belief in regions + separators:
        _finite(belief.mean, "physical conditional mean")
        _finite(belief.second, "physical second moment")
    natural, means, covariances, masses = [], [], [], []
    for time in range(t - 1):
        for side in range(2):
            p, q = projections[time][side], separators[time + side]
            natural.append(Natural(*(_scaled_difference(a, b) for a, b in
                                     zip(outgoing[time][side].arrays(), messages[time][side].arrays()))))
            means.append(_scaled_difference(p.displacement, q.displacement))
            covariances.append(_scaled_difference(p.covariance, q.covariance))
            masses.append(p.probability - q.probability)
    learning = [statistics(model, time, r, xi[time]) for time, r in enumerate(regions)]
    auxiliary_target = np.array([value["target"] for value in learning])
    auxiliary_residual = _scaled_difference(auxiliary_target, xi)
    primal = sum(-entropy(r) - expectation(r, f) for r, f in zip(regions, factors))
    primal += sum(entropy(q) for q in separators[1:-1])
    primal -= sum(expectation(q, site) for q, site in zip(separators, model.sites))
    dual = -sum(r.log_normalizer for r in regions) + sum(q.log_normalizer for q in separators[1:-1])
    certificates = {"natural_residual": max(np.max(abs(x)) for n in natural for x in n.arrays()),
                    "conditional_mean_residual": np.max(abs(np.array(means))),
                    "conditional_covariance_residual": np.max(abs(np.array(covariances))),
                    "label_mass_residual": np.max(abs(np.array(masses))),
                    "auxiliary_residual": np.max(abs(auxiliary_residual)),
                    "primal_dual_residual": abs(primal - dual) / (1 + abs(primal))}
    _finite([primal, dual, *certificates.values()], "objective or certificate")
    return dict(messages=messages, xi=xi.copy(), outgoing=outgoing, regions=regions,
                separators=separators, projections=projections, learning=learning,
                auxiliary_target=auxiliary_target, natural_residuals=natural,
                conditional_mean_residuals=np.array(means),
                conditional_covariance_residuals=np.array(covariances),
                label_mass_residuals=np.array(masses), auxiliary_residuals=auxiliary_residual,
                free_energy=float(primal), dual_energy=float(dual),
                proper_active_beliefs=True, certificates=certificates)


def fixed_drive(weights, offsets, drive, source_log_site, destination_log_site):
    """D9 conditioned categorical reduction; no Gaussian entropy at a point."""
    k, sticks, d = weights.shape
    if (sticks != k - 1 or drive.shape != (d,) or offsets.shape != (k, sticks)
            or source_log_site.shape != (k,) or destination_log_site.shape != (k,)):
        raise ReferenceFailure("invalid fixed-drive shape")
    for value in (weights, offsets, drive):
        _finite(value, "fixed-drive input")
    for value in (source_log_site, destination_log_site):
        if np.isnan(value).any() or np.isposinf(value).any():
            raise ReferenceFailure("invalid fixed-drive support")
    psi = weights @ drive + offsets
    _finite(psi, "fixed-drive logit")
    xi = psi**2
    _finite(xi, "fixed-drive squared auxiliary")
    if np.any((psi != 0) & (xi < np.finfo(float).tiny)):
        raise ReferenceFailure("tiny positive fixed-drive auxiliary underflow")
    log_factor = np.zeros((k, k))
    bound = np.zeros_like(log_factor)
    for i in range(k):
        for j in range(k):
            for l in range(min(j + 1, sticks)):
                sign = .5 if j == l else -.5
                log_factor[i, j] -= np.logaddexp(0, (-1 if j == l else 1) * psi[i, l])
                c = np.sqrt(xi[i, l])
                bound[i, j] += sign * psi[i, l] - np.logaddexp(c / 2, -c / 2)
    log_mass = source_log_site[:, None] + destination_log_site[None, :] + log_factor
    log_z = logsumexp(log_mass)
    if not np.isfinite(log_z):
        raise ReferenceFailure("empty fixed-drive support")
    pair = np.exp(log_mass - log_z)
    if np.any(pair[np.isfinite(log_mass)] < np.finfo(float).tiny):
        raise ReferenceFailure("tiny positive fixed-drive probability underflow")
    h = np.append(drive, 1.)
    linear = np.zeros((k, sticks, d + 1))
    precision = np.zeros((k, sticks, d + 1, d + 1))
    exposure = np.zeros((k, sticks))
    for i in range(k):
        for l in range(sticks):
            exposure[i, l] = pair[i, l:].sum()
            linear[i, l] = (pair[i, l] - exposure[i, l] / 2) * h
            precision[i, l] = omega(xi[i, l]) * exposure[i, l] * np.outer(h, h)
    xi[exposure == 0] = 0
    _finite(linear, "fixed-drive linear statistic")
    _finite(precision, "fixed-drive precision statistic")
    return dict(pair=pair, xi=xi, log_factor=log_factor, bound=bound,
                free_energy=float(-log_z), linear=linear, precision=precision,
                left_message=logsumexp(log_factor + destination_log_site[None, :], axis=1),
                right_message=logsumexp(log_factor + source_log_site[:, None], axis=0))

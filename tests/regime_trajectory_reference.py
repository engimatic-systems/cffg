"""Tiny exhaustive D11 authority: no imports from cffg or the candidate.

K=2, scalar drives, at most six regimes. Each path is integrated by a dense
Gaussian completion; the reference alone may store a full trajectory matrix.
An independent adaptive quadrature also integrates the literal logistic
factor and arbitrary trial-belief free energy for a single recurrence node.
"""

import itertools

import numpy as np
from scipy.integrate import quad
from scipy.special import expit, logsumexp


def enumerate_paths(unary, weights, offsets, c, eta, precision):
    time, card = unary.shape
    if card != 2 or not 2 <= time <= 6:
        raise ValueError("reference restricted to K=2 and 2 <= T <= 6")
    drives = time - 1
    paths, scores, means, covariances = [], [], [], []
    for path in itertools.product(range(2), repeat=time):
        constant = sum(unary[t, path[t]] for t in range(time))
        if not np.isfinite(constant):
            continue
        jmat, h = precision.copy(), eta.copy()
        for t in range(drives):
            i, j = path[t:t + 2]
            r, bias, aux = weights[t, i], offsets[t, i], c[t, i]
            w = 0.25 if aux == 0 else np.tanh(aux / 2) / (2 * aux)
            sign = 1 if j == 0 else -1
            jmat[t, t] += w * r * r
            h[t] += (sign / 2 - w * bias) * r
            constant += sign * bias / 2 - np.logaddexp(aux / 2, -aux / 2)
            constant += w * (aux * aux - bias * bias) / 2
        chol = np.linalg.cholesky(jmat)
        covariance = np.linalg.solve(jmat, np.eye(drives))
        mean = np.linalg.solve(jmat, h)
        mass = drives * np.log(2 * np.pi) / 2 - np.log(np.diag(chol)).sum() + h @ mean / 2
        paths.append(path)
        scores.append(constant + mass)
        means.append(mean)
        covariances.append(covariance)
    if not paths:
        raise ValueError("empty supported component")
    normalizer = logsumexp(scores)
    probability = np.exp(scores - normalizer)
    singleton = np.zeros((time, 2))
    pair = np.zeros((drives, 2, 2))
    first = np.zeros_like(pair)
    second = np.zeros_like(pair)
    for path, p, m, s in zip(paths, probability, means, covariances):
        for t, i in enumerate(path):
            singleton[t, i] += p
        for t in range(drives):
            i, j = path[t:t + 2]
            pair[t, i, j] += p
            first[t, i, j] += p * m[t]
            second[t, i, j] += p * (s[t, t] + m[t] ** 2)
    return dict(log_normalizer=normalizer, singleton=singleton, pair=pair,
                first=first, second=second, path_probability=probability,
                path_mean=np.asarray(means), path_covariance=np.asarray(covariances))


def quadrature_node(left, right, weights, offsets, eta, precision, *, c=None):
    """Integrate exact logistic or a fixed JJ bound, with canonical cavities."""
    left, right, weights, offsets = [np.asarray(x, dtype=np.float64) for x in (left, right, weights, offsets)]
    eta, precision = float(eta), float(precision)
    c = None if c is None else np.asarray(c, dtype=np.float64)
    mass, first, second = (np.zeros((2, 2)) for _ in range(3))
    for i, j in itertools.product(range(2), repeat=2):
        if not np.isfinite(left[i] + right[j]):
            continue

        def density(x):
            psi = weights[i] * x + offsets[i]
            log_cavity = left[i] + right[j] + eta * x - precision * x * x / 2
            if c is None:
                factor = expit(psi if j == 0 else -psi)
                return np.exp(log_cavity) * factor
            aux = c[i]
            w = 0.25 if aux == 0 else np.tanh(aux / 2) / (2 * aux)
            bound = (0.5 if j == 0 else -0.5) * psi
            bound -= np.logaddexp(aux / 2, -aux / 2) + w * (psi * psi - aux * aux) / 2
            return np.exp(log_cavity + bound)

        for order, out in enumerate((mass, first, second)):
            out[i, j] = quad(lambda x: x ** order * density(x), -np.inf, np.inf,
                             epsabs=2e-11, epsrel=2e-11)[0]
    z = mass.sum()
    return dict(log_normalizer=np.log(z), pair=mass / z, first=first / z, second=second / z)


def quadrature_objective(probability, mean, variance, left, right, weights, offsets,
                         c, eta, precision, *, literal=False):
    """Numerically integrate E_b[log b - log factor - log cavities]."""
    probability, mean, variance, left, right, weights, offsets, c = [
        np.asarray(x, dtype=np.float64)
        for x in (probability, mean, variance, left, right, weights, offsets, c)
    ]
    eta, precision = float(eta), float(precision)
    result = 0.0
    for i, j in itertools.product(range(2), repeat=2):
        p = probability[i, j]
        if p == 0:
            continue
        m, v = mean[i, j], variance[i, j]
        aux = c[i]
        w = 0.25 if aux == 0 else np.tanh(aux / 2) / (2 * aux)

        def integrand(y):
            x = m + np.sqrt(v) * y
            log_q = np.log(p) - (np.log(2 * np.pi * v) + y * y) / 2
            psi = weights[i] * x + offsets[i]
            if literal:
                log_factor = -np.logaddexp(0, -psi if j == 0 else psi)
            else:
                log_factor = (0.5 if j == 0 else -0.5) * psi
                log_factor -= np.logaddexp(aux / 2, -aux / 2) + w * (psi * psi - aux * aux) / 2
            log_cavity = left[i] + right[j] + eta * x - precision * x * x / 2
            return p * np.exp(-y * y / 2) / np.sqrt(2 * np.pi) * (log_q - log_factor - log_cavity)

        result += quad(integrand, -np.inf, np.inf, epsabs=2e-11, epsrel=2e-11)[0]
    return result


def switched_prior_reference(source_means, source_variances=(1, 1)):
    """A source-dependent drive prior with observed recurrence success, R=r=0.

    Enumerate both source/destination pairs and integrate each labeled prior.
    The success likelihood is exactly 1/2, so there is no JJ error to blame.
    """
    from scipy.stats import norm

    pair, first, second = (np.zeros((2, 2)) for _ in range(3))
    for i, j in itertools.product(range(2), repeat=2):
        if j != 0:
            continue
        for power, output in enumerate((pair, first, second)):
            output[i, j] = quad(
                lambda x: x ** power * norm.pdf(x, float(source_means[i]), np.sqrt(float(source_variances[i]))) / 4,
                -np.inf, np.inf, epsabs=2e-11, epsrel=2e-11,
            )[0]
    z = pair.sum()
    return dict(pair=pair / z, first=first / z, second=second / z, log_normalizer=np.log(z))

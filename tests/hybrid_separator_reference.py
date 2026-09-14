"""Independent scalar, K=2, T<=6 Gaussian path enumeration for D12.

No candidate or production imports. Dense trajectory covariance and complete
path enumeration are confined to this tiny reference. The JJ polynomial is
written directly from the binary logistic bound, independently of D11/D12.
"""

import itertools

import numpy as np
from scipy.special import logsumexp


def enumerate_chain(site_constant, site_linear, site_precision, weights, offsets, c,
                    matrix, bias, noise, *, literal=False, order=32):
    time, card = site_constant.shape
    if card != 2 or not 2 <= time <= 6:
        raise ValueError("reference restricted to K=2 and 2<=T<=6")
    if literal and time > 3:
        raise ValueError("literal quadrature restricted to T<=3")
    if literal:
        nodes, weights_gh = np.polynomial.hermite.hermgauss(order)
        indices = np.array(list(itertools.product(range(order), repeat=time)))
        quadrature_points = nodes[indices]
        quadrature_weights = np.prod(weights_gh[indices] / np.sqrt(np.pi), axis=1)
    paths, scores, means, covariances = [], [], [], []
    for path in itertools.product(range(card), repeat=time):
        constant = sum(site_constant[t, i] for t, i in enumerate(path))
        if not np.isfinite(constant):
            continue
        h = np.array([site_linear[t, i] for t, i in enumerate(path)], dtype=float)
        precision = np.diag([site_precision[t, i] for t, i in enumerate(path)])
        for t, (i, j) in enumerate(zip(path[:-1], path[1:])):
            r, intercept, aux = weights[t, i], offsets[t, i], c[t, i]
            pg = 0.25 if aux == 0 else np.tanh(aux / 2) / (2 * aux)
            kappa = 0.5 if j == 0 else -0.5
            if not literal:
                constant += kappa * intercept - np.logaddexp(aux / 2, -aux / 2)
                constant += pg * (aux * aux - intercept * intercept) / 2
                h[t] += (kappa - pg * intercept) * r
                precision[t, t] += pg * r * r
            a, b, q = matrix[t, j], bias[t, j], noise[t, j]
            precision[t, t] += a * a / q
            precision[t + 1, t + 1] += 1 / q
            precision[t, t + 1] -= a / q
            precision[t + 1, t] -= a / q
            h[t] -= a * b / q
            h[t + 1] += b / q
            constant -= (np.log(2 * np.pi * q) + b * b / q) / 2
        chol = np.linalg.cholesky(precision)
        m = np.linalg.solve(precision, h)
        s = np.linalg.solve(precision, np.eye(time))
        score = constant + time * np.log(2 * np.pi) / 2 - np.log(np.diag(chol)).sum() + h @ m / 2
        if literal:
            from scipy.special import expit

            points = m + np.sqrt(2) * quadrature_points @ np.linalg.cholesky(s).T
            integrand = quadrature_weights.copy()
            for t, (i, j) in enumerate(zip(path[:-1], path[1:])):
                integrand *= expit((1 if j == 0 else -1) * (weights[t, i] * points[:, t] + offsets[t, i]))
            integral = integrand.sum()
            score += np.log(integral)
            m = np.einsum("n,nt->t", integrand / integral, points)
            centered = points - m
            s = np.einsum("n,na,nb->ab", integrand / integral, centered, centered)
        scores.append(score)
        paths.append(path)
        means.append(m)
        covariances.append(s)
    normalizer = logsumexp(scores)
    if not np.isfinite(normalizer):
        raise ValueError("empty or improper reference")
    pair = np.zeros((time - 1, card, card))
    first = np.zeros(pair.shape + (2,))
    second = np.zeros(pair.shape + (2, 2))
    for path, p, m, s in zip(paths, np.exp(scores - normalizer), means, covariances):
        for t, (i, j) in enumerate(zip(path[:-1], path[1:])):
            pair[t, i, j] += p
            first[t, i, j] += p * m[t:t + 2]
            second[t, i, j] += p * (s[t:t + 2, t:t + 2] + np.outer(m[t:t + 2], m[t:t + 2]))
    return dict(pair=pair, first=first, second=second, free_energy=-float(normalizer))


def conditional_moments(pair, first, second, side):
    axis = 1 - side
    p = pair.sum(axis=axis)
    m = first[..., side].sum(axis=axis) / p
    v = second[..., side, side].sum(axis=axis) / p - m * m
    return p, m, v


def recurrence_increments(pair, first, second, c):
    """Binary augmented-feature sufficient statistics, independent contractions."""
    linear, precision = np.zeros((2, 2)), np.zeros((2, 2, 2))
    for i, j in itertools.product(range(2), repeat=2):
        p, h, s = pair[i, j], first[i, j, 0], second[i, j, 0, 0]
        linear[i] += (0.5 if j == 0 else -0.5) * np.array([h, p])
        w = 0.25 if c[i] == 0 else np.tanh(c[i] / 2) / (2 * c[i])
        precision[i] += w * np.array([[s, h], [h, p]])
    return linear, precision


def reflected_mixture_gap(sign=1, *, full=False):
    """Two moment-consistent temporal regions with an exact, nonzero EC gap.

    Left mixture N(2,1), N(-1,1) is reflected for the twin. Right likelihood
    has destinations centered at +/-1/sqrt(2), variance 1/2. Both hybrid
    separators match N(0,1) after tilt, but four-component exact integration
    distinguishes the twins. Recurrence is constant, so JJ is exact.
    """
    from scipy.stats import norm

    p = np.exp(1) / (np.exp(1) + 2 * np.exp(0.25))
    source = np.array([2, -1]) * sign
    destination = np.array([1, -1]) / np.sqrt(2)
    mass, moment, second = (np.zeros((2, 2)) for _ in range(3))
    for a, b in itertools.product(range(2), repeat=2):
        mass[a, b] = [p, 1 - p][a] * 0.5 * norm.pdf(source[a], destination[b], np.sqrt(1.5))
        m = (source[a] + 2 * destination[b]) / 3
        moment[a, b] = mass[a, b] * m
        second[a, b] = mass[a, b] * (1 / 3 + m * m)
    mean = float(moment.sum() / mass.sum())
    probability = mass.sum(axis=0) / mass.sum()
    if full:
        moment_by_destination = moment.sum(axis=0) / mass.sum()
        a = np.array([(moment_by_destination[0] - moment_by_destination[1]) / 4,
                      (probability[0] - probability[1]) / 4])
        b = np.array([[second.sum() / mass.sum(), mean], [mean, 1]]) / 8
        return dict(mean=mean, second=second.sum() / mass.sum(), destination=probability, linear=a, precision=b)
    return mean, probability

"""Independent finite-history checks for the reset model in D1a."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import betaln, digamma, gammaln, logsumexp, xlogy


def log_evidence(alpha, counts):
    """Integral of an ordered sequence likelihood, or its fractional-count power."""

    def log_beta(a):
        return gammaln(a).sum() - gammaln(a.sum())

    return log_beta(alpha + counts) - log_beta(alpha)


def project(weights, columns):
    totals = columns.sum(axis=1)
    means = columns / totals[:, None]
    mean = weights @ means
    categorical_trace = (means * (1 - means)).sum(axis=1)
    within = categorical_trace / (totals + 1)
    between = np.square(means - mean).sum(axis=1)
    variance = weights @ (within + between)
    concentration = (weights @ (totals * within)) / variance
    return mean * concentration


def update(prior, carry, counts, hazard):
    """D1a column formula, evaluated on its declared simplex face."""
    prior, carry, counts = map(np.asarray, (prior, carry, counts))
    support = prior > 0
    if (
        prior.ndim != 1
        or counts.shape != prior.shape
        or carry.shape != prior.shape
        or not support.any()
        or not np.array_equal(carry > 0, support)
        or not all(
            np.isfinite(x).all() and (x >= 0).all() for x in (prior, carry, counts)
        )
        or not 0 <= hazard <= 1
    ):
        raise ValueError("invalid column or hazard")
    if counts[~support].any():
        raise ValueError("impossible outcome")
    if hazard == 0 or hazard == 1:
        return (carry if hazard == 0 else prior) + counts
    branches = np.stack([carry[support], prior[support]])
    log_weights = np.log([1 - hazard, hazard]) + [
        log_evidence(branch, counts[support]) for branch in branches
    ]
    weights = np.exp(log_weights - logsumexp(log_weights))
    posteriors = branches + counts[support]
    result = np.zeros_like(carry)
    if support.sum() == 1:
        # Every positive concentration represents the same point mass.
        result[support] = prior[support]
    else:
        result[support] = project(weights, posteriors)
    return result


def enumerate_histories(prior, carry, hazard, sequence):
    """Exact rational branch weights from the sequential categorical predictions."""
    outcomes = []
    for resets in product((False, True), repeat=len(sequence)):
        weight = Fraction(1)
        column = list(carry)
        for reset, outcome in zip(resets, sequence, strict=True):
            weight *= hazard if reset else 1 - hazard
            if reset:
                column = list(prior)
            weight *= Fraction(column[outcome], sum(column))
            column[outcome] += 1
        outcomes.append((weight, column))
    evidence = sum(w for w, _ in outcomes)
    return evidence, [(w / evidence, c) for w, c in outcomes]


def moments(branches):
    """E(theta) and trace E(theta theta^T), via exact rising factorials."""
    mean = np.zeros(len(branches[0][1]))
    second = 0.0
    for weight, column in branches:
        total = sum(column)
        mean += float(weight) * np.asarray(column) / total
        second += (
            float(weight) * sum(v * (v + 1) for v in column) / (total * (total + 1))
        )
    return mean, second


def novelty(branches):
    weights, columns = zip(*branches, strict=True)
    weights = np.array(weights, dtype=float)
    alpha = np.array(columns, dtype=float)
    total = alpha.sum(axis=1, keepdims=True)
    means = alpha / total
    mean = weights @ means
    entropy = (means * (digamma(total + 1) - digamma(alpha + 1))).sum(axis=1)
    return -xlogy(mean, mean).sum() - weights @ entropy


def test_history_reference_and_bounded_projection():
    prior, carry, h = [1, 1], [8, 2], Fraction(1, 4)
    a, b = np.array(prior, dtype=float), np.array(carry, dtype=float)
    counts = np.eye(2, dtype=float)
    first = update(a, b, counts[1], float(h))
    evidence, branches = enumerate_histories(prior, carry, h, [1])
    assert evidence == Fraction(11, 40)
    assert branches[1][0] == Fraction(5, 11)
    exact_first = moments(branches)
    for got, expected in zip(moments([(1, first)]), exact_first, strict=True):
        np.testing.assert_allclose(got, expected, atol=1e-14, rtol=0)

    # Independent beta quadrature of expected data entropy of the mixture.
    def integrand(x):
        density = sum(
            float(w)
            * np.exp((p - 1) * np.log(x) + (q - 1) * np.log1p(-x) - betaln(p, q))
            for w, (p, q) in branches
        )
        return -density * (xlogy(x, x) + xlogy(1 - x, 1 - x))

    entropy = -xlogy(exact_first[0], exact_first[0]).sum()
    np.testing.assert_allclose(
        novelty(branches), entropy - quad(integrand, 0, 1)[0], atol=1e-10
    )
    np.testing.assert_allclose(
        [novelty(branches), novelty([(1, first)])],
        [0.16489859185, 0.16720858211],
        atol=1e-10,
        rtol=0,
    )
    # Full history is the gold standard after repeated projection.
    sequence = [1, 0, 1, 1]
    bounded = b
    for outcome in sequence:
        bounded = update(a, bounded, counts[outcome], float(h))
    _, history = enumerate_histories(prior, carry, h, sequence)
    full_mean, _ = moments(history)
    np.testing.assert_allclose(
        [
            full_mean[0],
            bounded[0] / bounded.sum(),
            novelty(history),
            novelty([(1, bounded)]),
        ],
        [0.35270682, 0.34611533, 0.12687641536, 0.10977529881],
        atol=5e-9,
        rtol=0,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_stationary_reset_limits_and_empty_likelihood(dtype):
    prior = np.linspace(1, 2, 64, dtype=dtype)
    carry = prior + np.arange(64, dtype=dtype)
    counts = np.zeros_like(prior)
    counts[13] = 1
    np.testing.assert_array_equal(update(prior, carry, counts, 0), carry + counts)
    np.testing.assert_array_equal(update(prior, carry, counts, 1), prior + counts)
    predicted = update(prior, carry, 0 * counts, 0.25)
    expected = 0.75 * carry / carry.sum() + 0.25 * prior / prior.sum()
    np.testing.assert_allclose(
        predicted / predicted.sum(), expected, rtol=16 * np.finfo(dtype).eps
    )
    assert np.isfinite(predicted).all() and (predicted > 0).all()


def test_face_and_domain():
    a, b = np.array([1.0, 0.0, 2.0]), np.array([2.0, 0.0, 4.0])
    n = np.array([1.0, 0.0, 0.0])
    result = update(a, b, n, 0.5)
    assert result[1] == 0 and (result[[0, 2]] > 0).all()
    point = update(
        np.array([0.0, 2.0]), np.array([0.0, 9.0]), np.array([0.0, 1.0]), 0.5
    )
    np.testing.assert_array_equal(point / point.sum(), [0.0, 1.0])
    with pytest.raises(ValueError, match="impossible"):
        update(a, b, [0, 1, 0], 0.5)
    for bad in (np.zeros(3), [1, 0, -1], [1, 0, np.nan], [1, 1, 2]):
        with pytest.raises(ValueError, match="invalid"):
            update(bad, b, n, 0.5)


def test_fractional_counts_match_direct_integration():
    # Conditional density proportional to an incoming likelihood theta[0]^0.5.
    a, b = np.array([2.0, 3.0]), np.array([6.0, 2.0])

    def density(x):
        return np.sqrt(x) * sum(
            weight * x ** (p - 1) * (1 - x) ** (q - 1) / np.exp(betaln(p, q))
            for weight, (p, q) in ((0.8, b), (0.2, a))
        )

    norm = quad(density, 0, 1)[0]
    mean = quad(lambda x: x * density(x), 0, 1)[0] / norm
    var = quad(lambda x: (x - mean) ** 2 * density(x), 0, 1)[0] / norm
    projected = update(a, b, [0.5, 0], 0.2)
    p = projected[0] / projected.sum()
    np.testing.assert_allclose(
        [p, p * (1 - p) / (1 + projected.sum())], [mean, var], atol=1e-10
    )

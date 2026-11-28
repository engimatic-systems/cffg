"""D6d. Finite-population evidence and a logit-space resample–move example."""

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit, log_expit, logsumexp


def integrated_moment(successes, misses, degree=0):
    """Exact integral under Beta(2, 2) for success probability 4 theta (1-theta)."""
    result = Fraction()
    for j in range(misses + 1):
        exponent = successes + j
        factor = (-1) ** j * math.comb(misses, j) * 4**exponent
        beta = Fraction(
            6 * math.factorial(1 + degree + exponent) * math.factorial(1 + exponent),
            math.factorial(3 + degree + 2 * exponent),
        )
        result += factor * beta
    return result


def likelihoods(z):
    # The failure probability is tanh(z/2)^2, exactly zero at z=0.
    success = np.log(np.asarray(4, dtype=z.dtype)) + log_expit(z) + log_expit(-z)
    with np.errstate(divide="ignore"):
        miss = 2 * np.log(np.abs(np.tanh(z / 2)))
    return success, miss


def log_target(z, successes, misses):
    # The Beta(2,2) density contributes one copy of theta(1-theta);
    # transforming from theta to z contributes another identical Jacobian.
    result = math.log(6) + 2 * (log_expit(z) + log_expit(-z))
    success, miss = likelihoods(z)
    if successes:
        result = result + successes * success
    if misses:
        result = result + misses * miss
    return result


def stratified(weights, rng):
    count = len(weights)
    cdf = np.cumsum(weights, dtype=np.float64)
    cdf /= cdf[-1]
    points = (np.arange(count) + rng.random(count)) / count
    return np.searchsorted(cdf, points, side="right")


def condition(log_weights, scores):
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise FloatingPointError("invalid likelihood evaluation")
    updated = log_weights + scores
    log_increment = logsumexp(updated)
    if np.isneginf(log_increment):
        raise ValueError("sample population has zero likelihood")
    return updated - log_increment, log_increment


def resample_move(observations, n, dtype, rng):
    theta = rng.beta(2, 2, size=n)
    z = (np.log(theta) - np.log1p(-theta)).astype(dtype)
    weights = np.full(n, -math.log(n))
    successes = misses = 0
    log_evidence = 0.0
    copies = 0
    for observed in observations:
        if observed is None:
            continue
        success, miss = likelihoods(z)
        weights, increment = condition(weights, success if observed else miss)
        log_evidence += increment
        successes += int(observed)
        misses += int(not observed)
        if -logsumexp(2 * weights) < math.log(0.8 * n):
            z = z[stratified(np.exp(weights), rng)]
            weights.fill(-math.log(n))
            copies += 1
        proposal = z + rng.normal(size=n).astype(dtype)
        log_ratio = log_target(proposal, successes, misses) - log_target(
            z, successes, misses
        )
        # Zero-weight particles can sit outside target support. Give them the
        # identity kernel there; their contribution to the measure is zero.
        accepted = np.isfinite(weights) & (np.log(rng.random(n)) < log_ratio)
        z = np.where(accepted, proposal, z)
    return np.exp(log_evidence), expit(z), np.exp(weights), copies


@pytest.mark.parametrize(("dtype", "n"), [(np.float32, 128), (np.float64, 1280)])
def test_particle_evidence_and_nonidentifiability(dtype, n):
    trajectory = [1, 1, 0, None, 0, 0, 0]
    exact = integrated_moment(2, 4)
    second = integrated_moment(2, 4, 2) / exact
    assert integrated_moment(2, 4, 1) == exact / 2
    estimates = []
    resamples = 0
    for seed in range(48):
        evidence, theta, w, copies = resample_move(
            trajectory, n, dtype, np.random.default_rng(seed)
        )
        assert theta.dtype == dtype
        estimates.append([evidence, w @ theta, w @ (theta**2), w @ (theta > 0.5)])
        resamples += copies
    estimates = np.array(estimates)
    expected = [float(exact), 0.5, float(second), 0.5]
    # Monte Carlo qualification on this fixed seed panel. This diagnoses
    # finite-population calibration and cannot prove estimator unbiasedness.
    error = np.abs(estimates.mean(axis=0) - expected)
    stderr = estimates.std(axis=0, ddof=1) / np.sqrt(len(estimates))
    assert (error <= 6 * stderr).all()
    assert (error <= [0.05 * float(exact), 0.03, 0.03, 0.05]).all()
    assert 0 < resamples < 48 * 6


def test_transformed_density_and_mh_balance():
    z1, z2 = np.array(-1.8), np.array(0.7)

    def acceptance(x, y):
        return min(1.0, np.exp(log_target(y, 1, 2) - log_target(x, 1, 2)))

    np.testing.assert_allclose(
        np.exp(log_target(z1, 1, 2)) * acceptance(z1, z2),
        np.exp(log_target(z2, 1, 2)) * acceptance(z2, z1),
    )
    transformed, _ = quad(
        lambda z: np.exp(log_target(np.asarray(z), 1, 2)), -np.inf, np.inf
    )
    np.testing.assert_allclose(transformed, float(integrated_moment(1, 2)), rtol=1e-11)


def test_sampling_failure_and_missing_observation():
    with pytest.raises(ValueError, match="population"):
        condition(np.log([0.5, 0.5]), likelihoods(np.zeros(2))[1])
    # The population misses the positive-likelihood part of the prior.
    assert integrated_moment(0, 1) == Fraction(1, 5)
    with pytest.raises(FloatingPointError):
        condition(np.log([0.5, 0.5]), [np.nan, 0])

    def results(events):
        return resample_move(events, 32, np.float32, np.random.default_rng(53))

    for x, y in zip(results([1, 0, 1]), results([None, 1, None, 0, 1]), strict=True):
        np.testing.assert_array_equal(x, y)


def test_resampling_branch_must_share_the_same_move_for_evidence_proof():
    # Two particles, prior odds 1:3, first potentials L(0)=1 and L(1)=2.
    # Posterior mass on 1 is 2/5. The symmetric flip proposes the other
    # state; Metropolis accepts with 2/3 from 0 and with 1 from 1.
    estimates = [Fraction(), Fraction()]
    for ones in range(3):
        prob = (
            math.comb(2, ones) * Fraction(1, 4) ** ones * Fraction(3, 4) ** (2 - ones)
        )
        normalizer = Fraction(2 + ones, 2)
        fraction_one = Fraction(2 * ones, 2 + ones)
        moved = (1 - fraction_one) * Fraction(2, 3)
        estimates[0] += prob * normalizer * moved
        estimates[1] += prob * normalizer * (moved if ones == 1 else fraction_one)
    assert estimates == [Fraction(1, 2), Fraction(1, 4)]


def test_prior_predictive_calibration():
    # The true draw should lie below the estimated posterior median half of
    # the time over new data sets, including the two indistinguishable modes.
    data = np.random.default_rng(71)
    below_median = 0
    ranks = []
    for seed in range(128):
        true_theta = data.beta(2, 2)
        observations = data.binomial(1, 4 * true_theta * (1 - true_theta), size=6)
        _, theta, w, _ = resample_move(
            observations, 256, np.float64, np.random.default_rng(seed)
        )
        rank = w @ (theta <= true_theta)
        below_median += rank < 0.5
        ranks.append(rank)
    assert abs(below_median - 64) < 4 * np.sqrt(128 / 4)
    assert abs(np.mean(ranks) - 0.5) < 4 / np.sqrt(12 * 128)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nonlinear_observed_gaussian_likelihood_against_quadrature(dtype):
    from scipy.special import roots_jacobi

    from cffg.finite_hypotheses import FiniteHypotheses

    nodes, weights = roots_jacobi(64, 1, 1)
    theta = ((nodes + 1) / 2)[:, None]
    sigma = 0.18
    y = np.array([[0.28], [0.69]])
    x = np.array([1.0, 2.0])
    means = x[:, None, None] * theta**2
    bank = FiniteHypotheses(
        theta, [[sigma**2]], prior=weights / weights.sum(), dtype=dtype
    )
    bank.update(means, y)

    def integrand(p):
        residual = (y[:, 0] - x * p**2) / sigma
        return (
            6
            * p
            * (1 - p)
            * np.exp(-0.5 * (residual @ residual))
            / (2 * math.pi * sigma**2)
        )

    evidence = quad(integrand, 0, 1)[0]
    mean = quad(lambda p: p * integrand(p), 0, 1)[0] / evidence
    np.testing.assert_allclose(
        [bank.log_evidence, bank.weights @ theta[:, 0]],
        [math.log(evidence), mean],
        atol=64 * np.finfo(dtype).eps,
        rtol=0,
    )

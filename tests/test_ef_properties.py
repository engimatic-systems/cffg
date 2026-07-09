"""Property tests for cffg.ef (cff-8s6a): scipy as an independent oracle.

`tests/test_ef.py` already pins moments/entropy/KL against numerical
trapezoid quadrature of the density and checks moments-are-gradients via
finite differences on `log_partition`. This file does not repeat that.
Instead every closed form here is cross-checked against `scipy.stats` /
`scipy.special` — a second, independently-implemented library — over a
grid that spans small, unit, moderate, and very large (but strictly
proper) shape/rate/a/b, plus algebraic identities quadrature can't touch:
exact conjugate updates, the `is_proper` boundary, and batch broadcasting.

Tolerance policy: where a closed form here meets a closed form in ef.py
(digamma/gammaln/betaln formulas, entropy, log_partition, the KL formulas
re-derived below from Wikipedia's closed forms), tolerances are tight
(rtol/atol ~1e-8 to 1e-10) — both sides are exact algebra, so any looser
tolerance would be hiding a real bug. Where the oracle is
`scipy.stats.*.expect` (genuine numerical quadrature via scipy.integrate),
tolerances are loosened to rtol=1e-5 and the reason is called out inline.
"""

import itertools

import numpy as np
import pytest
from scipy import special, stats

from cffg import ef

# shape/rate/a/b grid: small, unit, moderate, large, huge — all strictly
# proper (>0) so quadrature-based oracles and the closed forms both apply
# without hitting the is_proper boundary (tested separately, exactly at
# the natural-parameter vacuous point).
PARAM_GRID = [0.05, 0.5, 1.0, 3.7, 40.0, 500.0]
GAMMA_CASES = list(itertools.product(PARAM_GRID, PARAM_GRID))  # (shape, rate)
BETA_CASES = list(itertools.product(PARAM_GRID, PARAM_GRID))  # (a, b)

# scipy.stats.*.expect() does generic scipy.integrate.quad over the
# support, which loses accuracy near the integrable endpoint singularities
# that shape/a/b < 1 create (same caveat test_ef.py notes for its own
# quadrature cases) — restrict this spot-check subset to params >= 1.
GAMMA_EXPECT_SUBSET = [(1.0, 1.0), (3.7, 1.0), (3.7, 40.0), (40.0, 3.7), (500.0, 500.0)]
BETA_EXPECT_SUBSET = [(1.0, 1.0), (3.7, 1.0), (3.7, 40.0), (40.0, 3.7), (500.0, 500.0)]

# a handful of diverse "second" params to pair against the full grid for
# O(n) KL tests instead of O(n^2); chosen to span the same small/unit/huge
# range rather than a slice of GAMMA_CASES (which would coincidentally
# hand back only the diagonal a==b pairs at stride 7).
GAMMA_KL_PARTNERS = [(0.05, 500.0), (0.5, 40.0), (1.0, 1.0), (3.7, 3.7), (40.0, 0.5), (500.0, 0.05)]
BETA_KL_PARTNERS = [(0.05, 500.0), (0.5, 40.0), (1.0, 1.0), (3.7, 3.7), (40.0, 0.5), (500.0, 0.05)]


# --- independent KL formulas -------------------------------------------------
#
# Coded fresh from scipy.special (Wikipedia's closed forms for the Gamma
# and Beta KL divergence), not derived from ef.py's generic
# EFFamily.kl = (η1-η2)·∇A(η1) - A(η1) + A(η2). A bug shared between
# ef.py's mean_stats and log_partition could otherwise cancel inside that
# generic formula and hide itself from a self-consistency check.


def gamma_kl_scipy(a1, b1, a2, b2):
    """KL(Gamma(a1,b1) || Gamma(a2,b2)), shape-rate parameterization."""
    return (
        (a1 - a2) * special.digamma(a1)
        - special.gammaln(a1)
        + special.gammaln(a2)
        + a2 * np.log(b1 / b2)
        + a1 * (b2 - b1) / b1
    )


def beta_kl_scipy(a1, b1, a2, b2):
    """KL(Beta(a1,b1) || Beta(a2,b2))."""
    return (
        special.betaln(a2, b2)
        - special.betaln(a1, b1)
        + (a1 - a2) * special.digamma(a1)
        + (b1 - b2) * special.digamma(b1)
        + (a2 - a1 + b2 - b1) * special.digamma(a1 + b1)
    )


# ---------------------------------------------------------------------------
# 1. Moments vs scipy.stats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape,rate", GAMMA_CASES)
def test_gamma_mean_matches_scipy(shape, rate):
    eta = ef.GAMMA.to_natural(shape, rate)
    ref = stats.gamma(a=shape, scale=1.0 / rate).mean()
    np.testing.assert_allclose(ef.GAMMA.mean(eta), ref, rtol=1e-10)
    np.testing.assert_allclose(ef.GAMMA.mean_stats(eta)[..., 1], ref, rtol=1e-10)


@pytest.mark.parametrize("shape,rate", GAMMA_CASES)
def test_gamma_elogx_matches_digamma_formula(shape, rate):
    """E[log x] = psi(shape) - ln(rate); scipy.special.digamma is a
    distinct implementation from ef.py's vendored recurrence+asymptotic
    psi, so this pins both the mean_stats formula and the vendored psi."""
    eta = ef.GAMMA.to_natural(shape, rate)
    ref = special.digamma(shape) - np.log(rate)
    np.testing.assert_allclose(ef.GAMMA.mean_stats(eta)[..., 0], ref, rtol=1e-10)


@pytest.mark.parametrize("shape,rate", GAMMA_EXPECT_SUBSET)
def test_gamma_moments_match_scipy_expect_spotcheck(shape, rate):
    """Validates the digamma-formula oracle itself against a fully
    independent numerical quadrature of the scipy density (not ef.py's
    formula, not the digamma closed form used above) — looser tolerance
    because scipy.integrate.quad under `expect` carries its own error."""
    eta = ef.GAMMA.to_natural(shape, rate)
    g = stats.gamma(a=shape, scale=1.0 / rate)
    ex = g.expect(lambda x: x)
    elogx = g.expect(lambda x: np.log(x))
    mstats = ef.GAMMA.mean_stats(eta)
    np.testing.assert_allclose(ef.GAMMA.mean(eta), ex, rtol=1e-5)
    np.testing.assert_allclose(mstats[..., 1], ex, rtol=1e-5)
    np.testing.assert_allclose(mstats[..., 0], elogx, rtol=1e-5)


@pytest.mark.parametrize("a,b", BETA_CASES)
def test_beta_mean_matches_scipy(a, b):
    eta = ef.BETA.to_natural(a, b)
    ref = stats.beta(a, b).mean()
    np.testing.assert_allclose(ef.BETA.mean(eta), ref, rtol=1e-10)


@pytest.mark.parametrize("a,b", BETA_CASES)
def test_beta_logx_and_log1mx_match_digamma_formula(a, b):
    eta = ef.BETA.to_natural(a, b)
    mstats = ef.BETA.mean_stats(eta)
    np.testing.assert_allclose(mstats[..., 0], special.digamma(a) - special.digamma(a + b), rtol=1e-10)
    np.testing.assert_allclose(mstats[..., 1], special.digamma(b) - special.digamma(a + b), rtol=1e-10)


@pytest.mark.parametrize("a,b", BETA_EXPECT_SUBSET)
def test_beta_moments_match_scipy_expect_spotcheck(a, b):
    eta = ef.BETA.to_natural(a, b)
    be = stats.beta(a, b)
    ex = be.expect(lambda x: x)
    elogx = be.expect(lambda x: np.log(x))
    elog1mx = be.expect(lambda x: np.log(1 - x))
    mstats = ef.BETA.mean_stats(eta)
    np.testing.assert_allclose(ef.BETA.mean(eta), ex, rtol=1e-5)
    np.testing.assert_allclose(mstats[..., 0], elogx, rtol=1e-5)
    np.testing.assert_allclose(mstats[..., 1], elog1mx, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. Entropy vs scipy.stats (both distributions have closed-form entropy in
#    scipy — this is closed-form vs closed-form, not vs quadrature)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape,rate", GAMMA_CASES)
def test_gamma_entropy_matches_scipy(shape, rate):
    eta = ef.GAMMA.to_natural(shape, rate)
    ref = stats.gamma(a=shape, scale=1.0 / rate).entropy()
    np.testing.assert_allclose(ef.GAMMA.entropy(eta), ref, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("a,b", BETA_CASES)
def test_beta_entropy_matches_scipy(a, b):
    eta = ef.BETA.to_natural(a, b)
    ref = stats.beta(a, b).entropy()
    np.testing.assert_allclose(ef.BETA.entropy(eta), ref, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# 3. log_partition vs scipy.special closed forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape,rate", GAMMA_CASES)
def test_gamma_log_partition_matches_gammaln(shape, rate):
    eta = ef.GAMMA.to_natural(shape, rate)
    ref = special.gammaln(shape) - shape * np.log(rate)
    np.testing.assert_allclose(ef.GAMMA.log_partition(eta), ref, rtol=1e-10)


@pytest.mark.parametrize("a,b", BETA_CASES)
def test_beta_log_partition_matches_betaln(a, b):
    eta = ef.BETA.to_natural(a, b)
    ref = special.betaln(a, b)
    np.testing.assert_allclose(ef.BETA.log_partition(eta), ref, rtol=1e-10)


# ---------------------------------------------------------------------------
# 4. KL vs an independent scipy.special formulation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a1,b1", GAMMA_CASES)
@pytest.mark.parametrize("a2,b2", GAMMA_KL_PARTNERS)
def test_gamma_kl_matches_independent_formula(a1, b1, a2, b2):
    eta1, eta2 = ef.GAMMA.to_natural(a1, b1), ef.GAMMA.to_natural(a2, b2)
    ref = gamma_kl_scipy(a1, b1, a2, b2)
    np.testing.assert_allclose(ef.GAMMA.kl(eta1, eta2), ref, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("a1,b1", BETA_CASES)
@pytest.mark.parametrize("a2,b2", BETA_KL_PARTNERS)
def test_beta_kl_matches_independent_formula(a1, b1, a2, b2):
    eta1, eta2 = ef.BETA.to_natural(a1, b1), ef.BETA.to_natural(a2, b2)
    ref = beta_kl_scipy(a1, b1, a2, b2)
    np.testing.assert_allclose(ef.BETA.kl(eta1, eta2), ref, rtol=1e-8, atol=1e-8)


def test_kl_self_is_zero_for_grid():
    for shape, rate in GAMMA_CASES:
        eta = ef.GAMMA.to_natural(shape, rate)
        np.testing.assert_allclose(ef.GAMMA.kl(eta, eta), 0.0, atol=1e-9)
    for a, b in BETA_CASES:
        eta = ef.BETA.to_natural(a, b)
        np.testing.assert_allclose(ef.BETA.kl(eta, eta), 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 5. Conjugate identities: natural-parameter addition == exact Bayes
# ---------------------------------------------------------------------------

GAMMA_POISSON_CASES = [
    ((2.0, 1.0), [3, 0, 5]),
    ((0.5, 0.05), [0, 0, 0, 1]),
    ((40.0, 3.7), [100, 250, 7]),
    ((500.0, 500.0), []),
    ((3.7, 40.0), [1]),
]


@pytest.mark.parametrize("prior_std,counts", GAMMA_POISSON_CASES)
def test_gamma_poisson_conjugate_update_is_exact(prior_std, counts):
    """Gamma(alpha0,beta0) prior times Poisson(x_i|lambda) likelihoods ==
    Gamma(alpha0+sum(x), beta0+n) exactly, via plain natural-parameter
    addition — the identity the future EF arena's message products rely
    on being exact, not approximate."""
    counts = np.asarray(counts, dtype=np.float64)
    prior = ef.GAMMA.to_natural(*prior_std)
    post = prior + sum((ef.poisson_likelihood_message(x) for x in counts), np.zeros(2))
    alpha, beta = ef.GAMMA.from_natural(post)
    np.testing.assert_allclose(alpha, prior_std[0] + counts.sum(), rtol=1e-12)
    np.testing.assert_allclose(beta, prior_std[1] + len(counts), rtol=1e-12)


BETA_BERNOULLI_CASES = [
    ((1.0, 1.0), [1, 1, 0, 1]),
    ((0.5, 0.05), [0]),
    ((40.0, 3.7), [1, 1, 1, 1, 1, 0, 0]),
    ((500.0, 500.0), []),
    ((3.7, 40.0), [0, 1]),
]


@pytest.mark.parametrize("prior_std,ys", BETA_BERNOULLI_CASES)
def test_beta_bernoulli_conjugate_update_is_exact(prior_std, ys):
    """Beta(a0,b0) prior times Bernoulli(y_i|p) likelihoods ==
    Beta(a0+sum(y), b0+sum(1-y)) exactly."""
    ys = np.asarray(ys, dtype=np.float64)
    prior = ef.BETA.to_natural(*prior_std)
    post = prior + sum((ef.bernoulli_likelihood_message(y) for y in ys), np.zeros(2))
    a, b = ef.BETA.from_natural(post)
    np.testing.assert_allclose(a, prior_std[0] + ys.sum(), rtol=1e-12)
    np.testing.assert_allclose(b, prior_std[1] + (1.0 - ys).sum(), rtol=1e-12)


# ---------------------------------------------------------------------------
# 6. is_proper: boundary behavior at the vacuous point and bare likelihoods
# ---------------------------------------------------------------------------


def test_gamma_vacuous_natural_params_are_improper():
    """eta = 0 in Gamma-natural coordinates is (alpha,beta) = (1, 0): a
    shape-1 density with zero rate has no normalizing mass — improper."""
    eta = np.zeros(2)
    alpha, beta = ef.GAMMA.from_natural(eta)
    assert (alpha, beta) == (1.0, 0.0)
    assert not bool(ef.GAMMA.is_proper(eta))


def test_beta_vacuous_natural_params_are_proper():
    """eta = 0 in Beta-natural coordinates is (a,b) = (1, 1): Uniform(0,1)
    — proper, unlike the Gamma vacuous point."""
    eta = np.zeros(2)
    a, b = ef.BETA.from_natural(eta)
    assert (a, b) == (1.0, 1.0)
    assert bool(ef.BETA.is_proper(eta))


@pytest.mark.parametrize("x", [0, 1, 5, 100])
def test_bare_poisson_likelihood_message_is_proper(x):
    """A single Poisson count's likelihood message (x, -1) is, standing
    alone, a proper Gamma(x+1, 1) — a bare count observation already
    defines a normalizable density over lambda, even with no explicit
    prior message added first."""
    eta = ef.poisson_likelihood_message(x)
    alpha, beta = ef.GAMMA.from_natural(eta)
    np.testing.assert_allclose((alpha, beta), (x + 1.0, 1.0))
    assert bool(ef.GAMMA.is_proper(eta))


# ---------------------------------------------------------------------------
# 7. Batch / broadcast: every table function accepts stacked eta [N,2]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family,cases", [(ef.GAMMA, GAMMA_CASES[:8]), (ef.BETA, BETA_CASES[:8])])
def test_family_functions_broadcast_over_leading_batch_dim(family, cases):
    etas = np.stack([family.to_natural(*std) for std in cases], axis=0)
    n = len(cases)
    assert etas.shape == (n, 2)

    assert family.log_partition(etas).shape == (n,)
    assert family.mean_stats(etas).shape == (n, 2)
    assert family.mean(etas).shape == (n,)
    assert family.entropy(etas).shape == (n,)
    assert family.is_proper(etas).shape == (n,)

    a, b = family.from_natural(etas)
    assert a.shape == (n,) and b.shape == (n,)

    other = np.roll(etas, 1, axis=0)
    assert family.kl(etas, other).shape == (n,)

    # shape-correct isn't enough: each batched row must equal the scalar call
    for i, std in enumerate(cases):
        eta_i = family.to_natural(*std)
        np.testing.assert_allclose(family.log_partition(etas)[i], family.log_partition(eta_i))
        np.testing.assert_allclose(family.mean_stats(etas)[i], family.mean_stats(eta_i))
        np.testing.assert_allclose(family.entropy(etas)[i], family.entropy(eta_i))
        np.testing.assert_allclose(family.kl(etas, other)[i], family.kl(eta_i, other[i]))


def test_poisson_bernoulli_messages_and_energies_broadcast():
    xs = np.array([0.0, 3.0, 7.0, 100.0])
    msg = ef.poisson_likelihood_message(xs)
    assert msg.shape == (4, 2)
    eta_q = ef.GAMMA.to_natural(5.0, 2.0)
    energy = ef.poisson_obs_energy(eta_q, xs)
    assert energy.shape == (4,)
    for i in range(4):
        np.testing.assert_allclose(energy[i], ef.poisson_obs_energy(eta_q, xs[i]))

    ys = np.array([0.0, 1.0, 1.0, 0.0])
    bmsg = ef.bernoulli_likelihood_message(ys)
    assert bmsg.shape == (4, 2)
    eta_qb = ef.BETA.to_natural(3.0, 4.0)
    benergy = ef.bernoulli_obs_energy(eta_qb, ys)
    assert benergy.shape == (4,)
    for i in range(4):
        np.testing.assert_allclose(benergy[i], ef.bernoulli_obs_energy(eta_qb, ys[i]))

    # batched eta_q paired with batched x too — [N,2] belief per observation
    eta_q_batch = np.stack([ef.GAMMA.to_natural(a, b) for a, b in [(2, 1), (5, 2), (9, 3), (1, 1)]])
    energy_batch = ef.poisson_obs_energy(eta_q_batch, xs)
    assert energy_batch.shape == (4,)
    for i in range(4):
        np.testing.assert_allclose(energy_batch[i], ef.poisson_obs_energy(eta_q_batch[i], xs[i]))

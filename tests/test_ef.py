"""EF family tables (cff-8s6a): every closed form checked against quadrature.

The contract is 'moments are gradients of A': we verify ∇A numerically, all
moments/entropies against direct integration of the density, KL against its
definition, and the conjugate-update semantics (natural-parameter addition
== exact Bayes) that the future EF arena relies on.
"""

import numpy as np
import pytest

from cffg import ef

GRID = 200_001


def gamma_pdf(x, shape, rate):
    return np.exp(
        (shape - 1) * np.log(x) - rate * x + shape * np.log(rate) - ef.gammaln(shape)
    )


def beta_pdf(x, a, b):
    return np.exp(
        (a - 1) * np.log(x) + (b - 1) * np.log(1 - x)
        - (ef.gammaln(a) + ef.gammaln(b) - ef.gammaln(a + b))
    )


# quadrature cases keep shape > 1 (same endpoint-singularity caveat as Beta)
CASES_GAMMA = [(2.0, 1.0), (1.3, 3.5), (9.3, 0.4)]
# quadrature cases keep a,b > 1 (integrable endpoint singularities defeat
# trapezoid); sub-1 parameters are covered by the gradient and KL tests
CASES_BETA = [(2.0, 2.0), (1.5, 4.0), (7.5, 1.3)]


@pytest.mark.parametrize("shape,rate", CASES_GAMMA)
def test_gamma_moments_match_quadrature(shape, rate):
    eta = ef.GAMMA.to_natural(shape, rate)
    hi = (shape / rate) + 12 * np.sqrt(shape) / rate
    x = np.linspace(1e-9, hi, GRID)
    p = gamma_pdf(x, shape, rate)
    dx = x[1] - x[0]
    stats = ef.GAMMA.mean_stats(eta)
    np.testing.assert_allclose(stats[..., 0], np.trapezoid(p * np.log(x), dx=dx), atol=2e-4)
    np.testing.assert_allclose(stats[..., 1], np.trapezoid(p * x, dx=dx), atol=2e-4)
    np.testing.assert_allclose(ef.GAMMA.mean(eta), shape / rate, rtol=1e-12)
    ent = -np.trapezoid(p * np.log(np.maximum(p, 1e-300)), dx=dx)
    np.testing.assert_allclose(ef.GAMMA.entropy(eta), ent, atol=2e-3)


@pytest.mark.parametrize("a,b", CASES_BETA)
def test_beta_moments_match_quadrature(a, b):
    eta = ef.BETA.to_natural(a, b)
    x = np.linspace(1e-9, 1 - 1e-9, GRID)
    p = beta_pdf(x, a, b)
    dx = x[1] - x[0]
    stats = ef.BETA.mean_stats(eta)
    np.testing.assert_allclose(stats[..., 0], np.trapezoid(p * np.log(x), dx=dx), atol=2e-3)
    np.testing.assert_allclose(stats[..., 1], np.trapezoid(p * np.log(1 - x), dx=dx), atol=2e-3)
    np.testing.assert_allclose(ef.BETA.mean(eta), a / (a + b), rtol=1e-12)
    ent = -np.trapezoid(p * np.log(np.maximum(p, 1e-300)), dx=dx)
    np.testing.assert_allclose(ef.BETA.entropy(eta), ent, atol=2e-3)


@pytest.mark.parametrize("family,std", [(ef.GAMMA, (3.0, 2.0)), (ef.BETA, (2.5, 4.0))])
def test_moments_are_gradients_of_log_partition(family, std):
    eta = family.to_natural(*std)
    eps = 1e-6
    for i in range(family.n_params):
        d = np.zeros(family.n_params)
        d[i] = eps
        fd = (family.log_partition(eta + d) - family.log_partition(eta - d)) / (2 * eps)
        np.testing.assert_allclose(family.mean_stats(eta)[..., i], fd, rtol=1e-5)


@pytest.mark.parametrize("family,p1,p2", [
    (ef.GAMMA, (2.0, 1.0), (3.5, 0.7)),
    (ef.BETA, (2.0, 2.0), (1.2, 5.0)),
])
def test_kl_matches_quadrature(family, p1, p2):
    eta1, eta2 = family.to_natural(*p1), family.to_natural(*p2)
    if family is ef.GAMMA:
        hi = max(p1[0] / p1[1], p2[0] / p2[1]) * 12
        x = np.linspace(1e-9, hi, GRID)
        q1, q2 = gamma_pdf(x, *p1), gamma_pdf(x, *p2)
    else:
        x = np.linspace(1e-9, 1 - 1e-9, GRID)
        q1, q2 = beta_pdf(x, *p1), beta_pdf(x, *p2)
    dx = x[1] - x[0]
    ref = np.trapezoid(q1 * (np.log(np.maximum(q1, 1e-300)) - np.log(np.maximum(q2, 1e-300))), dx=dx)
    np.testing.assert_allclose(family.kl(eta1, eta2), ref, atol=3e-3)
    np.testing.assert_allclose(family.kl(eta1, eta1), 0.0, atol=1e-12)


def test_conjugate_updates_are_natural_addition():
    """Products are sums == exact Bayes: a Gamma(α,β) prior times Poisson
    likelihoods for counts x_i is Gamma(α + Σx, β + n); Beta-Bernoulli ditto.
    This identity is what makes the future EF arena's message products
    correct by construction."""
    prior = ef.GAMMA.to_natural(2.0, 1.0)
    counts = np.array([3, 0, 5])
    post = prior + sum(ef.poisson_likelihood_message(x) for x in counts)
    a, b = ef.GAMMA.from_natural(post)
    assert (a, b) == (2.0 + counts.sum(), 1.0 + len(counts))

    prior_b = ef.BETA.to_natural(1.0, 1.0)
    ys = np.array([1, 1, 0, 1])
    post_b = prior_b + sum(ef.bernoulli_likelihood_message(y) for y in ys)
    a2, b2 = ef.BETA.from_natural(post_b)
    assert (a2, b2) == (1.0 + ys.sum(), 1.0 + (1 - ys).sum())

    # the vacuous message (η = 0) is improper — Gamma(1, 0) has no mass;
    # any proper prior added to it restores propriety (products are sums)
    vacuous = np.zeros(2)
    assert not ef.GAMMA.is_proper(vacuous)
    assert ef.GAMMA.is_proper(vacuous + prior)


def test_observation_energies_close_the_ledger():
    """The conditioned-observation energies must equal −E_q[log lik] by
    quadrature — these are the FE ledger terms the EF factors will report."""
    eta = ef.GAMMA.to_natural(3.0, 2.0)
    x_obs = 4
    hi = 20.0
    lam = np.linspace(1e-9, hi, GRID)
    q = gamma_pdf(lam, 3.0, 2.0)
    dx = lam[1] - lam[0]
    loglik = x_obs * np.log(lam) - lam - ef.gammaln(x_obs + 1.0)
    ref = -np.trapezoid(q * loglik, dx=dx)
    np.testing.assert_allclose(ef.poisson_obs_energy(eta, x_obs), ref, atol=1e-3)

    eta_b = ef.BETA.to_natural(2.0, 5.0)
    p = np.linspace(1e-9, 1 - 1e-9, GRID)
    qb = beta_pdf(p, 2.0, 5.0)
    dxb = p[1] - p[0]
    ref1 = -np.trapezoid(qb * np.log(p), dx=dxb)
    np.testing.assert_allclose(ef.bernoulli_obs_energy(eta_b, 1), ref1, atol=1e-3)


def test_special_functions_match_jax():
    """The vendored ψ/lnΓ must agree with jax.scipy to float64 precision —
    they feed FE numbers."""
    import jax.numpy as jnp
    from jax.scipy.special import digamma as jdg, gammaln as jgl

    x = np.array([0.05, 0.3, 1.0, 2.5, 7.7, 42.0, 1234.5])
    np.testing.assert_allclose(ef.digamma(x), np.asarray(jdg(jnp.asarray(x))), rtol=1e-10)
    np.testing.assert_allclose(ef.gammaln(x), np.asarray(jgl(jnp.asarray(x))), rtol=1e-10)

"""JAX EF rules (cffg.efrules): parity with the NumPy contract (cffg.ef),
the closed Bethe ledger at conjugate fixed points, propriety loudness, and
the float32 case the robustness contract requires."""

import jax.numpy as jnp
import numpy as np
import pytest

from cffg import ef, efrules

GRID = np.array(
    [  # proper (α, β) / (a, b) pairs, from harsh-but-legal corners
        [0.05, 0.05],
        [0.5, 3.0],
        [1.0, 1.0],
        [3.7, 0.2],
        [40.0, 7.0],
        [500.0, 250.0],
    ]
)


@pytest.mark.parametrize("name", ["gamma", "beta"])
def test_jnp_tables_match_numpy(name):
    npf, jf = ef.FAMILIES[name], efrules.JFAMILIES[name]
    eta = npf.to_natural(GRID[:, 0], GRID[:, 1])
    for np_fn, j_fn in [
        (npf.mean_stats, jf.mean_stats),
        (npf.log_partition, jf.log_partition),
        (npf.entropy, jf.entropy),
        (npf.is_proper, jf.is_proper),
    ]:
        np.testing.assert_allclose(
            np.asarray(j_fn(jnp.asarray(eta))), np_fn(eta), rtol=1e-12, atol=1e-12
        )


def test_gamma_poisson_ledger_closes():
    """F_prior + F_obs + H_edge at the conjugate fixed point == −log p(x),
    the negative-binomial evidence — the tree identity BFE = −log Z."""
    a0, b0, x = 2.5, 1.5, 4.0
    eta_p = jnp.asarray(ef.GAMMA.to_natural(a0, b0))[None, :]
    lik = efrules.poisson_likelihood_message(jnp.array([x]), jnp.array([1.0]))
    total = (
        efrules.prior_energy(efrules.GAMMA, lik, eta_p)
        + efrules.poisson_obs_energy(eta_p, jnp.array([x]), jnp.array([1.0]))
        + efrules.edge_entropy(efrules.GAMMA, eta_p, lik)
    )
    from math import lgamma, log

    log_ev = (
        lgamma(a0 + x) - lgamma(a0) - lgamma(x + 1.0)
        + a0 * log(b0) - (a0 + x) * log(b0 + 1.0)
    )
    np.testing.assert_allclose(float(total[0]), -log_ev, rtol=1e-12)


def test_beta_bernoulli_ledger_closes():
    a0, b0, y = 2.0, 5.0, 1.0
    eta_p = jnp.asarray(ef.BETA.to_natural(a0, b0))[None, :]
    lik = efrules.bernoulli_likelihood_message(jnp.array([y]), jnp.array([1.0]))
    total = (
        efrules.prior_energy(efrules.BETA, lik, eta_p)
        + efrules.bernoulli_obs_energy(eta_p, jnp.array([y]), jnp.array([1.0]))
        + efrules.edge_entropy(efrules.BETA, eta_p, lik)
    )
    log_ev = np.log(a0 / (a0 + b0))  # p(y=1) = E[p] under the prior
    np.testing.assert_allclose(float(total[0]), -log_ev, rtol=1e-12)


def test_missing_observation_drops_out():
    """mask 0: vacuous message, energy telescopes to −H[μ_in] — exactly what
    the interior edge's +H cancels, so a missing node costs the ledger 0."""
    eta_p = jnp.asarray(ef.GAMMA.to_natural(3.0, 2.0))[None, :]
    x, mk = jnp.array([0.0]), jnp.array([0.0])
    assert np.all(np.asarray(efrules.poisson_likelihood_message(x, mk)) == 0.0)
    f_obs = efrules.poisson_obs_energy(eta_p, x, mk)
    h_edge = efrules.edge_entropy(efrules.GAMMA, eta_p, jnp.zeros((1, 2)))
    np.testing.assert_allclose(float(f_obs[0] + h_edge[0]), 0.0, atol=1e-12)


def test_zero_count_is_legal():
    """x = 0 (hostile input): the likelihood message (0, −1) is a proper
    Gamma(1, 1) increment; energies stay finite."""
    eta_p = jnp.asarray(ef.GAMMA.to_natural(2.0, 1.0))[None, :]
    x, mk = jnp.array([0.0]), jnp.array([1.0])
    lik = efrules.poisson_likelihood_message(x, mk)
    assert bool(efrules.GAMMA.is_proper(lik)[0])
    assert np.isfinite(float(efrules.poisson_obs_energy(eta_p, x, mk)[0]))


def test_improper_beliefs_are_plus_inf():
    """Vacuous Gamma belief (η = 0 ⇒ β = 0) is improper: every ledger term
    returns +inf — including edge entropy, so the total can't NaN-cancel."""
    vac = jnp.zeros((1, 2))
    assert not bool(efrules.GAMMA.is_proper(vac)[0])
    assert float(efrules.prior_energy(efrules.GAMMA, vac, vac)[0]) == np.inf
    assert float(efrules.edge_entropy(efrules.GAMMA, vac, vac)[0]) == np.inf
    assert float(efrules.equality_energy(efrules.GAMMA, jnp.zeros((1, 3, 2)))[0]) == np.inf
    # The vacuous Beta belief is Uniform(0,1): proper, entropy 0.
    assert bool(efrules.BETA.is_proper(vac)[0])
    np.testing.assert_allclose(float(efrules.edge_entropy(efrules.BETA, vac, vac)[0]), 0.0)


def test_equality_messages_sum_other_legs():
    eta = jnp.asarray(np.random.default_rng(0).normal(size=(4, 3, 2)))
    out = efrules.equality_messages(eta)
    np.testing.assert_allclose(
        np.asarray(out[:, 0]), np.asarray(eta[:, 1] + eta[:, 2]), rtol=1e-15
    )


def test_float32_stays_finite_and_close():
    """Consumers get JAX's default f32; jax.scipy special functions are
    dtype-aware, so rules must survive without the x64 flag the suite sets."""
    eta64 = jnp.asarray(ef.GAMMA.to_natural(GRID[:, 0], GRID[:, 1]))
    eta32 = eta64.astype(jnp.float32)
    for fn in (efrules.GAMMA.mean_stats, efrules.GAMMA.log_partition, efrules.GAMMA.entropy):
        out32 = fn(eta32)
        assert out32.dtype == jnp.float32
        assert np.all(np.isfinite(np.asarray(out32)))
        np.testing.assert_allclose(
            np.asarray(out32), np.asarray(fn(eta64)), rtol=2e-4, atol=2e-4
        )

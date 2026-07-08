"""Gaussian rule library vs moment-form ground truth (cff-4e5z).

Every message rule is checked against the covariance-form textbook result on
proper inputs, and against exact vacuous propagation on improper ones — the
property the information-form derivation was chosen for.
"""

import numpy as np
import pytest

from cffg import gauss

RNG = np.random.default_rng(7)


def spd(d):
    m = RNG.normal(size=(d, d))
    return m @ m.T + d * np.eye(d)


def info(mean, cov):
    lam = np.linalg.inv(cov)
    return lam @ mean, lam


DO, DI = 2, 3
A = RNG.normal(size=(1, DO, DI))
B = RNG.normal(size=(1, DO))
V = spd(DO)[None]
W = np.linalg.inv(V)


def test_forward_matches_moment_form():
    m_i, p_i = RNG.normal(size=DI), spd(DI)
    eta_i, lam_i = info(m_i, p_i)
    eta, lam = gauss.linear_forward(eta_i[None], lam_i[None], A, B, W)
    cov = np.linalg.inv(lam[0])
    np.testing.assert_allclose(cov, A[0] @ p_i @ A[0].T + V[0], rtol=1e-8)
    np.testing.assert_allclose(cov @ eta[0], A[0] @ m_i + B[0], rtol=1e-8)


def test_forward_vacuous_propagates():
    # Square invertible A: no information in ⇒ none out.
    a = RNG.normal(size=(1, DI, DI)) + 3 * np.eye(DI)
    v = spd(DI)[None]
    eta, lam = gauss.linear_forward(
        np.zeros((1, DI)), np.zeros((1, DI, DI)), a, np.zeros((1, DI)), np.linalg.inv(v)
    )
    assert np.abs(lam).max() < 1e-6 and np.abs(eta).max() < 1e-6


def test_backward_matches_moment_form():
    m_o, p_o = RNG.normal(size=DO), spd(DO)
    eta_o, lam_o = info(m_o, p_o)
    eta, lam = gauss.linear_backward(eta_o[None], lam_o[None], A, B, W)
    g = np.linalg.inv(V[0] + p_o)  # (V + Σ_o)⁻¹
    np.testing.assert_allclose(lam[0], A[0].T @ g @ A[0], rtol=1e-8)
    np.testing.assert_allclose(eta[0], A[0].T @ g @ (m_o - B[0]), rtol=1e-8, atol=1e-10)


def test_backward_vacuous_propagates():
    eta, lam = gauss.linear_backward(np.zeros((1, DO)), np.zeros((1, DO, DO)), A, B, W)
    assert np.abs(lam).max() < 1e-9 and np.abs(eta).max() < 1e-9


def test_observed_backward_is_the_clamp_limit():
    y = RNG.normal(size=(1, DO))
    eta, lam = gauss.linear_observed_backward(y, np.ones(1), A, B, W)
    # Same as backward with a huge-precision message at y.
    big = 1e9
    eta_l, lam_l = gauss.linear_backward(big * y, big * np.eye(DO)[None], A, B, W)
    np.testing.assert_allclose(lam[0], lam_l[0], rtol=1e-6)
    np.testing.assert_allclose(eta[0], eta_l[0], rtol=1e-6)
    # Missing observation = vacuous.
    eta0, lam0 = gauss.linear_observed_backward(y, np.zeros(1), A, B, W)
    assert np.abs(lam0).max() == 0 and np.abs(eta0).max() == 0


def test_equality_messages_sum_others():
    eta = RNG.normal(size=(1, 3, DI))
    lam = np.stack([spd(DI) for _ in range(3)])[None]
    to, tl = gauss.equality_messages(eta, lam)
    np.testing.assert_allclose(to[0, 0], eta[0, 1] + eta[0, 2], rtol=1e-12)
    np.testing.assert_allclose(tl[0, 2], lam[0, 0] + lam[0, 1], rtol=1e-12)


def test_prior_energy_zero_against_vacuous():
    # q = f when nothing else arrives: F[q] = E[log q − log f] = 0.
    m, v = RNG.normal(size=(1, DI)), spd(DI)[None]
    f = gauss.prior_energy(np.zeros((1, DI)), np.zeros((1, DI, DI)), m, np.linalg.inv(v))
    assert abs(float(f[0])) < 1e-6


def test_linear_energy_reduces_to_input_neg_entropy():
    # Vacuous out-message: q(y,x) = f(y|x) g(x), so F = E[log g] = −H[g].
    m_i, p_i = RNG.normal(size=DI), spd(DI)
    eta_i, lam_i = info(m_i, p_i)
    f = gauss.linear_energy(
        np.zeros((1, DO)), np.zeros((1, DO, DO)), eta_i[None], lam_i[None], A, B, W
    )
    neg_h = -0.5 * np.linalg.slogdet(2 * np.pi * np.e * p_i)[1]
    np.testing.assert_allclose(float(f[0]), neg_h, rtol=1e-6)


def test_single_node_tree_identity():
    # Prior N(m0, P0) + observed emission y = Hx + noise R: node energies plus
    # the one interior-edge entropy must equal −log p(y) exactly.
    d = 2
    m0, p0 = RNG.normal(size=d), spd(d)
    h = RNG.normal(size=(1, d, d))
    r = spd(d)[None]
    w = np.linalg.inv(r)
    y = RNG.normal(size=(1, d))

    eta0, lam0 = info(m0, p0)  # prior → x message
    eta_b, lam_b = gauss.linear_observed_backward(y, np.ones(1), h, np.zeros((1, d)), w)

    f_prior = gauss.prior_energy(eta_b, lam_b, m0[None], np.linalg.inv(p0)[None])
    f_obs = gauss.linear_observed_energy(
        eta0[None], lam0[None], y, np.ones(1), h, np.zeros((1, d)), w
    )
    h_edge = gauss.edge_entropy(lam0[None], lam_b)
    bfe = float(f_prior[0] + f_obs[0] + h_edge[0])

    s = h[0] @ p0 @ h[0].T + r[0]  # marginal likelihood N(y; Hm0, S)
    resid = y[0] - h[0] @ m0
    nll = 0.5 * (np.linalg.slogdet(2 * np.pi * s)[1] + resid @ np.linalg.inv(s) @ resid)
    np.testing.assert_allclose(bfe, nll, rtol=1e-7)


@pytest.mark.parametrize("d", [1, 3])
def test_moments_round_trip(d):
    m, p = RNG.normal(size=(1, d)), spd(d)[None]
    eta, lam = np.linalg.solve(p[0], m[0])[None], np.linalg.inv(p[0])[None]
    mean, cov = gauss.moments(eta, lam)
    np.testing.assert_allclose(mean[0], m[0], rtol=1e-6)
    np.testing.assert_allclose(cov[0], p[0], rtol=1e-6)

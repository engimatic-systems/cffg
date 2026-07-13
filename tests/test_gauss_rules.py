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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_forward_vacuous_is_scale_and_coordinate_invariant(dtype):
    """Regression cff-3hgi: no absolute ridge may manufacture precision.

    Noise precision spans 24 orders and the invertible coordinate map spans
    12, independently; the symbolic vacuous limit is exactly zero in both
    working dtypes.
    """
    pairs = [(wp, ap) for wp in (-12, 0, 12) for ap in (-6, 0, 6)]
    w = np.asarray([[[10.0**wp]] for wp, _ in pairs], dtype=dtype)
    a = np.asarray([[[10.0**ap]] for _, ap in pairs], dtype=dtype)
    z1 = np.zeros((len(pairs), 1), dtype=dtype)
    z2 = np.zeros((len(pairs), 1, 1), dtype=dtype)
    eta, lam = gauss.linear_forward(z1, z2, a, z1, w)
    np.testing.assert_array_equal(np.asarray(eta), 0.0)
    np.testing.assert_array_equal(np.asarray(lam), 0.0)


def test_forward_rank_deficient_vacuous_limit_is_finite_psd():
    """A rank-deficient map retains information only off its column space."""
    a = np.array([[[1.0], [0.0]]])
    w = np.array([np.diag([1e-8, 1e8])])
    eta, lam = gauss.linear_forward(
        np.zeros((1, 1)), np.zeros((1, 1, 1)), a, np.zeros((1, 2)), w
    )
    np.testing.assert_allclose(eta, 0.0, atol=0.0)
    np.testing.assert_allclose(lam[0], np.diag([0.0, 1e8]), rtol=1e-12)
    assert np.linalg.eigvalsh(np.asarray(lam[0])).min() >= 0.0


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


def test_backward_float32_preserves_weak_proper_precision():
    dtype = np.float32
    eta_o = np.zeros((1, 1), dtype=dtype)
    lam_o = np.full((1, 1, 1), 1e-6, dtype=dtype)
    eta, lam = gauss.linear_backward(
        eta_o,
        lam_o,
        np.ones((1, 1, 1), dtype=dtype),
        eta_o,
        np.ones((1, 1, 1), dtype=dtype),
    )
    np.testing.assert_allclose(eta, 0.0, atol=0.0)
    np.testing.assert_allclose(float(lam[0, 0, 0]), 1e-6, rtol=2e-3)


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


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[-1.0]]),
        np.zeros((1, 1)),
        np.array([[1.0, 0.25], [0.0, 1.0]]),
        np.diag([1.0, -1.0]),
    ],
)
def test_improper_precision_readout_and_entropy_are_loud(bad):
    eta = np.zeros(bad.shape[0])
    mean, cov = gauss.moments(eta, bad)
    assert np.isnan(np.asarray(mean)).all() and np.isnan(np.asarray(cov)).all()
    assert float(gauss._neg_entropy(bad)) == np.inf


def test_invalid_factor_precision_cannot_produce_finite_energy():
    """A strong incoming message must not hide a user-mutated negative W."""
    bad_w = np.array([[[-1.0]]])
    eta = np.zeros((1, 1))
    strong = np.array([[[3.0]]])
    prior = gauss.prior_energy(eta, strong, eta, bad_w)
    linear = gauss.linear_energy(
        eta, strong, eta, strong, np.ones((1, 1, 1)), eta, bad_w
    )
    observed = gauss.linear_observed_energy(
        eta, strong, eta, np.ones(1), np.ones((1, 1, 1)), eta, bad_w
    )
    assert np.isposinf(np.asarray(prior)).all()
    assert np.isposinf(np.asarray(linear)).all()
    assert np.isposinf(np.asarray(observed)).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mixed_scale_d100_readout_survives(dtype):
    """Hostile scale/dimension contract: d>=100 remains proper and finite."""
    d = 100
    diag = np.logspace(-12, 12, d, dtype=dtype)
    lam = np.diag(diag)[None]
    eta = np.linspace(-1.0, 1.0, d, dtype=dtype)[None] * diag[None]
    mean, cov = gauss.moments(eta, lam)
    assert np.isfinite(np.asarray(mean)).all()
    assert np.isfinite(np.asarray(cov)).all()
    np.testing.assert_allclose(np.diag(np.asarray(cov[0])), 1.0 / diag, rtol=2e-5)

    # The same uncomfortable mixed scale cannot inject information through
    # an invertible identity map with a vacuous input.
    z1 = np.zeros((1, d), dtype=dtype)
    z2 = np.zeros((1, d, d), dtype=dtype)
    _, lam_f = gauss.linear_forward(z1, z2, np.eye(d, dtype=dtype)[None], z1, lam)
    np.testing.assert_array_equal(np.asarray(lam_f), 0.0)

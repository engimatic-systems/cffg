"""Hybrid mixture rules (cffg.mixg) vs DERIVATIONS.md D3: K=1 degeneracy to
the plain Gaussian rules, 1-D quadrature oracles for every message and the
node energy, the backward weight trap, the ledger bound direction, f32."""

import jax.numpy as jnp
import numpy as np
import pytest

from cffg import gauss, mixg

# One member (M=1), K=2 regimes, 1-D everything: strong-enough separation
# that a wrong collapse is unmissable, mild enough that quadrature is easy.
A2 = np.array([[[[0.9]], [[-0.6]]]])  # [M, K, do, di]
B2 = np.array([[[0.3], [-1.1]]])  # [M, K, do]
W2 = np.array([[[[1.0 / 0.4]], [[1.0 / 1.7]]]])  # [M, K, do, do]
MU_E = np.log(np.array([[0.65, 0.35]]))  # [M, K] selector log-message
ETA_O, LAM_O = np.array([[1.2]]), np.array([[[2.5]]])  # proper mu_o
ETA_I, LAM_I = np.array([[-0.4]]), np.array([[[1.8]]])  # proper mu_i

YS = np.linspace(-25, 25, 8001)
XS = np.linspace(-25, 25, 8001)


def _norm(pdf, xs):
    z = np.trapezoid(pdf, xs)
    return pdf / z, z


def _moments(pdf, xs):
    p, _ = _norm(pdf, xs)
    m = np.trapezoid(p * xs, xs)
    v = np.trapezoid(p * (xs - m) ** 2, xs)
    return m, v


def _mu(eta, lam, xs):
    return np.exp(-0.5 * lam * xs**2 + eta * xs)


def _f_s(y, x, k):
    a, b, w = A2[0, k, 0, 0], B2[0, k, 0], W2[0, k, 0, 0]
    return np.sqrt(w / (2 * np.pi)) * np.exp(-0.5 * w * (y - a * x - b) ** 2)


# --- K=1 degeneracy -------------------------------------------------------------


def test_k1_reduces_to_linear_rules():
    a, b, w = A2[:, :1], B2[:, :1], W2[:, :1]
    le = np.zeros((1, 1))
    e_f, l_f = mixg.mixlin_forward(ETA_I, LAM_I, le, a, b, w)
    e_ref, l_ref = gauss.linear_forward(ETA_I, LAM_I, a[:, 0], b[:, 0], w[:, 0])
    np.testing.assert_allclose(e_f, e_ref, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(l_f, l_ref, rtol=1e-8, atol=1e-10)

    e_b, l_b = mixg.mixlin_backward(ETA_O, LAM_O, le, a, b, w)
    e_ref, l_ref = gauss.linear_backward(ETA_O, LAM_O, a[:, 0], b[:, 0], w[:, 0])
    np.testing.assert_allclose(e_b, e_ref, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(l_b, l_ref, rtol=1e-8, atol=1e-10)

    f = mixg.mixlin_energy(ETA_O, LAM_O, ETA_I, LAM_I, le, a, b, w)
    f_ref = gauss.linear_energy(ETA_O, LAM_O, ETA_I, LAM_I, a[:, 0], b[:, 0], w[:, 0])
    np.testing.assert_allclose(f, f_ref, rtol=1e-9)


def test_identical_components_collapse_exactly():
    a = np.repeat(A2[:, :1], 2, axis=1)
    b = np.repeat(B2[:, :1], 2, axis=1)
    w = np.repeat(W2[:, :1], 2, axis=1)
    e_f, l_f = mixg.mixlin_forward(ETA_I, LAM_I, MU_E, a, b, w)
    e_ref, l_ref = gauss.linear_forward(ETA_I, LAM_I, a[:, 0], b[:, 0], w[:, 0])
    np.testing.assert_allclose(e_f, e_ref, rtol=1e-8)
    np.testing.assert_allclose(l_f, l_ref, rtol=1e-8)
    # identical regimes explain data identically: selector message uniform
    logmu = mixg.mixlin_to_selector(ETA_O, LAM_O, ETA_I, LAM_I, a, b, w)
    np.testing.assert_allclose(logmu[0, 0], logmu[0, 1], atol=1e-10)


# --- quadrature oracles ----------------------------------------------------------


def test_selector_message_matches_quadrature():
    logmu = np.asarray(mixg.mixlin_to_selector(ETA_O, LAM_O, ETA_I, LAM_I, A2, B2, W2))[0]
    vals = []
    for k in range(2):
        fyx = _f_s(YS[:, None], XS[None, :], k)
        integ = fyx * _mu(ETA_O[0, 0], LAM_O[0, 0, 0], YS)[:, None] * _mu(
            ETA_I[0, 0], LAM_I[0, 0, 0], XS
        )[None, :]
        vals.append(np.trapezoid(np.trapezoid(integ, XS, axis=1), YS))
    ref = np.log(vals)
    np.testing.assert_allclose(logmu - logmu.max(), ref - ref.max(), rtol=1e-6, atol=1e-8)


def test_forward_collapse_matches_quadrature():
    e_f, l_f = (np.asarray(v)[0] for v in mixg.mixlin_forward(ETA_I, LAM_I, MU_E, A2, B2, W2))
    w_e = np.exp(MU_E[0]) / np.exp(MU_E[0]).sum()
    mix = sum(
        w_e[k]
        * _norm(
            np.trapezoid(
                _f_s(YS[:, None], XS[None, :], k) * _mu(ETA_I[0, 0], LAM_I[0, 0, 0], XS)[None, :],
                XS,
                axis=1,
            ),
            YS,
        )[0]
        for k in range(2)
    )
    m_ref, v_ref = _moments(mix, YS)
    np.testing.assert_allclose(e_f[0] / l_f[0, 0], m_ref, rtol=1e-6)
    np.testing.assert_allclose(1.0 / l_f[0, 0], v_ref, rtol=1e-6)


def test_backward_collapse_matches_quadrature():
    """The weight-trap oracle: the collapsed backward message must have the
    moments of Σ_s μ_e(s)·∫f_s·μ_o dy — component scales included."""
    e_b, l_b = (np.asarray(v)[0] for v in mixg.mixlin_backward(ETA_O, LAM_O, MU_E, A2, B2, W2))
    mix = sum(
        np.exp(MU_E[0, k])
        * np.trapezoid(
            _f_s(YS[:, None], XS[None, :], k) * _mu(ETA_O[0, 0], LAM_O[0, 0, 0], YS)[:, None],
            YS,
            axis=0,
        )
        for k in range(2)
    )
    m_ref, v_ref = _moments(mix, XS)
    np.testing.assert_allclose(e_b[0] / l_b[0, 0], m_ref, rtol=1e-6)
    np.testing.assert_allclose(1.0 / l_b[0, 0], v_ref, rtol=1e-6)


def test_backward_weights_are_not_selector_weights():
    """Regression for the weight trap: with strongly different |A_s| and V_s
    the naive collapse (selector weights only) is materially wrong."""
    a = np.array([[[[2.0]], [[0.4]]]])
    b = np.array([[[0.0], [0.0]]])
    w = np.array([[[[10.0]], [[0.5]]]])
    e_b, l_b = (np.asarray(v)[0] for v in mixg.mixlin_backward(ETA_O, LAM_O, np.log([[0.5, 0.5]]), a, b, w))

    def naive():
        comps = []
        for k in range(2):
            (e, l) = gauss.linear_backward(ETA_O, LAM_O, a[:, k], b[:, k], w[:, k])
            comps.append((np.asarray(e)[0, 0], np.asarray(l)[0, 0, 0]))
        m = sum(0.5 * e / l for e, l in comps)
        v = sum(0.5 * (1 / l + (e / l - m) ** 2) for e, l in comps)
        return m, v

    m_naive, v_naive = naive()
    m_rule = e_b[0] / l_b[0, 0]
    v_rule = 1.0 / l_b[0, 0]
    assert abs(v_rule - v_naive) / v_naive > 0.2  # materially different
    # and the rule is the one matching quadrature
    mix = sum(
        0.5
        * np.trapezoid(
            (np.sqrt(w[0, k, 0, 0] / (2 * np.pi))
             * np.exp(-0.5 * w[0, k, 0, 0] * (YS[:, None] - a[0, k, 0, 0] * XS[None, :]) ** 2))
            * _mu(ETA_O[0, 0], LAM_O[0, 0, 0], YS)[:, None],
            YS,
            axis=0,
        )
        for k in range(2)
    )
    m_ref, v_ref = _moments(mix, XS)
    np.testing.assert_allclose(m_rule, m_ref, rtol=1e-5)
    np.testing.assert_allclose(v_rule, v_ref, rtol=1e-5)


def test_node_energy_matches_quadrature():
    f = float(mixg.mixlin_energy(ETA_O, LAM_O, ETA_I, LAM_I, MU_E, A2, B2, W2)[0])
    # q(y,x,s) ∝ f_s(y,x) mu_o(y) mu_i(x) mu_e(s) on the grid
    q = np.stack(
        [
            np.exp(MU_E[0, k])
            * _f_s(YS[:, None], XS[None, :], k)
            * _mu(ETA_O[0, 0], LAM_O[0, 0, 0], YS)[:, None]
            * _mu(ETA_I[0, 0], LAM_I[0, 0, 0], XS)[None, :]
            for k in range(2)
        ]
    )
    z = np.trapezoid(np.trapezoid(q, XS, axis=2), YS, axis=1).sum()
    q /= z
    with np.errstate(divide="ignore", invalid="ignore"):
        logf = np.stack([np.log(_f_s(YS[:, None], XS[None, :], k)) for k in range(2)])
        integrand = np.where(q > 1e-300, q * (np.log(np.maximum(q, 1e-300)) - logf), 0.0)
    f_ref = np.trapezoid(np.trapezoid(integrand, XS, axis=2), YS, axis=1).sum()
    np.testing.assert_allclose(f, f_ref, rtol=1e-5)


def test_collapsed_entropy_upper_bounds_mixture_entropy():
    """The ledger's one approximation (D3.1.3): the Gaussian matched to the
    mixture's moments has entropy >= the mixture's — so interior-edge +H
    terms are over-estimates and the reported BFE is a lower bound."""
    e_f, l_f = (np.asarray(v)[0] for v in mixg.mixlin_forward(ETA_I, LAM_I, MU_E, A2, B2, W2))
    var = 1.0 / l_f[0, 0]
    h_gauss = 0.5 * (1.0 + np.log(2 * np.pi * var))
    w_e = np.exp(MU_E[0]) / np.exp(MU_E[0]).sum()
    mix = sum(
        w_e[k]
        * _norm(
            np.trapezoid(
                _f_s(YS[:, None], XS[None, :], k) * _mu(ETA_I[0, 0], LAM_I[0, 0, 0], XS)[None, :],
                XS,
                axis=1,
            ),
            YS,
        )[0]
        for k in range(2)
    )
    p, _ = _norm(mix, YS)
    h_mix = -np.trapezoid(np.where(p > 1e-300, p * np.log(np.maximum(p, 1e-300)), 0.0), YS)
    assert h_gauss >= h_mix - 1e-12
    assert h_gauss - h_mix > 1e-4  # and strictly so for separated components


# --- prior variant, guards, dtype -------------------------------------------------


def test_prior_variant_matches_quadrature():
    mean = np.array([[[0.5], [-2.0]]])
    logmu = np.asarray(mixg.mix_to_selector(ETA_O, LAM_O, mean, W2))[0]
    vals = []
    for k in range(2):
        comp = np.sqrt(W2[0, k, 0, 0] / (2 * np.pi)) * np.exp(
            -0.5 * W2[0, k, 0, 0] * (YS - mean[0, k, 0]) ** 2
        )
        vals.append(np.trapezoid(comp * _mu(ETA_O[0, 0], LAM_O[0, 0, 0], YS), YS))
    ref = np.log(vals)
    np.testing.assert_allclose(logmu - logmu.max(), ref - ref.max(), rtol=1e-6, atol=1e-8)

    e_f, l_f = (np.asarray(v)[0] for v in mixg.mix_forward(MU_E, mean, W2))
    w_e = np.exp(MU_E[0]) / np.exp(MU_E[0]).sum()
    mix = sum(
        w_e[k]
        * np.sqrt(W2[0, k, 0, 0] / (2 * np.pi))
        * np.exp(-0.5 * W2[0, k, 0, 0] * (YS - mean[0, k, 0]) ** 2)
        for k in range(2)
    )
    m_ref, v_ref = _moments(mix, YS)
    np.testing.assert_allclose(e_f[0] / l_f[0, 0], m_ref, rtol=1e-6)
    np.testing.assert_allclose(1.0 / l_f[0, 0], v_ref, rtol=1e-6)

    f = float(mixg.mix_energy(ETA_O, LAM_O, MU_E, mean, W2)[0])
    q = np.stack(
        [
            np.exp(MU_E[0, k])
            * np.sqrt(W2[0, k, 0, 0] / (2 * np.pi))
            * np.exp(-0.5 * W2[0, k, 0, 0] * (YS - mean[0, k, 0]) ** 2)
            * _mu(ETA_O[0, 0], LAM_O[0, 0, 0], YS)
            for k in range(2)
        ]
    )
    z = np.trapezoid(q, YS, axis=1).sum()
    q /= z
    logf = np.stack(
        [
            np.log(np.sqrt(W2[0, k, 0, 0] / (2 * np.pi)))
            - 0.5 * W2[0, k, 0, 0] * (YS - mean[0, k, 0]) ** 2
            for k in range(2)
        ]
    )
    integrand = np.where(q > 1e-300, q * (np.log(np.maximum(q, 1e-300)) - logf), 0.0)
    np.testing.assert_allclose(f, np.trapezoid(integrand, YS, axis=1).sum(), rtol=1e-5)


def test_vacuous_inputs_fall_back_not_nan():
    """Mid-flooding hostile inputs: vacuous mu_i makes forward components
    improper -> vacuous message out; vacuous mu_o likewise backward; the
    selector message goes uniform; the energy goes +inf. Never NaN."""
    z1, z2 = np.zeros((1, 1)), np.zeros((1, 1, 1))
    e, l = mixg.mixlin_forward(z1, z2, MU_E, A2, B2, W2)
    assert np.all(np.asarray(e) == 0.0) and np.all(np.asarray(l) == 0.0)
    e, l = mixg.mixlin_backward(z1, z2, MU_E, A2, B2, W2)
    assert np.all(np.asarray(e) == 0.0) and np.all(np.asarray(l) == 0.0)
    logmu = np.asarray(mixg.mixlin_to_selector(z1, z2, z1, z2, A2, B2, W2))
    np.testing.assert_allclose(logmu, 0.0)
    f = mixg.mixlin_energy(z1, z2, z1, z2, MU_E, A2, B2, W2)
    assert float(f[0]) == np.inf


def test_float32_finite_and_close():
    args64 = (ETA_O, LAM_O, ETA_I, LAM_I, MU_E, A2, B2, W2)
    args32 = tuple(jnp.asarray(a, dtype=jnp.float32) for a in args64)
    f64 = float(mixg.mixlin_energy(*args64)[0])
    f32 = mixg.mixlin_energy(*args32)
    assert f32.dtype == jnp.float32 and np.isfinite(float(f32[0]))
    np.testing.assert_allclose(float(f32[0]), f64, rtol=5e-4)
    for fn in (mixg.mixlin_forward, mixg.mixlin_backward):
        e, l = fn(args32[0], args32[1], args32[4], args32[5], args32[6], args32[7])
        assert np.all(np.isfinite(np.asarray(e))) and np.all(np.isfinite(np.asarray(l)))

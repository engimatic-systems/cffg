"""Tests of the switching-Kalman-filter reference itself (cff-pksr), not of
the engine (there is no engine hybrid node yet). Pins the two independent
implementations in `tests/swkf_reference.py` against each other and against
a from-scratch plain Kalman smoother, and calibrates the collapse-vs-
enumeration gap that the future engine test tier will need as a tolerance.

Golden numbers and measured gaps recorded here (see individual tests for the
runs that produced them):
    CASE_1D  log evidence = -3.9988525512882926
             sweeps to converge = 40  (tol=1e-12)
             regime-marginal gap (collapsed vs enumerate) ~= 0.0274
             state-mean gap                                ~= 0.1837
    CASE_2D  log evidence = -2.3641760776262357
             sweeps to converge = 19  (tol=1e-12)
             regime-marginal gap                           ~= 0.0683
             state-mean gap                                ~= 0.1613
    MISSING  (CASE_1D with y_2 dropped)
             log evidence = -1.9943626004355166
             sweeps to converge = 41
             regime-marginal gap                           ~= 0.0287
             state-mean gap                                ~= 0.0925
CASE_2D's gap is larger than CASE_1D's for a documented structural reason,
not a bug: with a 1-D observation of a 2-D state (dy < d), the *backward*
collapse message at a chain boundary is exactly rank-deficient (provably:
rank(Lambda_b) == rank(Lambda_o) whenever A_s is square-invertible), so D3's
propriety guard sends it vacuous rather than inverting a singular precision.
That is a real, bounded information loss at low-observability boundaries,
distinct from the moment-matching approximation CASE_1D isolates (d=1=dy,
never rank-deficient). Both are legitimate for the engine test tier to
inherit as calibrated tolerances.
"""

import numpy as np
import pytest

from swkf_reference import (
    SwitchingKFModel,
    _linear_backward,
    _log_mass,
    _moments,
    _softmax,
    collapsed_fixed_point,
    enumerate_exact,
)

# --- Fixture cases (frozen; shared with the future engine test tier) -------

# CASE_1D: d=1, K=2, T=3, moderately-separated regimes (opposite-sign drift,
# different noise scales) with a mildly sticky transition matrix.
CASE_1D_A = np.array([[[0.9]], [[-0.8]]])
CASE_1D_B = np.array([[0.0], [1.5]])
CASE_1D_Q = np.array([[[0.05]], [[0.1]]])
CASE_1D_M0 = np.array([0.0])
CASE_1D_P0 = np.array([[1.0]])
CASE_1D_H = np.array([[1.0]])
CASE_1D_R = np.array([[0.2]])
CASE_1D_PI = np.array([0.5, 0.5])
CASE_1D_P = np.array([[0.9, 0.2], [0.1, 0.8]])  # [out, in]=[s_t,s_{t-1}], columns sum to 1
CASE_1D_YS = [np.array([0.5]), np.array([-0.3]), np.array([1.0])]


def case_1d_model() -> SwitchingKFModel:
    return SwitchingKFModel(
        K=2,
        d=1,
        dy=1,
        pi=CASE_1D_PI,
        P=CASE_1D_P,
        A=CASE_1D_A,
        b=CASE_1D_B,
        Q=CASE_1D_Q,
        m0=CASE_1D_M0,
        P0=CASE_1D_P0,
        H=CASE_1D_H,
        R=CASE_1D_R,
    )


# CASE_2D: d=2 (state), 1-D observation, K=2, T=2. Regimes differ sharply in
# both dynamics (rotation-like vs. near-diagonal-with-offset) and noise
# shape (isotropic vs. correlated) -- deliberately dy < d so the boundary
# backward-collapse guard (see module docstring) is exercised too.
CASE_2D_A = np.array(
    [
        [[0.9, 0.1], [0.0, 0.8]],
        [[0.2, -0.1], [0.1, 0.95]],
    ]
)
CASE_2D_B = np.array([[0.0, 0.0], [1.0, -0.5]])
CASE_2D_Q = np.array(
    [
        [[0.05, 0.0], [0.0, 0.05]],
        [[0.2, 0.05], [0.05, 0.15]],
    ]
)
CASE_2D_M0 = np.zeros(2)
CASE_2D_P0 = np.eye(2)
CASE_2D_H = np.array([[1.0, 0.0]])
CASE_2D_R = np.array([[0.1]])
CASE_2D_PI = np.array([0.6, 0.4])
CASE_2D_P = np.array([[0.7, 0.3], [0.3, 0.7]])
CASE_2D_YS = [np.array([0.4]), np.array([-0.2])]


def case_2d_model() -> SwitchingKFModel:
    return SwitchingKFModel(
        K=2,
        d=2,
        dy=1,
        pi=CASE_2D_PI,
        P=CASE_2D_P,
        A=CASE_2D_A,
        b=CASE_2D_B,
        Q=CASE_2D_Q,
        m0=CASE_2D_M0,
        P0=CASE_2D_P0,
        H=CASE_2D_H,
        R=CASE_2D_R,
    )


# --- Test 1: K=1 degeneracy --------------------------------------------------


def _direct_kalman_smoother(A, b, Q, H, R, m0, P0, ys):
    """Independent, from-scratch (no swkf_reference internals) plain
    Kalman filter + RTS smoother for a single (non-switching) linear-
    Gaussian model. Used only to sanity-check the K=1 reduction."""
    T = len(ys)
    m_filt, p_filt = [m0], [P0]
    m_pred, p_pred = [None], [None]
    for t in range(1, T + 1):
        mp = A @ m_filt[t - 1] + b
        pp = A @ p_filt[t - 1] @ A.T + Q
        m_pred.append(mp)
        p_pred.append(pp)
        y = ys[t - 1]
        if y is not None:
            s = H @ pp @ H.T + R
            v = y - H @ mp
            gain = pp @ H.T @ np.linalg.inv(s)
            m_filt.append(mp + gain @ v)
            p_filt.append(pp - gain @ s @ gain.T)
        else:
            m_filt.append(mp)
            p_filt.append(pp)
    m_smooth = [None] * (T + 1)
    p_smooth = [None] * (T + 1)
    m_smooth[T], p_smooth[T] = m_filt[T], p_filt[T]
    for t in range(T - 1, -1, -1):
        c = p_filt[t] @ A.T @ np.linalg.inv(p_pred[t + 1])
        m_smooth[t] = m_filt[t] + c @ (m_smooth[t + 1] - m_pred[t + 1])
        p_smooth[t] = p_filt[t] + c @ (p_smooth[t + 1] - p_pred[t + 1]) @ c.T
    return np.array(m_smooth), np.array(p_smooth)


def test_k1_degeneracy_matches_plain_kalman_smoother():
    # Full-rank observation (dy=d): avoids the boundary rank-deficient
    # backward-guard case documented above, which is orthogonal to what
    # this test checks (the K=1 mixture-collapse reduction identity).
    A = np.array([[0.9, 0.1], [0.0, 0.8]])
    b = np.array([0.1, -0.2])
    Q = np.array([[0.05, 0.0], [0.0, 0.05]])
    m0, P0 = np.array([0.0, 0.0]), np.eye(2)
    H, R = np.eye(2), np.diag([0.1, 0.15])
    ys = [np.array([0.3, 0.1]), np.array([0.5, -0.2]), None, np.array([-0.1, 0.4])]

    m_direct, p_direct = _direct_kalman_smoother(A, b, Q, H, R, m0, P0, ys)

    model = SwitchingKFModel(
        K=1, d=2, dy=2,
        pi=np.array([1.0]), P=np.array([[1.0]]),
        A=A[None], b=b[None], Q=Q[None],
        m0=m0, P0=P0, H=H, R=R,
    )
    ex = enumerate_exact(model, ys)
    co = collapsed_fixed_point(model, ys, n_sweeps=50, tol=1e-14)

    np.testing.assert_allclose(ex.state_mean, m_direct, atol=1e-9)
    np.testing.assert_allclose(ex.state_cov, p_direct, atol=1e-9)
    np.testing.assert_allclose(co.state_mean, m_direct, atol=1e-9)
    np.testing.assert_allclose(co.state_cov, p_direct, atol=1e-9)
    assert co.converged


# --- Test 2: identical components -> collapse is exact ----------------------


def test_identical_components_collapse_is_exact():
    A = np.array([[0.9, 0.1], [0.0, 0.8]])
    b = np.array([0.1, -0.2])
    Q = np.array([[0.05, 0.0], [0.0, 0.05]])
    m0, P0 = np.array([0.0, 0.0]), np.eye(2)
    H, R = np.eye(2), np.diag([0.1, 0.15])
    ys = [np.array([0.3, 0.1]), np.array([0.5, -0.2]), None, np.array([-0.1, 0.4])]

    K = 2
    model = SwitchingKFModel(
        K=K, d=2, dy=2,
        pi=np.array([0.3, 0.7]),
        P=np.array([[0.6, 0.4], [0.4, 0.6]]),
        A=np.repeat(A[None], K, axis=0),
        b=np.repeat(b[None], K, axis=0),
        Q=np.repeat(Q[None], K, axis=0),
        m0=m0, P0=P0, H=H, R=R,
    )
    ex = enumerate_exact(model, ys)
    co = collapsed_fixed_point(model, ys, n_sweeps=100, tol=1e-14)

    assert co.converged
    np.testing.assert_allclose(co.state_mean, ex.state_mean, atol=1e-9)
    np.testing.assert_allclose(co.state_cov, ex.state_cov, atol=1e-9)
    np.testing.assert_allclose(co.regime_marginals, ex.regime_marginals, atol=1e-9)

    # Regime posterior should also equal the pure discrete-chain posterior
    # (A/b/Q carry zero information about s when identical across regimes,
    # so p(s|y) collapses to the plain HMM forward-backward answer).
    T = len(ys)
    log_pi = np.log(model.pi)
    log_p = np.log(model.P)
    alpha = np.empty((T, K))
    alpha[0] = log_pi
    for t in range(1, T):
        for k in range(K):
            alpha[t, k] = np.logaddexp.reduce(log_p[k] + alpha[t - 1])
    beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        for k in range(K):
            beta[t, k] = np.logaddexp.reduce(log_p[:, k] + beta[t + 1])
    log_post = alpha + beta
    log_post -= np.logaddexp.reduce(log_post, axis=1, keepdims=True)
    hmm_marginals = np.exp(log_post)
    np.testing.assert_allclose(ex.regime_marginals, hmm_marginals, atol=1e-9)
    np.testing.assert_allclose(co.regime_marginals, hmm_marginals, atol=1e-9)


# --- Test 3: fixture cases ---------------------------------------------------


def test_case_1d_converges_with_calibrated_gap():
    model = case_1d_model()
    co = collapsed_fixed_point(model, CASE_1D_YS, n_sweeps=200, tol=1e-12)
    ex = enumerate_exact(model, CASE_1D_YS)

    assert co.converged
    assert co.n_sweeps < 200

    # Deltas should trend down (Jacobi flooding on a lossy fixed-point map
    # need not be strictly monotone every sweep, but the overall decay
    # should be geometric-ish): the tail of the recorded trajectory should
    # be far smaller than the head.
    deltas = co.delta_history
    assert len(deltas) >= 10
    assert deltas[-1] < 0.05 * deltas[0]

    regime_gap = np.max(np.abs(co.regime_marginals - ex.regime_marginals))
    mean_gap = np.max(np.abs(co.state_mean - ex.state_mean))
    # Measured: regime_gap ~= 0.0274, mean_gap ~= 0.1837. ~2x headroom.
    assert regime_gap < 0.06
    assert mean_gap < 0.4


def test_case_1d_log_evidence_golden():
    model = case_1d_model()
    ex = enumerate_exact(model, CASE_1D_YS)
    np.testing.assert_allclose(ex.log_evidence, -3.9988525512882926, rtol=1e-10)
    # recomputation is deterministic
    ex2 = enumerate_exact(model, CASE_1D_YS)
    np.testing.assert_allclose(ex2.log_evidence, ex.log_evidence, rtol=1e-10)


def test_case_2d_converges_with_calibrated_gap():
    model = case_2d_model()
    co = collapsed_fixed_point(model, CASE_2D_YS, n_sweeps=200, tol=1e-12)
    ex = enumerate_exact(model, CASE_2D_YS)

    assert co.converged
    assert co.n_sweeps < 200

    deltas = co.delta_history
    assert len(deltas) >= 8
    assert deltas[-1] < 0.05 * deltas[0]

    regime_gap = np.max(np.abs(co.regime_marginals - ex.regime_marginals))
    mean_gap = np.max(np.abs(co.state_mean - ex.state_mean))
    # Measured: regime_gap ~= 0.0683, mean_gap ~= 0.1613 (dy < d: the
    # boundary backward-guard effect discussed in the module docstring adds
    # to CASE_1D's pure moment-matching gap). ~2x headroom.
    assert regime_gap < 0.14
    assert mean_gap < 0.35


def test_case_2d_log_evidence_golden():
    model = case_2d_model()
    ex = enumerate_exact(model, CASE_2D_YS)
    np.testing.assert_allclose(ex.log_evidence, -2.3641760776262357, rtol=1e-10)
    ex2 = enumerate_exact(model, CASE_2D_YS)
    np.testing.assert_allclose(ex2.log_evidence, ex.log_evidence, rtol=1e-10)


# --- Test 4: weight-trap regression ------------------------------------------


def test_backward_collapse_weight_trap():
    """A single gaussian_mixture_linear node (T=1, K=2, d=1) with sharply
    different |A_s| and Q_s: the naive backward-collapse weighting (softmax
    of the incoming selector message alone, ignoring the per-component
    evidence scale c-tilde_s * Z(eta_b(s), Lambda_b(s))) is implemented only
    here, not in the reference module, and shown to diverge materially from
    D3's weights and from the true (enumerated) posterior."""
    K, d, dy = 2, 1, 1
    A = np.array([[[8.0]], [[0.02]]])
    b = np.array([[0.0], [0.0]])
    Q = np.array([[[0.01]], [[8.0]]])
    m0, p0 = np.array([0.0]), np.array([[50.0]])  # near-vacuous prior
    H, R = np.array([[1.0]]), np.array([[0.05]])
    pi = np.array([0.5, 0.5])
    p_trans = np.array([[1.0, 1.0], [0.0, 0.0]])  # irrelevant at T=1
    model = SwitchingKFModel(K, d, dy, pi, p_trans, A, b, Q, m0, p0, H, R)
    ys = [np.array([1.0])]

    ex = enumerate_exact(model, ys)
    true_mean = ex.state_mean[0, 0]

    w = np.linalg.inv(Q)
    logdet_w = np.array([np.linalg.slogdet(w[s])[1] for s in range(K)])
    rinv = np.linalg.inv(R)
    obs_eta = H.T @ rinv @ ys[0]
    obs_lam = H.T @ rinv @ H
    mu_e = np.log(pi) - np.log(np.sum(pi))

    # D3 rule (the reference implementation).
    co = collapsed_fixed_point(model, ys, n_sweeps=5, tol=1e-14)
    d3_mean = co.state_mean[0, 0]

    # Naive backward collapse: weights = softmax(mu_e) only, dropping the
    # c-tilde_s * Z(eta_b(s), Lambda_b(s)) evidence-scale factors.
    comps = [_linear_backward(obs_eta, obs_lam, A[s], b[s], w[s]) for s in range(K)]
    weights_naive = _softmax(mu_e)
    ms = np.array([_moments(*c)[0] for c in comps])
    cs = np.array([_moments(*c)[1] for c in comps])
    mbar = np.einsum("s,sd->d", weights_naive, ms)
    cbar = np.einsum(
        "s,sij->ij", weights_naive, cs + np.einsum("si,sj->sij", ms, ms)
    ) - np.outer(mbar, mbar)
    lam_naive = np.linalg.inv(cbar)
    eta_naive = lam_naive @ mbar
    lam0 = np.linalg.inv(p0)
    eta0 = lam0 @ m0
    naive_mean, _ = _moments(eta0 + eta_naive, lam0 + lam_naive)
    naive_mean = naive_mean[0]

    # D3's own weights, for a direct look at the trap.
    log_ctilde = np.empty(K)
    for s in range(K):
        lam_c = w[s] + obs_lam
        eta_c = w[s] @ b[s] + obs_eta
        log_ctilde[s] = (
            _log_mass(eta_c, lam_c)
            - 0.5 * (b[s] @ w[s] @ b[s])
            + 0.5 * logdet_w[s]
            - 0.5 * d * np.log(2 * np.pi)
        )
    log_w_d3 = np.array(
        [mu_e[s] + log_ctilde[s] + _log_mass(*comps[s]) for s in range(K)]
    )
    weights_d3 = _softmax(log_w_d3)

    # The weight vectors themselves are materially different...
    assert np.max(np.abs(weights_d3 - weights_naive)) > 0.3
    # ...and using the naive weights gives a materially worse answer.
    d3_err = abs(d3_mean - true_mean)
    naive_err = abs(naive_mean - true_mean)
    assert d3_err < 1e-3
    assert naive_err > 5 * d3_err


# --- Test 5: missing observation ---------------------------------------------


def test_missing_observation_matches_enumeration_on_reduced_data():
    model = case_1d_model()
    ys = [CASE_1D_YS[0], None, CASE_1D_YS[2]]

    ex = enumerate_exact(model, ys)
    co = collapsed_fixed_point(model, ys, n_sweeps=200, tol=1e-12)

    assert co.converged
    np.testing.assert_allclose(ex.log_evidence, -1.9943626004355166, rtol=1e-10)

    regime_gap = np.max(np.abs(co.regime_marginals - ex.regime_marginals))
    mean_gap = np.max(np.abs(co.state_mean - ex.state_mean))
    # Measured: regime_gap ~= 0.0287, mean_gap ~= 0.0925. ~2x headroom.
    assert regime_gap < 0.06
    assert mean_gap < 0.2

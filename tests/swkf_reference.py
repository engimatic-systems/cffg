"""Numpy reference for the switching-Kalman-filter test tier (cff-pksr).

Two *independent* inference implementations for the same model family:

    Discrete regime chain   s_1..s_T in {0..K-1}:
        s_1 ~ Categorical(pi),  s_t | s_{t-1} ~ Categorical(P[:, s_{t-1}])
    Continuous state chain   x_0..x_T:
        x_0 ~ N(m0, P0)
        x_t = A[s_t] @ x_{t-1} + b[s_t] + N(0, Q[s_t])
    Emissions (all optional, one per continuous state after x_0):
        y_t = H @ x_t + N(0, R),   t = 1..T   (y_t may be None: missing)

Matrix convention (checked against tests/conftest.py::make_hmm_slice and
cffg/rules.py::bp_matrix_cat, cffg/compiler.py::_stochastic): the transition
array is stored [out, in] with the OUT axis (rows) summing to 1 per column,
i.e. `P[:, i].sum() == 1` and `P[s_t, s_{t-1}]` is the transition
probability. `_stochastic` in cffg/compiler.py checks `arr.sum(axis=0)`
(the out/row axis) == 1, and `bp_matrix_cat`'s forward message
`to_out = logsumexp(logf + m_in[:, None, :], axis=2)` sums the log-factor
(shape [ko, ki] = [out, in]) over the `in` axis weighted by the incoming
`in`-message -- i.e. logf[o, i] = log P(out=o | in=i). This module's `P`
follows the identical [out, in], column-stochastic layout.

Deliverable (a): `enumerate_exact` -- brute-force posterior by enumerating
all K**T regime sequences, running a conditioned Kalman filter/RTS smoother
per sequence, and mixing with sequence posterior weights.

Deliverable (b): `collapsed_fixed_point` -- synchronous-flooding message
passing implementing the collapse rules of docs/DERIVATIONS.md D3 (Gaussian
mixtures) on the same factor graph: a `gaussian_mixture_linear(out, in,
selector)` node per time step t (out=x_t, in=x_{t-1}, selector=s_t), plus a
plain `matrix_cat` chain for the regime sequence's own transitions, plus
Gaussian priors/observations. Gaussian edges carry (eta, Lambda) information
pairs (scale-free); regime/selector edges carry normalized log-probability
vectors. Every sweep recomputes every message from the *previous* sweep's
messages only (Jacobi-style flooding), so the vacuous (eta=0, Lambda=0) /
uniform (log-vector = 0) initialization is a legal starting point -- the D3
propriety guards (fall back to vacuous / uniform when a component's
precision is not positive definite) are exactly what keeps early sweeps from
crashing on singular intermediate precisions.

NumPy + SciPy only (SciPy is a repo dev dependency); no JAX.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

_LOG2PI = 1.8378770664093453  # log(2*pi)
_PD_TOL = 1e-9  # eigenvalue floor for "is this precision proper"


# --- Model & result containers ----------------------------------------------


@dataclass
class SwitchingKFModel:
    """Parameters for one switching-Kalman-filter instance.

    K: number of regimes. d: continuous state dimension. dy: observation
    dimension. `pi` [K]: initial regime distribution. `P` [K, K]: transition
    array, [out, in] = [s_t, s_{t-1}], column-stochastic (see module
    docstring). `A` [K, d, d], `b` [K, d], `Q` [K, d, d]: per-regime linear
    dynamics x_t = A[s_t] @ x_{t-1} + b[s_t] + N(0, Q[s_t]). `m0` [d], `P0`
    [d, d]: x_0 prior. `H` [dy, d], `R` [dy, dy]: shared (regime-independent)
    emission model y_t = H @ x_t + N(0, R).
    """

    K: int
    d: int
    dy: int
    pi: np.ndarray
    P: np.ndarray
    A: np.ndarray
    b: np.ndarray
    Q: np.ndarray
    m0: np.ndarray
    P0: np.ndarray
    H: np.ndarray
    R: np.ndarray


@dataclass
class ExactResult:
    log_evidence: float
    regime_marginals: np.ndarray  # [T, K]     p(s_t | y)
    state_mean: np.ndarray  # [T+1, d]         mixture mean, t = 0..T
    state_cov: np.ndarray  # [T+1, d, d]       mixture cov (incl. between-component spread)


@dataclass
class CollapsedResult:
    regime_marginals: np.ndarray  # [T, K]
    state_mean: np.ndarray  # [T+1, d]
    state_cov: np.ndarray  # [T+1, d, d]
    n_sweeps: int
    converged: bool
    max_delta: float  # final sweep's max abs message change
    delta_history: list  # max abs message change per sweep (diagnostic)


# --- Small linear-algebra helpers -------------------------------------------


def _is_pd(mat: np.ndarray, tol: float = _PD_TOL) -> bool:
    sym = 0.5 * (mat + mat.T)
    try:
        w = np.linalg.eigvalsh(sym)
    except np.linalg.LinAlgError:
        return False
    return bool(np.all(w > tol))


def _log_mass(eta: np.ndarray, lam: np.ndarray) -> float:
    """log Z(eta, Lambda) = d/2 log 2pi - 1/2 logdet Lambda + 1/2 eta.Lambda^-1.eta.
    Caller guarantees Lambda is PD."""
    d = eta.shape[0]
    _, logdet = np.linalg.slogdet(lam)
    quad = eta @ np.linalg.solve(lam, eta)
    return 0.5 * d * _LOG2PI - 0.5 * logdet + 0.5 * quad


def _softmax(logp: np.ndarray) -> np.ndarray:
    return np.exp(logp - logsumexp(logp))


def _moments(eta: np.ndarray, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cov = np.linalg.inv(lam)
    return cov @ eta, cov


# --- gaussian_linear algebra (ports docs/DERIVATIONS.md D2/gauss.py to numpy,
#     single-instance -- no vmap needed since we drive an explicit chain) ----


def _linear_forward(eta_i, lam_i, a, b, w):
    """f(y,x) = N(y; Ax+b, W^-1) -> message to y (out), marginalizing x
    against incoming (eta_i, lam_i) on x (in). Returns None if the curvature
    A'WA + lam_i is singular (the propriety guard upstream)."""
    s = a.T @ w @ a + lam_i
    try:
        c_eta = np.linalg.solve(s, eta_i)
        c_awt = np.linalg.solve(s, a.T @ w)
    except np.linalg.LinAlgError:
        return None
    lam = w - w @ a @ c_awt
    eta = lam @ b + w @ a @ c_eta
    return eta, lam


def _linear_backward(eta_o, lam_o, a, b, w):
    """f(y,x) = N(y; Ax+b, W^-1) -> message to x (in), marginalizing y
    against incoming (eta_o, lam_o) on y (out). W + lam_o is always PD (W is
    PD, lam_o is PSD), so this never raises."""
    k = np.linalg.solve(w + lam_o, w)
    g = w - w @ k
    lam = a.T @ g @ a
    eta = a.T @ (k.T @ eta_o - g @ b)
    return eta, lam


# --- D3 gaussian_mixture_linear rules ---------------------------------------
#
# Per docs/DERIVATIONS.md D3.2. f_s(y, x) = N(y; A_s x + b_s, Q_s), W_s =
# Q_s^-1, K regimes. Incoming: mu_o = (eta_o, lam_o) on `out` (=y), mu_i =
# (eta_i, lam_i) on `in` (=x), mu_e[s] a normalized selector log-message.


def mix_to_selector(eta_o, lam_o, eta_i, lam_i, mu_e, a, b, w, logdet_w, d):
    """Exact per-regime selector message (joint log-partition). Guard: if
    any regime's joint precision Lambda_J is not PD, the whole message falls
    back to uniform (vacuous mid-flooding)."""
    k = a.shape[0]
    vals = np.empty(k)
    for s in range(k):
        lam_j = np.block(
            [
                [w[s] + lam_o, -w[s] @ a[s]],
                [-a[s].T @ w[s], a[s].T @ w[s] @ a[s] + lam_i],
            ]
        )
        if not _is_pd(lam_j):
            return np.zeros(k) - logsumexp(np.zeros(k))
        eta_j = np.concatenate([w[s] @ b[s] + eta_o, -a[s].T @ w[s] @ b[s] + eta_i])
        log_z = _log_mass(eta_j, lam_j)
        vals[s] = log_z - 0.5 * (b[s] @ w[s] @ b[s]) + 0.5 * logdet_w[s] - 0.5 * d * _LOG2PI
    return vals - logsumexp(vals)


def mix_to_out(eta_i, lam_i, mu_e, a, b, w, d):
    """Collapsed forward message (to `out`): component masses are
    s-independent, so weights are exactly softmax(mu_e). Guard: any
    non-PD component precision -> vacuous."""
    k = a.shape[0]
    comps = []
    for s in range(k):
        res = _linear_forward(eta_i, lam_i, a[s], b[s], w[s])
        if res is None or not _is_pd(res[1]):
            return np.zeros(d), np.zeros((d, d))
        comps.append(res)
    weights = _softmax(mu_e)
    ms = np.array([_moments(*c)[0] for c in comps])
    cs = np.array([_moments(*c)[1] for c in comps])
    mbar = np.einsum("s,sd->d", weights, ms)
    cbar = np.einsum("s,sij->ij", weights, cs + np.einsum("si,sj->sij", ms, ms)) - np.outer(
        mbar, mbar
    )
    if not _is_pd(cbar):
        return np.zeros(d), np.zeros((d, d))
    lam_out = np.linalg.inv(cbar)
    return lam_out @ mbar, lam_out


def mix_to_in(eta_o, lam_o, mu_e, a, b, w, logdet_w, d):
    """Collapsed backward message (to `in`) -- the weight trap: component
    masses differ, weights need both the y-marginal evidence c-tilde_s and
    the shape-mass Z(eta_b(s), Lambda_b(s)), NOT just exp(mu_e(s)). Guard:
    any non-PD component precision -> vacuous."""
    k = a.shape[0]
    comps = []
    log_ctilde = np.empty(k)
    for s in range(k):
        res = _linear_backward(eta_o, lam_o, a[s], b[s], w[s])
        if not _is_pd(res[1]):
            return np.zeros(d), np.zeros((d, d))
        comps.append(res)
        lam_c = w[s] + lam_o
        eta_c = w[s] @ b[s] + eta_o
        log_z_c = _log_mass(eta_c, lam_c)
        log_ctilde[s] = (
            log_z_c - 0.5 * (b[s] @ w[s] @ b[s]) + 0.5 * logdet_w[s] - 0.5 * d * _LOG2PI
        )
    log_w = np.array(
        [mu_e[s] + log_ctilde[s] + _log_mass(*comps[s]) for s in range(k)]
    )
    weights = _softmax(log_w)
    ms = np.array([_moments(*c)[0] for c in comps])
    cs = np.array([_moments(*c)[1] for c in comps])
    mbar = np.einsum("s,sd->d", weights, ms)
    cbar = np.einsum("s,sij->ij", weights, cs + np.einsum("si,sj->sij", ms, ms)) - np.outer(
        mbar, mbar
    )
    if not _is_pd(cbar):
        return np.zeros(d), np.zeros((d, d))
    lam_in = np.linalg.inv(cbar)
    return lam_in @ mbar, lam_in


# --- matrix_cat (regime chain) sum-product, both directions ----------------


def matrix_cat_messages(log_p, m_in, m_out):
    """log_p: [K_out, K_in] = [s_t, s_{t-1}] (see module docstring).
    Returns (to_out, to_in), both normalized log-probability vectors."""
    to_out = logsumexp(log_p + m_in[None, :], axis=1)
    to_in = logsumexp(log_p + m_out[:, None], axis=0)
    return to_out - logsumexp(to_out), to_in - logsumexp(to_in)


# --- Deliverable (a): exact enumeration -------------------------------------


def enumerate_exact(model: SwitchingKFModel, ys: list) -> ExactResult:
    """Exact posterior by enumerating all K**T regime sequences; a
    conditioned Kalman filter/RTS smoother per sequence, mixed with sequence
    posterior weights. `ys[t]` (t=0..T-1, for y_{t+1}) may be None (missing
    -> that emission node drops out)."""
    K, d = model.K, model.d
    T = len(ys)
    log_pi = np.log(np.clip(model.pi, 1e-300, None))
    log_p = np.log(np.clip(model.P, 1e-300, None))

    seqs = list(itertools.product(range(K), repeat=T))
    log_w = np.empty(len(seqs))
    means = np.empty((len(seqs), T + 1, d))
    covs = np.empty((len(seqs), T + 1, d, d))

    for n, seq in enumerate(seqs):
        m_filt = [None] * (T + 1)
        p_filt = [None] * (T + 1)
        m_pred = [None] * (T + 1)
        p_pred = [None] * (T + 1)
        m_filt[0], p_filt[0] = model.m0.copy(), model.P0.copy()
        loglik = 0.0
        for t in range(1, T + 1):
            s = seq[t - 1]
            a_s, b_s, q_s = model.A[s], model.b[s], model.Q[s]
            mp = a_s @ m_filt[t - 1] + b_s
            pp = a_s @ p_filt[t - 1] @ a_s.T + q_s
            m_pred[t], p_pred[t] = mp, pp
            y = ys[t - 1]
            if y is not None:
                s_res = model.H @ pp @ model.H.T + model.R
                v = y - model.H @ mp
                k_gain = pp @ model.H.T @ np.linalg.inv(s_res)
                mf = mp + k_gain @ v
                pf = pp - k_gain @ s_res @ k_gain.T
                _, logdet_s = np.linalg.slogdet(s_res)
                dy = y.shape[0]
                loglik += -0.5 * (
                    dy * _LOG2PI + logdet_s + v @ np.linalg.solve(s_res, v)
                )
            else:
                mf, pf = mp, pp
            m_filt[t], p_filt[t] = mf, pf

        m_smooth = [None] * (T + 1)
        p_smooth = [None] * (T + 1)
        m_smooth[T], p_smooth[T] = m_filt[T], p_filt[T]
        for t in range(T - 1, -1, -1):
            a_s = model.A[seq[t]]
            c = p_filt[t] @ a_s.T @ np.linalg.inv(p_pred[t + 1])
            m_smooth[t] = m_filt[t] + c @ (m_smooth[t + 1] - m_pred[t + 1])
            p_smooth[t] = p_filt[t] + c @ (p_smooth[t + 1] - p_pred[t + 1]) @ c.T

        means[n] = np.array(m_smooth)
        covs[n] = np.array(p_smooth)
        log_prior = log_pi[seq[0]] + sum(log_p[seq[t], seq[t - 1]] for t in range(1, T))
        log_w[n] = loglik + log_prior

    log_evidence = float(logsumexp(log_w))
    w = np.exp(log_w - log_evidence)

    regime_marginals = np.zeros((T, K))
    for n, seq in enumerate(seqs):
        for t in range(T):
            regime_marginals[t, seq[t]] += w[n]

    state_mean = np.einsum("n,ntd->td", w, means)
    second_moment = np.einsum(
        "n,ntij->tij", w, covs + np.einsum("ntd,nte->ntde", means, means)
    )
    state_cov = second_moment - np.einsum("td,te->tde", state_mean, state_mean)

    return ExactResult(
        log_evidence=log_evidence,
        regime_marginals=regime_marginals,
        state_mean=state_mean,
        state_cov=state_cov,
    )


# --- Deliverable (b): collapsed fixed point ---------------------------------


def collapsed_fixed_point(
    model: SwitchingKFModel, ys: list, n_sweeps: int = 200, tol: float = 1e-10
) -> CollapsedResult:
    """Synchronous-flooding message passing with D3's collapse rules.

    Factor graph (a tree): x_0..x_T Gaussian variables, s_1..s_T discrete
    variables (0-indexed here as s-index 0..T-1); `mix[i]` (i=0..T-1) is the
    gaussian_mixture_linear node with out=x[i+1], in=x[i], selector=s[i];
    `trans[j]` (j=0..T-2) is the matrix_cat node with out=s[j+1], in=s[j];
    plus constant leaves: the x_0 Gaussian prior, the s_1 categorical prior,
    and one Gaussian-observation leaf per non-missing y_t attached to x[t].

    Every sweep recomputes every message from the PREVIOUS sweep's full
    message set only (Jacobi/flooding), so the initial vacuous state (all
    Gaussian messages (0,0), all discrete messages the zero log-vector, i.e.
    uniform) is valid; D3's propriety guards keep the singular intermediate
    precisions this produces from crashing the linear algebra.
    """
    K, d = model.K, model.d
    T = len(ys)
    w = np.linalg.inv(model.Q)  # [K,d,d]
    logdet_w = np.array([np.linalg.slogdet(w[s])[1] for s in range(K)])
    log_p = np.log(np.clip(model.P, 1e-300, None))
    log_pi = np.log(np.clip(model.pi, 1e-300, None))

    lam0 = np.linalg.inv(model.P0)
    eta0 = lam0 @ model.m0

    rinv = np.linalg.inv(model.R)
    ht_rinv_h = model.H.T @ rinv @ model.H
    obs_eta = np.zeros((T, d))
    obs_lam = np.zeros((T, d, d))
    for t in range(T):
        if ys[t] is not None:
            obs_eta[t] = model.H.T @ rinv @ ys[t]
            obs_lam[t] = ht_rinv_h

    mix_out_eta = np.zeros((T, d))
    mix_out_lam = np.zeros((T, d, d))
    mix_in_eta = np.zeros((T, d))
    mix_in_lam = np.zeros((T, d, d))
    mix_sel = np.zeros((T, K))  # mix[i] -> s[i]

    n_trans = max(T - 1, 0)
    trans_to_out = np.zeros((n_trans, K))  # trans[j] -> s[j+1]
    trans_to_in = np.zeros((n_trans, K))  # trans[j] -> s[j]

    sweeps_used = 0
    last_delta = np.inf
    converged = False
    delta_history: list = []

    for sweep in range(n_sweeps):
        new_mix_out_eta = np.zeros_like(mix_out_eta)
        new_mix_out_lam = np.zeros_like(mix_out_lam)
        new_mix_in_eta = np.zeros_like(mix_in_eta)
        new_mix_in_lam = np.zeros_like(mix_in_lam)
        new_mix_sel = np.zeros_like(mix_sel)

        for i in range(T):
            if i == 0:
                mu_i_eta, mu_i_lam = eta0, lam0
            else:
                mu_i_eta = mix_out_eta[i - 1] + obs_eta[i - 1]
                mu_i_lam = mix_out_lam[i - 1] + obs_lam[i - 1]

            if i == T - 1:
                mu_o_eta, mu_o_lam = obs_eta[i], obs_lam[i]
            else:
                mu_o_eta = mix_in_eta[i + 1] + obs_eta[i]
                mu_o_lam = mix_in_lam[i + 1] + obs_lam[i]

            left = log_pi if i == 0 else trans_to_out[i - 1]
            right = trans_to_in[i] if i <= T - 2 else np.zeros(K)
            mu_e = left + right
            mu_e = mu_e - logsumexp(mu_e)

            new_mix_sel[i] = mix_to_selector(
                mu_o_eta, mu_o_lam, mu_i_eta, mu_i_lam, mu_e, model.A, model.b, w, logdet_w, d
            )
            new_mix_out_eta[i], new_mix_out_lam[i] = mix_to_out(
                mu_i_eta, mu_i_lam, mu_e, model.A, model.b, w, d
            )
            new_mix_in_eta[i], new_mix_in_lam[i] = mix_to_in(
                mu_o_eta, mu_o_lam, mu_e, model.A, model.b, w, logdet_w, d
            )

        new_trans_to_out = np.zeros_like(trans_to_out)
        new_trans_to_in = np.zeros_like(trans_to_in)
        for j in range(n_trans):
            m_in = (log_pi if j == 0 else trans_to_out[j - 1]) + mix_sel[j]
            m_in = m_in - logsumexp(m_in)
            m_out = (trans_to_in[j + 1] if j <= n_trans - 2 else np.zeros(K)) + mix_sel[j + 1]
            m_out = m_out - logsumexp(m_out)
            new_trans_to_out[j], new_trans_to_in[j] = matrix_cat_messages(log_p, m_in, m_out)

        delta = max(
            np.max(np.abs(new_mix_out_eta - mix_out_eta)) if T else 0.0,
            np.max(np.abs(new_mix_out_lam - mix_out_lam)) if T else 0.0,
            np.max(np.abs(new_mix_in_eta - mix_in_eta)) if T else 0.0,
            np.max(np.abs(new_mix_in_lam - mix_in_lam)) if T else 0.0,
            np.max(np.abs(new_mix_sel - mix_sel)) if T else 0.0,
            np.max(np.abs(new_trans_to_out - trans_to_out)) if n_trans else 0.0,
            np.max(np.abs(new_trans_to_in - trans_to_in)) if n_trans else 0.0,
        )

        mix_out_eta, mix_out_lam = new_mix_out_eta, new_mix_out_lam
        mix_in_eta, mix_in_lam = new_mix_in_eta, new_mix_in_lam
        mix_sel = new_mix_sel
        trans_to_out, trans_to_in = new_trans_to_out, new_trans_to_in

        sweeps_used = sweep + 1
        last_delta = delta
        delta_history.append(delta)
        if delta < tol:
            converged = True
            break

    # Readout: belief at each x[k], k=0..T (sum of ALL neighbor messages).
    state_mean = np.zeros((T + 1, d))
    state_cov = np.zeros((T + 1, d, d))
    for k in range(T + 1):
        if k == 0:
            eta, lam = eta0 + mix_in_eta[0], lam0 + mix_in_lam[0]
        elif k == T:
            eta, lam = mix_out_eta[T - 1] + obs_eta[T - 1], mix_out_lam[T - 1] + obs_lam[T - 1]
        else:
            eta = mix_out_eta[k - 1] + mix_in_eta[k] + obs_eta[k - 1]
            lam = mix_out_lam[k - 1] + mix_in_lam[k] + obs_lam[k - 1]
        state_mean[k], state_cov[k] = _moments(eta, lam)

    regime_marginals = np.zeros((T, K))
    for i in range(T):
        left = log_pi if i == 0 else trans_to_out[i - 1]
        right = trans_to_in[i] if i <= T - 2 else np.zeros(K)
        belief = left + right + mix_sel[i]
        regime_marginals[i] = _softmax(belief)

    return CollapsedResult(
        regime_marginals=regime_marginals,
        state_mean=state_mean,
        state_cov=state_cov,
        n_sweeps=sweeps_used,
        converged=converged,
        max_delta=last_delta,
        delta_history=delta_history,
    )

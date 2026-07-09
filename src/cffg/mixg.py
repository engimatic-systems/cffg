"""Hybrid Gaussian-mixture rules (cff-pksr): DERIVATIONS.md D3 in jnp.

The two factor types these serve:

- ``gaussian_mixture``        f_s(y)    = N(y; m_s, V_s)          (out, selector)
- ``gaussian_mixture_linear`` f_s(y, x) = N(y; A_s x + b_s, V_s)  (out, in, selector)

Messages to the selector are exact (per-regime joint log-partitions).
Messages on the Gaussian edges are mixtures collapsed to one Gaussian by
moment matching; D3 derives the direction-dependent component weights —
forward weights are the incoming selector probabilities alone, backward
weights need the per-component scales the engine's scale-free message
convention drops (the "weight trap").

Conventions match cffg.gauss: batched over group members ([M, ...] leading
axis) via vmap, per-member parameter stacks carry the regime axis K
([M, K, ...]); dims and K are static per group. Propriety guards fall back
to the *vacuous* message mid-flooding (never NaN); ledger terms go +inf on
improper beliefs, the engine-wide loudness convention. Guards are
dtype-derived per the robustness contract: ridge = eps**0.75 of the working
dtype (f64 ≈ 2e-12, f32 ≈ 6e-6) — big enough to be representable, small
enough to be an O(ulp) perturbation at fixed points.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_LOG2PI = 1.8378770664093453


def _ridge(dtype) -> jnp.ndarray:
    # eps**0.75: invariant protected — solves on transiently singular
    # systems stay finite in BOTH f64 and f32 (an absolute 1e-10 ridge is
    # below f32's resolution and would be a silent no-op there).
    return jnp.asarray(jnp.finfo(dtype).eps ** 0.75)


def _pd_ok(mat: jnp.ndarray) -> jnp.ndarray:
    """Strict-PD test usable under jit. Cholesky won't do: it is finite on
    exactly-singular PSD input, and a singular Λ is precisely the improper
    case the guards must catch (vacuous forward, rank-deficient backward).
    Invariant protected: moments/entropies are only evaluated where they
    exist. Threshold is dtype-scaled (√eps): below it a direction is
    indistinguishable from unconstrained at working precision."""
    return jnp.linalg.eigvalsh(mat).min(axis=-1) > jnp.sqrt(jnp.finfo(mat.dtype).eps)


def _log_z(eta: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
    """log ∫ exp(−½yᵀΛy + ηᵀy) dy = ½ηᵀΛ⁻¹η + d/2·log2π − ½·logdet Λ.
    Caller guarantees Λ ≻ 0 (guarded upstream); ridged solve keeps the
    untaken branch of a jnp.where finite."""
    d = lam.shape[-1]
    ridged = lam + _ridge(lam.dtype) * jnp.eye(d, dtype=lam.dtype)
    sol = jnp.linalg.solve(ridged, eta[..., None])[..., 0]
    _, logdet = jnp.linalg.slogdet(ridged)
    return 0.5 * (eta * sol).sum(-1) + 0.5 * d * _LOG2PI - 0.5 * logdet


def _joint_blocks(eta_o, lam_o, eta_i, lam_i, a, b, w):
    """Per-regime joint (η_J, Λ_J) over (y, x) of node × μ_o × μ_i — the
    same blocks as gauss.linear_energy, vectorized over the K axis."""
    wa = jnp.einsum("kij,kjl->kil", w, a)  # [K, do, di]
    top = jnp.concatenate([w + lam_o[None], -wa], axis=-1)
    bot = jnp.concatenate([-wa.swapaxes(-1, -2), jnp.einsum("kji,kjl->kil", a, wa) + lam_i[None]], axis=-1)
    lam_j = jnp.concatenate([top, bot], axis=-2)  # [K, do+di, do+di]
    wb = jnp.einsum("kij,kj->ki", w, b)
    eta_j = jnp.concatenate(
        [wb + eta_o[None], -jnp.einsum("kji,kj->ki", a, wb) + eta_i[None]], axis=-1
    )
    return eta_j, lam_j


def _node_const(b, w):
    """Per-regime constant of log f_s: −½ b_sᵀW_s b_s + ½ logdet W_s − ½ d_y log2π."""
    _, logdet_w = jnp.linalg.slogdet(w)
    return -0.5 * jnp.einsum("ki,kij,kj->k", b, w, b) + 0.5 * logdet_w - 0.5 * b.shape[-1] * _LOG2PI


def _collapse(log_w: jnp.ndarray, eta: jnp.ndarray, lam: jnp.ndarray):
    """Moment-match Σ_s softmax(log_w)_s · N(η_s, Λ_s) → one (η, Λ).
    All components must be proper (guarded by callers)."""
    w_s = jax.nn.softmax(log_w)  # [K]
    d = lam.shape[-1]
    ridge_eye = _ridge(lam.dtype) * jnp.eye(d, dtype=lam.dtype)
    cov = jnp.linalg.inv(lam + ridge_eye)  # [K, d, d]
    m = jnp.einsum("kij,kj->ki", cov, eta)
    m_bar = jnp.einsum("k,ki->i", w_s, m)
    spread = m - m_bar[None]
    c_bar = jnp.einsum("k,kij->ij", w_s, cov) + jnp.einsum("k,ki,kj->ij", w_s, spread, spread)
    lam_out = jnp.linalg.inv(c_bar + ridge_eye)
    return lam_out @ m_bar, lam_out


def _vacuous_like(eta, lam):
    return jnp.zeros_like(eta), jnp.zeros_like(lam)


# --- gaussian_mixture_linear ---------------------------------------------------


def mixlin_to_selector(eta_o, lam_o, eta_i, lam_i, a, b, w):
    """→ selector, exact: log μ(s) = log Z(η_J, Λ_J) + node const (D3).
    Any improper joint ⇒ uniform (zeros: log-messages are shift-free)."""

    def one(eo, lo, ei, li, A, bb, W):
        eta_j, lam_j = _joint_blocks(eo, lo, ei, li, A, bb, W)
        ok = _pd_ok(lam_j).all()
        logmu = _log_z(eta_j, lam_j) + _node_const(bb, W)
        return jnp.where(ok, logmu - logmu.max(), jnp.zeros_like(logmu))

    return jax.vmap(one)(eta_o, lam_o, eta_i, lam_i, a, b, w)


def mixlin_forward(eta_i, lam_i, log_mu_e, a, b, w):
    """→ out: collapse of per-regime forward predictives; weights are the
    incoming selector message alone (component masses are s-independent)."""

    def one(ei, li, le, A, bb, W):
        def fwd(Ak, bk, Wk):
            s = Ak.T @ Wk @ Ak + li
            ridge_eye = _ridge(s.dtype) * jnp.eye(s.shape[-1], dtype=s.dtype)
            c_awt = jnp.linalg.solve(s + ridge_eye, Ak.T @ Wk)
            lam = Wk - Wk @ Ak @ c_awt
            eta = lam @ bk + Wk @ Ak @ jnp.linalg.solve(s + ridge_eye, ei)
            return eta, lam

        eta_f, lam_f = jax.vmap(fwd)(A, bb, W)  # [K, do], [K, do, do]
        ok = _pd_ok(lam_f).all()
        eta, lam = _collapse(le, eta_f, lam_f)
        return jnp.where(ok, eta, 0.0), jnp.where(ok, lam, 0.0)

    return jax.vmap(one)(eta_i, lam_i, log_mu_e, a, b, w)


def mixlin_backward(eta_o, lam_o, log_mu_e, a, b, w):
    """→ in: collapse of per-regime backward messages with the D3 weights
    w̃_s ∝ μ_e(s) · c̃_s · Z(η_b, Λ_b) — the scale factors the scale-free
    message convention drops must be rebuilt here (the weight trap)."""

    def one(eo, lo, le, A, bb, W):
        def bwd(Ak, bk, Wk):
            d = Wk.shape[-1]
            ridge_eye = _ridge(Wk.dtype) * jnp.eye(d, dtype=Wk.dtype)
            k = jnp.linalg.solve(Wk + lo + ridge_eye, Wk)
            g = Wk - Wk @ k
            lam = Ak.T @ g @ Ak
            eta = Ak.T @ (k.T @ eo - g @ bk)
            # log c̃: mass of ∫ N(y; b_k, V_k) μ_o(y) dy (D3)
            log_c = _log_z(Wk @ bk + eo, Wk + lo) - 0.5 * bk @ Wk @ bk
            _, logdet_w = jnp.linalg.slogdet(Wk)
            log_c = log_c + 0.5 * logdet_w - 0.5 * bk.shape[-1] * _LOG2PI
            return eta, lam, log_c

        eta_b, lam_b, log_c = jax.vmap(bwd)(A, bb, W)
        ok = _pd_ok(lam_b).all()
        log_w = le + log_c + jnp.where(ok, _log_z(eta_b, lam_b), 0.0)
        eta, lam = _collapse(log_w - log_w.max(), eta_b, lam_b)
        return jnp.where(ok, eta, 0.0), jnp.where(ok, lam, 0.0)

    return jax.vmap(one)(eta_o, lam_o, log_mu_e, a, b, w)


def mixlin_energy(eta_o, lam_o, eta_i, lam_i, log_mu_e, a, b, w):
    """Node energy, exact for the held beliefs (D3): mixture belief over
    (y, x, s) with regime weights ρ_s ∝ μ_e(s)·Z_J(s); joint entropy by the
    chain rule; per-regime E[log f] via the linear_energy residual algebra.
    Improper ⇒ +inf (loud)."""

    def one(eo, lo, ei, li, le, A, bb, W):
        eta_j, lam_j = _joint_blocks(eo, lo, ei, li, A, bb, W)
        ok = _pd_ok(lam_j).all()
        d = lam_j.shape[-1]
        ridge_eye = _ridge(lam_j.dtype) * jnp.eye(d, dtype=lam_j.dtype)
        log_rho = le + _log_z(eta_j, lam_j) + _node_const(bb, W)
        rho = jax.nn.softmax(log_rho)
        log_rho_n = jax.nn.log_softmax(log_rho)
        h_s = -(rho * log_rho_n).sum()
        _, logdet_j = jnp.linalg.slogdet(lam_j + ridge_eye)
        h_gauss = 0.5 * d * (1.0 + _LOG2PI) - 0.5 * logdet_j  # [K]

        sig_j = jnp.linalg.inv(lam_j + ridge_eye)
        m_j = jnp.einsum("kij,kj->ki", sig_j, eta_j)
        do = bb.shape[-1]
        jm = jnp.concatenate([jnp.eye(do, dtype=A.dtype)[None].repeat(A.shape[0], 0), -A], axis=-1)
        r = jnp.einsum("kij,kj->ki", jm, m_j) - bb  # [K, do]
        sig_r = jnp.einsum("kij,kjl,kml->kim", jm, sig_j, jm)
        _, logdet_w = jnp.linalg.slogdet(W)
        e_logf = (
            0.5 * logdet_w
            - 0.5 * do * _LOG2PI
            - 0.5 * (jnp.einsum("ki,kij,kj->k", r, W, r) + jnp.einsum("kij,kji->k", W, sig_r))
        )
        f = -h_s - (rho * h_gauss).sum() - (rho * e_logf).sum()
        return jnp.where(ok, f, jnp.inf)

    return jax.vmap(one)(eta_o, lam_o, eta_i, lam_i, log_mu_e, a, b, w)


# --- gaussian_mixture (prior variant, f_s(y) = N(y; m_s, V_s)) -------------------


def mix_to_selector(eta_o, lam_o, mean, w):
    """→ selector, exact and always proper: W_s + Λ_o ≻ 0."""

    def one(eo, lo, m, W):
        wm = jnp.einsum("kij,kj->ki", W, m)
        logmu = (
            _log_z(wm + eo[None], W + lo[None])
            - 0.5 * jnp.einsum("ki,kij,kj->k", m, W, m)
            + 0.5 * jnp.linalg.slogdet(W)[1]
            - 0.5 * m.shape[-1] * _LOG2PI
        )
        return logmu - logmu.max()

    return jax.vmap(one)(eta_o, lam_o, mean, w)


def mix_forward(log_mu_e, mean, w):
    """→ out: collapse of the component priors N(m_s, V_s) with selector
    weights (masses are all 1 — no trap in this direction)."""

    def one(le, m, W):
        return _collapse(le, jnp.einsum("kij,kj->ki", W, m), W)

    return jax.vmap(one)(log_mu_e, mean, w)


def mix_energy(eta_o, lam_o, log_mu_e, mean, w):
    """Node energy of the prior mixture: same chain-rule structure as
    mixlin_energy with the joint being just y."""

    def one(eo, lo, le, m, W):
        eta_q = jnp.einsum("kij,kj->ki", W, m) + eo[None]
        lam_q = W + lo[None]
        d = lam_q.shape[-1]
        ridge_eye = _ridge(lam_q.dtype) * jnp.eye(d, dtype=lam_q.dtype)
        log_rho = (
            le + _log_z(eta_q, lam_q)
            - 0.5 * jnp.einsum("ki,kij,kj->k", m, W, m)
            + 0.5 * jnp.linalg.slogdet(W)[1]
            - 0.5 * d * _LOG2PI
        )
        rho = jax.nn.softmax(log_rho)
        h_s = -(rho * jax.nn.log_softmax(log_rho)).sum()
        _, logdet_q = jnp.linalg.slogdet(lam_q + ridge_eye)
        h_gauss = 0.5 * d * (1.0 + _LOG2PI) - 0.5 * logdet_q
        sig_q = jnp.linalg.inv(lam_q + ridge_eye)
        m_q = jnp.einsum("kij,kj->ki", sig_q, eta_q)
        r = m_q - m
        _, logdet_w = jnp.linalg.slogdet(W)
        e_logf = (
            0.5 * logdet_w
            - 0.5 * d * _LOG2PI
            - 0.5 * (jnp.einsum("ki,kij,kj->k", r, W, r) + jnp.einsum("kij,kji->k", W, sig_q))
        )
        ok = _pd_ok(lam_q).all()
        f = -h_s - (rho * h_gauss).sum() - (rho * e_logf).sum()
        return jnp.where(ok, f, jnp.inf)

    return jax.vmap(one)(eta_o, lam_o, log_mu_e, mean, w)

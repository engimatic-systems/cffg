"""Gaussian message-update and node-energy rules (cff-4e5z).

Messages on Gaussian edges live in information (canonical) form,

    mu(x)  ∝  exp(−½ xᵀ Λ x + ηᵀ x),           parameters (η [d], Λ [d,d]),

because the two operations message passing needs are trivial there: products
of messages add their (η, Λ), and the *vacuous* message (a half-edge, an
unbonded glue port, an uninitialized slot) is exactly (0, 0) — no infinities,
no special cases. Λ may be singular (an improper, partially-informative
message); that is a feature, and every rule below is derived so that it never
inverts an *incoming* Λ. The only matrices inverted are `noise + incoming`
combinations that are positive definite whenever the noise covariance is.

Derivations (recorded per AGENTS.md: durable math lives next to the code).
Both node rules come from one Gaussian integral. For f(y, x) =
N(y; Ax + b, V) with W = V⁻¹ (A [do,di], W ≻ 0):

FORWARD (marginalize x against mu_in = (η_i, Λ_i)):
    mu(y) = ∫ f(y,x) mu_in(x) dx.  The exponent is quadratic in x with
    curvature  S = AᵀWA + Λ_i  and linear term  AᵀW(y−b) + η_i; completing
    the square and integrating leaves a quadratic in y:
        Λ_out = W − W A S⁻¹ Aᵀ W
        η_out = Λ_out b + W A S⁻¹ η_i
    Checks: Λ_i = 0, A square invertible ⇒ Λ_out = 0 (vacuous propagates
    exactly, by Woodbury); Λ_i ≻ 0 ⇒ Λ_out = (V + A Λ_i⁻¹ Aᵀ)⁻¹, the
    covariance-form Kalman prediction. S can be singular while beliefs are
    still forming (vacuous Λ_i, rank-deficient A), so the solve is ridged
    with _RIDGE·I: transient messages stay finite, and on trees synchronous
    flooding overwrites every message each sweep, so the transient washes
    out; at a proper fixed point the ridge is an O(1e-10) relative
    perturbation. See LIMITATIONS (Gaussian numerics).

BACKWARD (marginalize y against mu_out = (η_o, Λ_o)):
    mu(x) = ∫ f(y,x) mu_out(y) dy.  Quadratic in y with curvature W + Λ_o
    (always PD) and linear term W(Ax+b) + η_o; integrating leaves
        G     = W − W (W + Λ_o)⁻¹ W          (= (V + Λ_o⁻¹)⁻¹ when Λ_o ≻ 0)
        Λ_in  = Aᵀ G A
        η_in  = Aᵀ ( W (W + Λ_o)⁻¹ η_o − G b )
    Checks: Λ_o = 0 ⇒ G = 0 ⇒ vacuous propagates exactly. Λ_o → ∞·I at an
    observation y* ⇒ G → W, η_in → AᵀW(y*−b): the observed rule below is
    this exact limit, taken symbolically.

OBSERVED OUT-EDGE (Gaussian DATA port — conditioning, not clamping):
    A continuous δ-clamp has −∞ differential entropy, so pinning y with a
    huge-precision message diverges the Bethe ledger. Instead the compiler
    marks the node observed and the rules substitute y* into f:
        message to in:  (AᵀW(y*−b), AᵀWA)          — finite, exact
        message to out: the forward predictive     — readout/innovation only
                                                      (the edge is exterior)
    and the node energy uses q(x) ∝ N(y*; Ax+b, V) mu_in(x), which makes the
    tree identity  BFE = −log p(y*)  hold exactly. A missing observation
    (mask 0) removes the node: vacuous message to in, zero energy
    (∫ f dy = 1 contributes nothing to x).

NODE ENERGIES  F[q_a] = E_{q_a}[log q_a − log f_a],  q_a ∝ f_a · Π incoming:
    all closed-form. For a Gaussian belief with precision Λ_q,
        −H[q] = ½ log det Λ_q − d/2 (1 + log 2π)
    and E_q[log N(r; 0, V)] = ½ log det W − d/2 log 2π − ½ (r̄ᵀWr̄ + tr(WΣ_r)).
    The `gaussian_linear` joint belief q(y,x) has precision
        Λ_j = [[W+Λ_o, −WA], [−AᵀW, AᵀWA+Λ_i]],   η_j = [Wb+η_o, −AᵀWb+η_i]
    (read the exponent of f·mu_o·mu_i), and the residual r = y − Ax − b has
    r̄ = J m_j − b, Σ_r = J Σ_j Jᵀ with J = [I, −A]. Equality nodes carry
    −H[q] with log f = 0 on their support, mirroring the discrete rule; the
    Bethe edge-entropy correction (+H per interior edge, engine.py) then
    makes F = −log Z exact on trees — the continuous tree identity that the
    Kalman tests pin down.

Conventions: rules are batched over a group's members ([M, ...] leading
axis) via vmap; dims are static per group; the engine slices [:d] out of the
d_max-padded arena before calling and zero-pads the results after. Energies
are only meaningful at (near-)fixed points with proper beliefs — improper
beliefs surface as ±inf, deliberately loud.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_RIDGE = 1e-10
_LOG2PI = 1.8378770664093453  # log(2*pi)


def _solve(mat: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    """Ridged symmetric solve: finite on transiently singular systems."""
    d = mat.shape[-1]
    return jnp.linalg.solve(mat + _RIDGE * jnp.eye(d, dtype=mat.dtype), rhs)


def _neg_entropy(lam_q: jnp.ndarray) -> jnp.ndarray:
    """−H[q] for precision Λ_q: ½ log det Λ_q − d/2 (1 + log 2π)."""
    d = lam_q.shape[-1]
    _, logdet = jnp.linalg.slogdet(lam_q)
    return 0.5 * logdet - 0.5 * d * (1.0 + _LOG2PI)


# --- BP messages -------------------------------------------------------------


def prior_message(eta_p, w):
    """gaussian node → out: the constant message (η, Λ) = (Wm, W)."""
    return eta_p, w


def linear_forward(eta_i, lam_i, a, b, w):
    """gaussian_linear → out. [M,·] batched; see module docstring."""

    def one(ei, li, A, bb, W):
        s = A.T @ W @ A + li
        c_eta = _solve(s, ei)  # S⁻¹ η_i
        c_awt = _solve(s, A.T @ W)  # S⁻¹ AᵀW
        lam = W - W @ A @ c_awt
        eta = lam @ bb + W @ A @ c_eta
        return eta, lam

    return jax.vmap(one)(eta_i, lam_i, a, b, w)


def linear_backward(eta_o, lam_o, a, b, w):
    """gaussian_linear → in. (W + Λ_o) is PD: no ridge needed, but harmless."""

    def one(eo, lo, A, bb, W):
        k = _solve(W + lo, W)  # (W+Λo)⁻¹ W
        g = W - W @ k  # = W − W(W+Λo)⁻¹W, symmetric PSD
        lam = A.T @ g @ A
        eta = A.T @ (k.T @ eo - g @ bb)
        return eta, lam

    return jax.vmap(one)(eta_o, lam_o, a, b, w)


def linear_observed_backward(y, mask, a, b, w):
    """Observed gaussian_linear → in: the exact conditioning limit.
    mask ∈ {0,1} per member; 0 = missing = vacuous (node absent)."""

    def one(yy, mk, A, bb, W):
        eta = A.T @ (W @ (yy - bb))
        lam = A.T @ W @ A
        return mk * eta, mk * lam

    return jax.vmap(one)(y, mask, a, b, w)


def equality_messages(eta, lam):
    """equality → each leg: sum of the other legs' (η, Λ).
    eta [M,R,d], lam [M,R,d,d] → same shapes."""
    tot_e, tot_l = eta.sum(axis=1, keepdims=True), lam.sum(axis=1, keepdims=True)
    return tot_e - eta, tot_l - lam


# --- Node energies F[q_a] ----------------------------------------------------


def prior_energy(eta_i, lam_i, mean, w):
    """gaussian node: q ∝ N(m,V)·mu_in with (η_q, Λ_q) = (Wm+η_i, W+Λ_i)."""

    def one(ei, li, m, W):
        lam_q = W + li
        m_q = _solve(lam_q, W @ m + ei)
        sig_q = jnp.linalg.inv(lam_q + _RIDGE * jnp.eye(W.shape[0], dtype=W.dtype))
        _, logdet_w = jnp.linalg.slogdet(W)
        d = m - m_q
        e_logf = 0.5 * logdet_w - 0.5 * W.shape[0] * _LOG2PI - 0.5 * (
            d @ W @ d + jnp.trace(W @ sig_q)
        )
        return _neg_entropy(lam_q) - e_logf

    return jax.vmap(one)(eta_i, lam_i, mean, w)


def linear_energy(eta_o, lam_o, eta_i, lam_i, a, b, w):
    """gaussian_linear (unobserved): joint belief over (out, in)."""

    def one(eo, lo, ei, li, A, bb, W):
        do, di = A.shape
        lam_j = jnp.block([[W + lo, -W @ A], [-A.T @ W, A.T @ W @ A + li]])
        eta_j = jnp.concatenate([W @ bb + eo, -A.T @ W @ bb + ei])
        sig_j = jnp.linalg.inv(lam_j + _RIDGE * jnp.eye(do + di, dtype=W.dtype))
        m_j = sig_j @ eta_j
        j = jnp.concatenate([jnp.eye(do, dtype=W.dtype), -A], axis=1)  # r = J[y;x] − b
        r = j @ m_j - bb
        sig_r = j @ sig_j @ j.T
        _, logdet_w = jnp.linalg.slogdet(W)
        e_logf = 0.5 * logdet_w - 0.5 * do * _LOG2PI - 0.5 * (
            r @ W @ r + jnp.trace(W @ sig_r)
        )
        return _neg_entropy(lam_j) - e_logf

    return jax.vmap(one)(eta_o, lam_o, eta_i, lam_i, a, b, w)


def linear_observed_energy(eta_i, lam_i, y, mask, a, b, w):
    """Observed gaussian_linear: q(x) ∝ N(y*; Ax+b, V)·mu_in(x); the energy is
    the node's likelihood contribution to −log p(y*). When the observation is
    missing (mask 0) only the likelihood part drops: the node belief becomes
    f(y|x)·mu_in(x), whose energy telescopes to −H[mu_in] (same cancellation
    as the discrete missing-data rule) — exactly what cancels the interior
    in-edge's +H in the Bethe ledger, keeping BFE = −log p(bound ys)."""

    def one(ei, li, yy, mk, A, bb, W):
        do, di = A.shape
        lam_q = mk * (A.T @ W @ A) + li
        m_q = _solve(lam_q, mk * (A.T @ (W @ (yy - bb))) + ei)
        sig_q = jnp.linalg.inv(lam_q + _RIDGE * jnp.eye(di, dtype=W.dtype))
        r = yy - A @ m_q - bb
        _, logdet_w = jnp.linalg.slogdet(W)
        e_logf = 0.5 * logdet_w - 0.5 * do * _LOG2PI - 0.5 * (
            r @ W @ r + jnp.trace(W @ (A @ sig_q @ A.T))
        )
        return _neg_entropy(lam_q) - mk * e_logf

    return jax.vmap(one)(eta_i, lam_i, y, mask, a, b, w)


def equality_energy(eta, lam):
    """equality: −H[q] from the product of all legs (log f = 0 on support)."""
    return _neg_entropy(lam.sum(axis=1))


def edge_entropy(lam0, lam1):
    """+H[q_e] per interior edge (Bethe counting number d_e − 1 = 1)."""
    return -_neg_entropy(lam0 + lam1)


# --- Readout -----------------------------------------------------------------


def moments(eta, lam):
    """(mean, cov) from information form; caller guarantees Λ ≻ 0."""
    d = lam.shape[-1]
    cov = jnp.linalg.inv(lam + _RIDGE * jnp.eye(d, dtype=lam.dtype))
    return jnp.einsum("...ij,...j->...i", cov, eta), cov

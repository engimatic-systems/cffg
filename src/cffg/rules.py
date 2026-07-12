"""Message-update and node-energy rules, keyed by factor type.

Sum-product (BP) updates in log space, vectorized over a group's members.
Every rule ships with its node-energy function so the global Bethe free energy
is structurally the sum of registered local terms (AGENTS.md invariant): for a
node a, F[q_a] = Σ q_a (log q_a − log f_a) with q_a ∝ f_a · Π incoming.

Conventions: `inc` is [M, R, k_max] gathered incoming log-messages (role order
fixed by the FactorType); returns are [M, R, k_max], each row normalized over
its role's cardinality and NEG-padded beyond it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from cffg.compiler import NEG

_TINY = 1e-12


def _norm_pad(x: jnp.ndarray, card: int, k_max: int) -> jnp.ndarray:
    """Normalize [M, card] log-messages and pad to k_max."""
    x = x - logsumexp(x, axis=-1, keepdims=True)
    pad = jnp.full((*x.shape[:-1], k_max - card), NEG, dtype=x.dtype)
    return jnp.concatenate([x, pad], axis=-1)


# --- BP updates ------------------------------------------------------------


def bp_categorical(inc, logf, cards, k_max):
    (k,) = cards
    return _norm_pad(logf, k, k_max)[:, None, :]


def bp_matrix_cat(inc, logf, cards, k_max):
    ko, ki = cards
    m_out, m_in = inc[:, 0, :ko], inc[:, 1, :ki]
    to_out = logsumexp(logf + m_in[:, None, :], axis=2)
    to_in = logsumexp(logf + m_out[:, :, None], axis=1)
    return jnp.stack([_norm_pad(to_out, ko, k_max), _norm_pad(to_in, ki, k_max)], axis=1)


def bp_transition_mixture(inc, logf, cards, k_max):
    ko, ki, ku = cards  # logf: [M, ko, ki, ku]
    m_out, m_in, m_sel = inc[:, 0, :ko], inc[:, 1, :ki], inc[:, 2, :ku]
    to_out = logsumexp(logf + m_in[:, None, :, None] + m_sel[:, None, None, :], axis=(2, 3))
    to_in = logsumexp(logf + m_out[:, :, None, None] + m_sel[:, None, None, :], axis=(1, 3))
    to_sel = logsumexp(logf + m_out[:, :, None, None] + m_in[:, None, :, None], axis=(1, 2))
    return jnp.stack(
        [_norm_pad(to_out, ko, k_max), _norm_pad(to_in, ki, k_max), _norm_pad(to_sel, ku, k_max)],
        axis=1,
    )


def bp_equality(inc, logf, cards, k_max):
    k = cards[0]
    a, b, c = inc[:, 0, :k], inc[:, 1, :k], inc[:, 2, :k]
    return jnp.stack(
        [_norm_pad(b + c, k, k_max), _norm_pad(a + c, k, k_max), _norm_pad(a + b, k, k_max)],
        axis=1,
    )


BP_UPDATES = {
    "categorical": bp_categorical,
    "matrix_cat": bp_matrix_cat,
    "transition_mixture": bp_transition_mixture,
    "equality": bp_equality,
}


# --- Node energies F[q_a] ---------------------------------------------------


def _energy_from(log_belief_un: jnp.ndarray, log_f: jnp.ndarray) -> jnp.ndarray:
    """F[q_a] per member from unnormalized log-belief and log-factor, both
    [M, ...]. 0·log0 terms are excised with a where-guard."""
    m = log_belief_un.shape[0]
    lb = log_belief_un.reshape(m, -1)
    lf = log_f.reshape(m, -1)
    lb = lb - logsumexp(lb, axis=-1, keepdims=True)
    q = jnp.exp(lb)
    return jnp.where(q > _TINY, q * (lb - lf), 0.0).sum(axis=-1)


def energy_categorical(inc, logf, cards):
    (k,) = cards
    return _energy_from(logf + inc[:, 0, :k], logf)


def energy_matrix_cat(inc, logf, cards):
    ko, ki = cards
    lb = logf + inc[:, 0, :ko, None] + inc[:, 1, None, :ki]
    return _energy_from(lb, logf)


def energy_transition_mixture(inc, logf, cards):
    ko, ki, ku = cards
    lb = (
        logf
        + inc[:, 0, :ko, None, None]
        + inc[:, 1, None, :ki, None]
        + inc[:, 2, None, None, :ku]
    )
    return _energy_from(lb, logf)


def energy_equality(inc, logf, cards):
    # f is the identity on the diagonal; the belief lives on the diagonal too,
    # so F[q_a] = Σ q log q (log f = 0 on support).
    k = cards[0]
    lb = inc[:, 0, :k] + inc[:, 1, :k] + inc[:, 2, :k]
    return _energy_from(lb, jnp.zeros_like(lb))


NODE_ENERGIES = {
    "categorical": energy_categorical,
    "matrix_cat": energy_matrix_cat,
    "transition_mixture": energy_transition_mixture,
    "equality": energy_equality,
}


# --- VMP (mean-field bead): geometric messages -------------------------------
#
# `inc2` is [M, R, 2, k_max]: both slots of each adjacent edge, so the rule
# works with edge *marginals* q(s_i) instead of the opposite message. Outgoing
# to edge i is exp(E_{q(others)}[log f]).


def _edge_q(inc2, r, card):
    lq = inc2[:, r, 0, :card] + inc2[:, r, 1, :card]
    lq = lq - logsumexp(lq, axis=-1, keepdims=True)
    return jnp.exp(lq), lq


def vmp_categorical(inc2, logf, cards, k_max):
    (k,) = cards
    return _norm_pad(logf, k, k_max)[:, None, :]


def vmp_matrix_cat(inc2, logf, cards, k_max):
    ko, ki = cards
    q_out, _ = _edge_q(inc2, 0, ko)
    q_in, _ = _edge_q(inc2, 1, ki)
    to_out = jnp.einsum("mi,moi->mo", q_in, logf)
    to_in = jnp.einsum("mo,moi->mi", q_out, logf)
    return jnp.stack([_norm_pad(to_out, ko, k_max), _norm_pad(to_in, ki, k_max)], axis=1)


def vmp_transition_mixture(inc2, logf, cards, k_max):
    ko, ki, ku = cards  # logf: [M, ko, ki, ku]
    q_out, _ = _edge_q(inc2, 0, ko)
    q_in, _ = _edge_q(inc2, 1, ki)
    q_sel, _ = _edge_q(inc2, 2, ku)
    to_out = jnp.einsum("mi,mu,moiu->mo", q_in, q_sel, logf)
    to_in = jnp.einsum("mo,mu,moiu->mi", q_out, q_sel, logf)
    to_sel = jnp.einsum("mo,mi,moiu->mu", q_out, q_in, logf)
    return jnp.stack(
        [_norm_pad(to_out, ko, k_max), _norm_pad(to_in, ki, k_max), _norm_pad(to_sel, ku, k_max)],
        axis=1,
    )


VMP_UPDATES = {
    "categorical": vmp_categorical,
    "matrix_cat": vmp_matrix_cat,
    "transition_mixture": vmp_transition_mixture,
}


def _neg_entropy(q, lq):
    return jnp.where(q > _TINY, q * lq, 0.0).sum(axis=-1)


def vmp_energy_categorical(inc2, logf, cards):
    (k,) = cards
    q, lq = _edge_q(inc2, 0, k)
    return _neg_entropy(q, lq) - (q * logf).sum(axis=-1)


def vmp_energy_matrix_cat(inc2, logf, cards):
    ko, ki = cards
    q_out, lq_out = _edge_q(inc2, 0, ko)
    q_in, lq_in = _edge_q(inc2, 1, ki)
    e = jnp.einsum("mo,mi,moi->m", q_out, q_in, logf)
    return _neg_entropy(q_out, lq_out) + _neg_entropy(q_in, lq_in) - e


def vmp_energy_transition_mixture(inc2, logf, cards):
    ko, ki, ku = cards
    q_out, lq_out = _edge_q(inc2, 0, ko)
    q_in, lq_in = _edge_q(inc2, 1, ki)
    q_sel, lq_sel = _edge_q(inc2, 2, ku)
    e = jnp.einsum("mo,mi,mu,moiu->m", q_out, q_in, q_sel, logf)
    return (
        _neg_entropy(q_out, lq_out)
        + _neg_entropy(q_in, lq_in)
        + _neg_entropy(q_sel, lq_sel)
        - e
    )


VMP_ENERGIES = {
    "categorical": vmp_energy_categorical,
    "matrix_cat": vmp_energy_matrix_cat,
    "transition_mixture": vmp_energy_transition_mixture,
}


# --- GFE composite (P-substituted goal-observation pair, Part II §5) ---------
#
# Stable backward state message via Corollary 1: solve the fixed point
# z* = softmax(rho(z*) + log d) with a gauge-fixed Newton iteration in logit
# space, where rho(z) = A^T (log c − log(Az)) − h(A) and h(A) is the vector of
# column entropies. The composite's local free-energy contribution is
# G = −z*ᵀ rho(z*) − H[z*]  (the goal node's term and the x-edge entropy are
# absorbed — see cff-007q for the derivation).

_N_NEWTON = 25


def _safe_log(x):
    """log clamped at the dtype's smallest normal. A literal 1e-300 floor is
    a silent no-op under float32 (it underflows to 0), and log(0) = −inf
    turns 0·inf into NaN one matmul later — the engine only ever ran f64 in
    tests, so this bit consumers on JAX's default f32 first (cff-i45j)."""
    return jnp.log(jnp.maximum(x, jnp.finfo(jnp.result_type(x)).tiny))


def _fixed_stats(logA):
    """(Ā, h) for a known matrix: Ā = A, h = column entropies."""
    a = jnp.exp(logA)
    h = -jnp.sum(jnp.where(a > _TINY, a * logA, 0.0), axis=0)  # [K]
    return a, h


def _dirichlet_stats(counts):
    """(Ā, h̃) under per-column Dirichlet q(A): Ā = E[A] = α/α₀, and the
    coupled term E[A_xz log A_xz] = (α/α₀)(ψ(α+1) − ψ(α₀+1)) in closed form,
    so h̃_z = −Σ_x E[A log A]. By Jensen h̃ ≤ H[Ā·z-column]: an *uncertain*
    column has lower expected entropy than its blurry mean suggests — the
    parameter-novelty channel that makes unexplored states attractive."""
    a0 = counts.sum(axis=0, keepdims=True)
    abar = counts / a0
    e_alog = abar * (jax.scipy.special.digamma(counts + 1.0) - jax.scipy.special.digamma(a0 + 1.0))
    h = -e_alog.sum(axis=0)  # [K]
    return abar, h


def _gfe_solve(abar, h, logc, log_d):
    """Single composite: z* in the simplex. abar [Mo,K], h [K], logc [Mo].

    log_d is clamped at −1e3 before the solve: exp(−1000) underflows to an
    exact 0 in f64, so the fixed point is unchanged, but Newton's residual
    and Jacobian stay O(1e3) instead of O(1e30) when the incoming message
    carries exact log-zeros (sparse supports, cff-i45j)."""
    log_d = jnp.maximum(log_d, -1e3)

    def rho(zbar):
        qx = abar @ zbar
        return abar.T @ (logc - _safe_log(qx)) - h

    def center(v):
        return v - v.mean()

    def resid(v):
        return center(v) - center(rho(jax.nn.softmax(v)) + log_d)

    # Newton with backtracking: the full step can overshoot and limit-cycle
    # on large simplexes (observed at a few hundred states, cff-i45j), so
    # each iteration tries fractional steps and keeps the best residual —
    # near the fixed point the full step wins and quadratic convergence is
    # untouched (the small-model goldens pin this).
    _TS = jnp.array([1.0, 0.5, 0.25, 0.125, 0.03125])

    def newton(_, v):
        r = resid(v)
        j = jax.jacfwd(resid)(v)
        k = v.shape[0]
        # The centered residual is gauge-invariant along 1, so j always has a
        # zero eigenvalue there; the ridge must sit above the dtype's
        # roundoff (a fixed 1e-9 is *below* f32 eps — numerically singular,
        # solve returns inf). Re-centering the step kills any gauge drift.
        ridge = jnp.sqrt(jnp.finfo(v.dtype).eps)
        full = jnp.linalg.solve(j + ridge * jnp.eye(k, dtype=v.dtype), r)
        full = full - full.mean()
        cands = v[None, :] - _TS[:, None].astype(v.dtype) * full[None, :]
        norms = jax.vmap(lambda vv: jnp.sum(resid(vv) ** 2))(cands)
        return cands[jnp.argmin(norms)]

    v = jax.lax.fori_loop(0, _N_NEWTON, newton, center(log_d))
    return jax.nn.softmax(v)


def _gfe_rho(abar, h, logc, zbar):
    qx = abar @ zbar
    return abar.T @ (logc - _safe_log(qx)) - h


def _stats(a_param, learned):
    return _dirichlet_stats(a_param) if learned else _fixed_stats(a_param)


def gfe_update(log_d, a_param, logc, k_max, learned=False):
    """Batched over members. log_d [M,K] normalized; a_param is logA (fixed)
    or Dirichlet counts α (learned); returns (μ2, log q_x) padded to k_max.

    μ2 is computed in the ρ form: the fixed point gives log z̄ = ρ(z̄) +
    log d − log Z, so μ2 = log z̄ − log d = ρ(z̄) − log Z, and _norm_pad
    absorbs the constant — identical to the subtraction form wherever
    d > 0, but bounded on zero-support states, where log z̄ − log d is a
    (−∞) − (−∞) that used to surface as ±1e30 junk and leak probability
    into downstream updates (cff-i45j)."""
    ko, ki = a_param.shape[1], a_param.shape[2]

    def one(a_p, lc, ld):
        abar, h = _stats(a_p, learned)
        zbar = _gfe_solve(abar, h, lc, ld)
        return _gfe_rho(abar, h, lc, zbar), abar @ zbar

    mu2, qx = jax.vmap(one)(a_param, logc, log_d)
    log_qx = _safe_log(qx)
    return _norm_pad(mu2, ki, k_max), _norm_pad(log_qx, ko, k_max)


def gfe_energy(log_d, a_param, logc, learned=False):
    """Composite FE term per member: −z*ᵀ rho(z*) − H[z*]."""

    def one(a_p, lc, ld):
        abar, h = _stats(a_p, learned)
        zbar = _gfe_solve(abar, h, lc, ld)
        rho = _gfe_rho(abar, h, lc, zbar)
        lz = _safe_log(zbar)
        return -(zbar * rho).sum(-1) + (zbar * lz).sum(-1)

    return jax.vmap(one)(a_param, logc, log_d)


# --- GFE-TM composite (P-substituted transition node, cff-fk2u) --------------
#
# The dynamics-curiosity channel. P-substitution on a transition_mixture node
# f(z', z, u) = p(z'|z,u,B) with per-action-column Dirichlet q(B): replace the
# node's conditional belief q(z'|z,u) with the model's own (parameter-averaged)
# conditional, exactly as Part II does for the observation node. The node-local
# BFE term F_a = E[log q_a − log f_a] becomes, under the substituted joint
# q̃(z',z,u,B) = q(z,u)·q(B)·p(z'|z,u,B) and with the joint entropy split as
# log q̃ → log q_w(z') + log q(z,u):
#
#   G_tm = −H[q_w] + E_{q(z,u)}[ĥ(z,u)] − H[q(z,u)]
#        = −I(z' ; (z,u), B) − H[q(z,u)]
#
# with q_w(z') = Σ_{z,u} q(z,u)·B̄_u[z',z] (the predictive next-state marginal),
# B̄ = E[B], and ĥ(z,u) = −Σ_{z'} E_q(B)[B log B] (the expected column pseudo-
# entropy — the same closed-form ψ(α+1)−ψ(α₀+1) channel as the A-composite).
# The mutual information decomposes as I(z'; z,u) + I(z'; B | z,u): a
# state-transition term and the PARAMETER NOVELTY term that makes low-count
# (z,u) columns attractive. Both are free — the Dirichlet algebra transfers
# from _dirichlet_stats untouched.
#
# Stationarity of the local Lagrangian (incoming messages μ_in(z), μ_e(u) as
# linear couplings; the downstream message μ_down(z') couples through q_w
# because the composite's z'-marginal IS q_w) gives the fixed point
#
#   q̄(z,u) = softmax( ρ(q̄)(z,u) + μ_in(z) + μ_e(u) )
#   ρ(z,u)  = Σ_{z'} B̄_u[z',z]·(μ_down(z') − log q_w(z')) − ĥ(z,u)
#
# — Corollary 1's equation over the JOINT (z,u) simplex, with μ_down playing
# the role log c played at the observation leaf. The observation composite is
# the special case where the substituted edge faces a goal prior and s = z.
# _gfe_solve/_gfe_rho are therefore reused verbatim with
#   abar := B̄ as a [K', K·U] column matrix,  h := ĥ flattened,
#   logc := μ_down,                          log_d := μ_in ⊕ μ_e.
#
# Outgoing messages (log q̄ = ρ̄ + μ_in + μ_e − log Z, so the node's opinion
# toward each edge drops that edge's own incoming term and marginalizes —
# the additive incoming message cancels exactly, keeping every output bounded
# on sparse supports, per the cff-i45j lesson):
#   to selector: logsumexp_z( ρ̄(z,u) + μ_in(z) )   ← the action-curiosity score
#   to in:       logsumexp_u( ρ̄(z,u) + μ_e(u) )
#   to out:      log q_w                             (substituted predictive)
# FE term: G_tm = −q̄ᵀρ̄ − H[q̄], the same closed form as the A-composite.
#
# HONESTY NOTE (ledger semantics, LIMITATIONS S1c): z' is never observed
# directly, so I(z'; B | z,u) scores information gain as if the next state
# were revealed. It is an OPTIMISTIC BOUND on realizable info gain about B —
# tight when observations pin the state down (good sensors), loose otherwise.
# This is the same semantics as the field's standard count-novelty bonus, but
# derived inside the constrained-Bethe machinery rather than bolted onto the
# objective. Cost note: the Newton solve is over K·U logits, O((K·U)³) per
# member per sweep — fine for planning-sized models, a structure-exploitation
# ticket at colony scale.


def _tm_fixed_stats(log_b):
    """(B̄, ĥ) for a known stack log B [U, K', K]: ĥ = per-(z,u) column entropy."""
    b = jnp.exp(log_b)
    h = -jnp.sum(jnp.where(b > _TINY, b * log_b, 0.0), axis=1)  # [U, K]
    return b, h


def _tm_dirichlet_stats(counts):
    """(B̄, ĥ) under per-(action, column) Dirichlets, counts [U, K', K]:
    B̄ = α/α₀ and E[B log B] = (α/α₀)(ψ(α+1) − ψ(α₀+1)) in closed form, so
    ĥ ≤ H[B̄-column] with the gap = the parameter-novelty channel (unexplored
    transition columns look attractive), mirroring _dirichlet_stats."""
    a0 = counts.sum(axis=1, keepdims=True)
    bbar = counts / a0
    e_blog = bbar * (
        jax.scipy.special.digamma(counts + 1.0) - jax.scipy.special.digamma(a0 + 1.0)
    )
    h = -e_blog.sum(axis=1)  # [U, K]
    return bbar, h


def _tm_joint_stats(b_param, learned):
    """Flatten [U, K', K] stats onto the joint column index j = z·U + u,
    matching (μ_in ⊕ μ_e).reshape(-1): abar [K', K·U], h [K·U]."""
    bbar, h = _tm_dirichlet_stats(b_param) if learned else _tm_fixed_stats(b_param)
    kp, ku = bbar.shape[1], bbar.shape[0]
    kz = bbar.shape[2]
    abar = bbar.transpose(1, 2, 0).reshape(kp, kz * ku)
    return abar, h.T.reshape(kz * ku)


def gfe_tm_update(mu_in, mu_e, mu_down, b_param, k_max, learned=False):
    """Batched over members. mu_in [M,K], mu_e [M,U], mu_down [M,K']
    normalized log-messages; b_param is log B (fixed) or Dirichlet counts α
    (learned), [M, U, K', K]. Returns (to_out, to_in, to_sel) padded."""
    ku, kp, kz = b_param.shape[1], b_param.shape[2], b_param.shape[3]

    def one(bp, mi, me, md):
        abar, h = _tm_joint_stats(bp, learned)
        log_d = (mi[:, None] + me[None, :]).reshape(-1)
        qj = _gfe_solve(abar, h, md, log_d)
        rho = _gfe_rho(abar, h, md, qj).reshape(kz, ku)
        to_sel = logsumexp(rho + mi[:, None], axis=0)
        to_in = logsumexp(rho + me[None, :], axis=1)
        to_out = _safe_log(abar @ qj)
        return to_out, to_in, to_sel

    to_out, to_in, to_sel = jax.vmap(one)(b_param, mu_in, mu_e, mu_down)
    return (
        _norm_pad(to_out, kp, k_max),
        _norm_pad(to_in, kz, k_max),
        _norm_pad(to_sel, ku, k_max),
    )


def gfe_tm_energy(mu_in, mu_e, mu_down, b_param, learned=False):
    """Composite FE term per member: G_tm = −q̄ᵀρ(q̄) − H[q̄] (a GFE term —
    optimistic-bound semantics, see the section note)."""

    def one(bp, mi, me, md):
        abar, h = _tm_joint_stats(bp, learned)
        log_d = (mi[:, None] + me[None, :]).reshape(-1)
        qj = _gfe_solve(abar, h, md, log_d)
        rho = _gfe_rho(abar, h, md, qj)
        return -(qj * rho).sum(-1) + (qj * _safe_log(qj)).sum(-1)

    return jax.vmap(one)(b_param, mu_in, mu_e, mu_down)

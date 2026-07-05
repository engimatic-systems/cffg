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


def _gfe_solve(logA, logc, log_d):
    """Single composite: z* in the simplex. logA [Mo,K], logc [Mo], log_d [K]."""
    a = jnp.exp(logA)
    h = -jnp.sum(jnp.where(a > _TINY, a * logA, 0.0), axis=0)  # [K]

    def rho(zbar):
        qx = a @ zbar
        return a.T @ (logc - jnp.log(jnp.maximum(qx, 1e-300))) - h

    def center(v):
        return v - v.mean()

    def resid(v):
        return center(v) - center(rho(jax.nn.softmax(v)) + log_d)

    def newton(_, v):
        j = jax.jacfwd(resid)(v)
        k = v.shape[0]
        return v - jnp.linalg.solve(j + 1e-9 * jnp.eye(k, dtype=v.dtype), resid(v))

    v = jax.lax.fori_loop(0, _N_NEWTON, newton, center(log_d))
    return jax.nn.softmax(v)


def _gfe_rho(logA, logc, zbar):
    a = jnp.exp(logA)
    h = -jnp.sum(jnp.where(a > _TINY, a * logA, 0.0), axis=0)
    qx = a @ zbar
    return a.T @ (logc - jnp.log(jnp.maximum(qx, 1e-300))) - h


def gfe_update(log_d, logA, logc, k_max):
    """Batched over members. log_d [M,K] normalized; returns (μ2, log q_x)
    padded to k_max."""
    ko, ki = logA.shape[1], logA.shape[2]
    zbar = jax.vmap(_gfe_solve)(logA, logc, log_d)  # [M, K]
    mu2 = jnp.log(zbar) - log_d
    qx = jnp.einsum("mok,mk->mo", jnp.exp(logA), zbar)
    log_qx = jnp.log(jnp.maximum(qx, 1e-300))
    return _norm_pad(mu2, ki, k_max), _norm_pad(log_qx, ko, k_max)


def gfe_energy(log_d, logA, logc):
    """Composite FE term per member: −z*ᵀ rho(z*) − H[z*]."""
    zbar = jax.vmap(_gfe_solve)(logA, logc, log_d)
    rho = jax.vmap(_gfe_rho)(logA, logc, zbar)
    lz = jnp.log(jnp.maximum(zbar, 1e-300))
    return -(zbar * rho).sum(-1) + (zbar * lz).sum(-1)

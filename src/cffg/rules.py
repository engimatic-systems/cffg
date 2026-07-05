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

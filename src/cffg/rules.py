"""Message-update and node-energy rules, keyed by factor type.

Sum-product (BP) updates in log space, vectorized over a group's members.
Every rule ships with its node-energy function so the global Bethe free energy
is structurally the sum of registered local terms (AGENTS.md invariant): for a
node a, F[q_a] = Σ q_a (log q_a − log f_a) with q_a ∝ f_a · Π incoming.

Conventions: `inc` is [M, R, k_max] gathered incoming log-messages, where
`k_max` is local to one homogeneous factor group (not the model); returns have
the same local shape, with each row normalized over its role's cardinality and
NEG-padded beyond it.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln, logsumexp

from cffg.compiler import NEG

_TINY = 1e-12


def _norm_pad(x: jnp.ndarray, card: int, k_max: int) -> jnp.ndarray:
    """Normalize [M, card] log-messages and pad to k_max."""
    x = x - logsumexp(x, axis=-1, keepdims=True)
    pad = jnp.full((*x.shape[:-1], k_max - card), NEG, dtype=x.dtype)
    return jnp.concatenate([x, pad], axis=-1)


# --- BP updates ------------------------------------------------------------


# The marginalization reduction is a parameter (cff-e8zk): `op` = logsumexp for
# sum-product (default; byte-identical to the original rules) and jnp.max for
# min-sum / max-product. Only the sum-over-the-other-variables reduction swaps;
# _norm_pad's normalization stays logsumexp (an arbitrary constant shift under
# max-product, so it does not affect the decode). categorical/equality carry no
# such reduction, so `op` is inert there.


def bp_categorical(inc, logf, cards, k_max, op=logsumexp):
    (k,) = cards
    return _norm_pad(logf, k, k_max)[:, None, :]


def bp_matrix_cat(inc, logf, cards, k_max, op=logsumexp):
    ko, ki = cards
    m_out, m_in = inc[:, 0, :ko], inc[:, 1, :ki]
    to_out = op(logf + m_in[:, None, :], axis=2)
    to_in = op(logf + m_out[:, :, None], axis=1)
    return jnp.stack([_norm_pad(to_out, ko, k_max), _norm_pad(to_in, ki, k_max)], axis=1)


def bp_transition_mixture(inc, logf, cards, k_max, op=logsumexp):
    ko, ki, ku = cards  # logf: [M, ko, ki, ku]
    m_out, m_in, m_sel = inc[:, 0, :ko], inc[:, 1, :ki], inc[:, 2, :ku]
    to_out = op(logf + m_in[:, None, :, None] + m_sel[:, None, None, :], axis=(2, 3))
    to_in = op(logf + m_out[:, :, None, None] + m_sel[:, None, None, :], axis=(1, 3))
    to_sel = op(logf + m_out[:, :, None, None] + m_in[:, None, :, None], axis=(1, 2))
    return jnp.stack(
        [_norm_pad(to_out, ko, k_max), _norm_pad(to_in, ki, k_max), _norm_pad(to_sel, ku, k_max)],
        axis=1,
    )


def bp_equality(inc, logf, cards, k_max, op=logsumexp):
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


# --- VMP likelihoods with Dirichlet parameter edges (D6a) ---------------------


def _dirichlet_counts(ftype, inc2, cards):
    """Structural edge marginals -> fractional counts in serialized layout."""
    qs = tuple(_edge_q(inc2, r, card) for r, card in enumerate(cards))
    if ftype == "categorical":
        return qs[0][0], qs
    if ftype == "matrix_cat":
        return jnp.einsum("mo,mi->moi", qs[0][0], qs[1][0]), qs
    if ftype == "transition_mixture":
        # Runtime/node layout is [out, in, selector]; parameter layout is
        # [selector, out, in], matching blobs and Learnables.
        return jnp.einsum(
            "mu,mo,mi->muoi", qs[2][0], qs[0][0], qs[1][0]
        ), qs
    raise KeyError(ftype)


def _dirichlet_expected_log(alpha, output_axis):
    axis = output_axis + 1  # leading member axis
    return digamma(alpha) - digamma(jnp.sum(alpha, axis=axis, keepdims=True))


def dirichlet_expected_log(alpha, axis=-1):
    """Expected log simplex probabilities under a proper Dirichlet.

    Unlike :func:`_dirichlet_expected_log`, this public form has no assumed
    leading member axis and is used by GFE goal-hyperprior coordinates.
    """
    return digamma(alpha) - digamma(jnp.sum(alpha, axis=axis, keepdims=True))


def _dirichlet_logf(ftype, expected_log):
    if ftype == "transition_mixture":
        return expected_log.transpose(0, 2, 3, 1)  # [M,U,O,I] -> [M,O,I,U]
    return expected_log


def dirichlet_vmp_update(ftype, inc2, alpha0, cards, k_max):
    """One D6a coordinate update.

    The returned parameter message is a fractional count array, while
    structural messages use E_q[log theta] under ``alpha0 + counts``.  The
    compiler packs ``alpha0`` itself in the facing prior slot (rather than
    alpha0-1) so sparse positive float32 concentrations cannot cancel at one.
    """
    counts, _ = _dirichlet_counts(ftype, inc2, cards)
    output_axis = 1 if ftype == "transition_mixture" else 0
    alpha = alpha0 + counts
    expected_log = _dirichlet_expected_log(alpha, output_axis)
    out = VMP_UPDATES[ftype](
        inc2, _dirichlet_logf(ftype, expected_log), cards, k_max
    )
    return out, counts


def _sum_nonmember(x):
    return x.reshape((x.shape[0], -1)).sum(axis=-1)


def _dirichlet_log_beta(alpha, output_axis):
    axis = output_axis + 1
    return (
        gammaln(alpha).sum(axis=axis)
        - gammaln(alpha.sum(axis=axis))
    )


def _state_negative_entropy(qs):
    total = jnp.zeros(qs[0][0].shape[0], dtype=qs[0][0].dtype)
    for q, lq in qs:
        total = total + jnp.where(q > 0, q * lq, 0.0).sum(axis=-1)
    return total


def dirichlet_vmp_energy(ftype, inc2, alpha, alpha0, cards):
    """Stable complete D6a likelihood/prior/parameter-edge VFE per member.

    Combining the three parameter ledger terms before multiplication yields

        -log B(alpha) + log B(alpha0)
        + (alpha-alpha0-counts) dot E_q[log theta].

    At a coordinate fixed point the last coefficient is exactly zero.  This
    avoids separately forming O(1/alpha) terms that catastrophically cancel
    for a sparse float32 Dirichlet.
    """
    counts, qs = _dirichlet_counts(ftype, inc2, cards)
    output_axis = 1 if ftype == "transition_mixture" else 0
    elog = _dirichlet_expected_log(alpha, output_axis)
    parameter = (
        -_sum_nonmember(_dirichlet_log_beta(alpha, output_axis))
        + _sum_nonmember(_dirichlet_log_beta(alpha0, output_axis))
        + _sum_nonmember((alpha - alpha0 - counts) * elog)
    )
    proper = (
        jnp.isfinite(alpha).reshape((alpha.shape[0], -1)).all(axis=-1)
        & (alpha > 0).reshape((alpha.shape[0], -1)).all(axis=-1)
    )
    return jnp.where(proper, _state_negative_entropy(qs) + parameter, jnp.inf)


def dirichlet_vmp_energy_parts(ftype, inc2, alpha, alpha0, cards):
    """Literal node/edge ledger parts for review and small direct oracles.

    Returns ``(prior_node, likelihood_node, parameter_edge_correction)``.
    The engine uses :func:`dirichlet_vmp_energy`'s algebraically combined
    form for hostile sparse inputs; these parts expose the exact bookkeeping
    identity from DERIVATIONS D6a.3.
    """
    counts, qs = _dirichlet_counts(ftype, inc2, cards)
    output_axis = 1 if ftype == "transition_mixture" else 0
    elog = _dirichlet_expected_log(alpha, output_axis)
    q_neg_entropy = (
        -_sum_nonmember(_dirichlet_log_beta(alpha, output_axis))
        + _sum_nonmember((alpha - 1) * elog)
    )
    prior_energy = (
        _sum_nonmember(_dirichlet_log_beta(alpha0, output_axis))
        - _sum_nonmember((alpha0 - 1) * elog)
    )
    likelihood_energy = -_sum_nonmember(counts * elog)
    prior_node = q_neg_entropy + prior_energy
    likelihood_node = (
        q_neg_entropy + _state_negative_entropy(qs) + likelihood_energy
    )
    return prior_node, likelihood_node, -q_neg_entropy


# --- GFE composite (P-substituted goal-observation pair, Part II §5) ---------
#
# Stable backward state message via Corollary 1: solve the fixed point
# z* = softmax(rho(z*) + log d) with a gauge-fixed Newton iteration in logit
# space, where rho(z) = A^T (log c − log(Az)) − h(A) and h(A) is the vector of
# column entropies. The composite's local free-energy contribution is
# G = −z*ᵀ rho(z*) − H[z*]  (the goal node's term and the x-edge entropy are
# absorbed — see cff-007q for the derivation).

_N_NEWTON = 64
# The gauge-fixed softmax/Jacobian round trip can settle a few hundred ULPs
# above algebraic zero even when a recomputed solution no longer moves. 1024
# ULPs is the smallest power-of-two allowance covering the exact-support
# factored regression; deriving it from eps keeps the certificate meaningful
# in both f32 and f64 instead of introducing an absolute residual floor.
_GFE_SOLVE_TOL_ULPS = 1024.0


class GfeSolveResult(NamedTuple):
    """JAX-compatible diagnostics for one Corollary-1 Newton solve.

    ``status`` is an integer so the record can flow through ``jit``/``vmap``:
    0 = converged, 1 = iteration budget exhausted, 2 = diverged.  The outer
    runtime translates it to the public string-valued ``RunStatus``.
    """

    solution: jax.Array
    status: jax.Array
    iterations: jax.Array
    residual: jax.Array


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


def _gfe_solve_result(abar, h, logc, log_d, max_iters: int = _N_NEWTON):
    """Solve one composite and expose termination diagnostics.

    The solve is performed in a ``K-1`` dimensional gauge-fixed chart rather
    than regularising the singular all-ones direction.  Exact-zero incoming
    support is represented at the dtype's smallest normal log-probability
    while Newton runs; this bounds the residual without a dtype-specific
    literal and does not give that support mass in the engine's cavity update.
    """
    dtype = jnp.result_type(abar, h, logc, log_d)
    abar = jnp.asarray(abar, dtype=dtype)
    h = jnp.asarray(h, dtype=dtype)
    logc = jnp.asarray(logc, dtype=dtype)
    log_d = jnp.asarray(log_d, dtype=dtype)
    # Protect the Newton residual from the finite NEG sentinel.  This guard is
    # the smallest normal value representable by the active dtype, so it
    # scales correctly in both f32 and f64.
    log_floor = jnp.log(jnp.finfo(dtype).tiny)
    log_d = jnp.maximum(log_d, log_floor)

    def rho(zbar):
        qx = abar @ zbar
        return abar.T @ (logc - _safe_log(qx)) - h

    def center(v):
        return v - v.mean()

    def resid(v):
        return center(v) - center(rho(jax.nn.softmax(v)) + log_d)

    k = log_d.shape[0]
    # K=1 has no free coordinate and is stationary by construction.
    if k == 1:
        zero = jnp.asarray(0.0, dtype=dtype)
        return GfeSolveResult(jnp.ones_like(log_d), jnp.asarray(0), jnp.asarray(0), zero)

    def unpack(u):
        return center(jnp.concatenate([u, jnp.zeros((1,), dtype=dtype)]))

    def reduced_resid(u):
        return resid(unpack(u))[:-1]

    # Newton with globalization: a zero-length candidate makes every step
    # non-increasing even when the local linear model is poor.  Step fractions
    # are dimensionless; numeric validity guards come only from finfo(dtype).
    fractions = jnp.asarray([1.0, 0.5, 0.25, 0.125, 0.0625, 0.0], dtype=dtype)
    # Below this dtype-scaled recomputation floor Newton can only repeat a
    # zero-length backtracking step.
    tol = (
        jnp.asarray(_GFE_SOLVE_TOL_ULPS, dtype=dtype)
        * jnp.finfo(dtype).eps
    )

    v0 = center(log_d)
    u0 = v0[:-1] - v0[-1]

    def residual_norm(u):
        return jnp.max(jnp.abs(reduced_resid(u)))

    r0 = residual_norm(u0)

    def cond(carry):
        i, _, rn = carry
        return (i < max_iters) & jnp.isfinite(rn) & (rn > tol)

    def body(carry):
        i, u, _ = carry
        r = reduced_resid(u)
        jac = jax.jacfwd(reduced_resid)(u)
        full = jnp.linalg.solve(jac, r)
        cands = u[None, :] - fractions[:, None] * full[None, :]
        norms = jax.vmap(residual_norm)(cands)
        best = jnp.argmin(jnp.where(jnp.isfinite(norms), norms, jnp.inf))
        next_u = cands[best]
        return i + 1, next_u, norms[best]

    iterations, u, residual = jax.lax.while_loop(
        cond, body, (jnp.asarray(0), u0, r0)
    )
    finite = jnp.isfinite(residual) & jnp.all(jnp.isfinite(u))
    status = jnp.where(~finite, 2, jnp.where(residual <= tol, 0, 1))
    return GfeSolveResult(jax.nn.softmax(unpack(u)), status, iterations, residual)


def _gfe_solve(abar, h, logc, log_d):
    """Compatibility helper returning only the stationary simplex belief.

    Runtime code uses :func:`_gfe_solve_result` so budget exhaustion and
    divergence cannot be discarded.
    """
    return _gfe_solve_result(abar, h, logc, log_d).solution


def _gfe_rho(abar, h, logc, zbar):
    qx = abar @ zbar
    return abar.T @ (logc - _safe_log(qx)) - h


def _stats(a_param, learned):
    return _dirichlet_stats(a_param) if learned else _fixed_stats(a_param)


def gfe_parameter_stats(a_param, learned=False):
    """Batched ``E[A]`` and ``E[h(A)]`` for a fixed or Dirichlet matrix."""
    return jax.vmap(lambda value: _stats(value, learned))(a_param)


def gfe_stationary_stats(log_d, abar, hbar, logc):
    """Stationary state beliefs for a batch of GFE composites with supplied
    matrix moments.

    This is the common coordinate used by both the analytic Dirichlet path
    and the functional ``mu3`` path.  Keeping the latter in moment form is
    intentional: its weighted particle posterior is not Dirichlet.
    """

    def one(a_mean, h_mean, lc, ld):
        solved = _gfe_solve_result(a_mean, h_mean, lc, ld)
        return solved.solution, solved.residual, solved.status

    return jax.vmap(one)(abar, hbar, logc, log_d)


def gfe_update_stats(
    log_d, abar, hbar, logc, k_max, *, return_diagnostics=False
):
    """GFE update from arbitrary valid ``E[A]`` and ``E[h(A)]`` moments."""
    ko, ki = abar.shape[1], abar.shape[2]
    zbar, residual, status = gfe_stationary_stats(log_d, abar, hbar, logc)
    rho = jax.vmap(_gfe_rho)(abar, hbar, logc, zbar)
    qx = jnp.einsum("moi,mi->mo", abar, zbar)
    messages = (_norm_pad(rho, ki, k_max), _norm_pad(_safe_log(qx), ko, k_max))
    return (*messages, residual, status) if return_diagnostics else messages


def gfe_update(log_d, a_param, logc, k_max, learned=False, return_diagnostics=False):
    """Batched over members. log_d [M,K] normalized; a_param is logA (fixed)
    or Dirichlet counts α (learned); returns (μ2, log q_x) padded to k_max.

    μ2 is computed in the ρ form: the fixed point gives log z̄ = ρ(z̄) +
    log d − log Z, so μ2 = log z̄ − log d = ρ(z̄) − log Z, and _norm_pad
    absorbs the constant — identical to the subtraction form wherever
    d > 0, but bounded on zero-support states, where log z̄ − log d is a
    (−∞) − (−∞) that used to surface as ±1e30 junk and leak probability
    into downstream updates (cff-i45j)."""
    abar, hbar = gfe_parameter_stats(a_param, learned)
    return gfe_update_stats(
        log_d, abar, hbar, logc, k_max,
        return_diagnostics=return_diagnostics,
    )


def gfe_energy_stats(log_d, abar, hbar, logc):
    """Composite FE term from arbitrary ``E[A]``/``E[h(A)]`` moments."""

    def one(a_mean, h_mean, lc, ld):
        zbar = _gfe_solve(a_mean, h_mean, lc, ld)
        rho = _gfe_rho(a_mean, h_mean, lc, zbar)
        lz = _safe_log(zbar)
        return -(zbar * rho).sum(-1) + (zbar * lz).sum(-1)

    return jax.vmap(one)(abar, hbar, logc, log_d)


def gfe_goal_structural_stats(log_d, abar, hbar, expected_log_c):
    """GFE structural term with the goal cross-entropy removed.

    The complete learned-goal ledger owns ``-qx·E[log c]`` together with the
    Dirichlet prior/posterior terms so hostile sparse concentrations cancel
    algebraically before evaluation. Returns the remaining per-member term
    and each member's predictive observation mass.
    """

    def one(a_mean, h_mean, elog_c, ld):
        zbar = _gfe_solve(a_mean, h_mean, elog_c, ld)
        qx = a_mean @ zbar
        rho = _gfe_rho(a_mean, h_mean, elog_c, zbar)
        lz = _safe_log(zbar)
        full = -(zbar * rho).sum(-1) + (zbar * lz).sum(-1)
        return full + (qx * elog_c).sum(-1), qx

    return jax.vmap(one)(abar, hbar, expected_log_c, log_d)


def goal_dirichlet_energy(alpha, alpha0, predicted_counts):
    """Stable shared-goal likelihood/prior/parameter-edge ledger.

    This is ``KL(Dir(alpha)||Dir(alpha0)) - n·E[log c]`` combined as

        log B(alpha0) - log B(alpha)
        + (alpha - alpha0 - n)·E[log c].

    At the coordinate update ``alpha = alpha0 + n`` the potentially enormous
    expected-log terms cancel exactly instead of numerically.
    """
    elog = dirichlet_expected_log(alpha)

    def log_beta(value):
        return gammaln(value).sum() - gammaln(value.sum())

    proper = (
        jnp.all(jnp.isfinite(alpha))
        & jnp.all(alpha > 0)
        & jnp.all(jnp.isfinite(alpha0))
        & jnp.all(alpha0 > 0)
    )
    value = (
        log_beta(alpha0)
        - log_beta(alpha)
        + jnp.sum((alpha - alpha0 - predicted_counts) * elog)
    )
    return jnp.where(proper, value, jnp.inf)


def gfe_energy(log_d, a_param, logc, learned=False):
    """Composite FE term per member: −z*ᵀ rho(z*) − H[z*]."""
    abar, hbar = gfe_parameter_stats(a_param, learned)
    return gfe_energy_stats(log_d, abar, hbar, logc)


def mu3_log_message(particles, zbars, logcs, abar):
    """Evaluate ``Σ_t log μ₃,t(A_s)`` for fixed Dirichlet particles.

    ``abar`` is the current weighted posterior mean and is held fixed while
    evaluating the functional, exactly as in Part II Fig. 5.  Exact goal
    zeros remain true log-zeros; masked products avoid the undefined
    numerical form ``0 * -inf`` without inventing support.
    """
    dtype = jnp.result_type(particles, zbars, logcs, abar)
    particles = jnp.asarray(particles, dtype=dtype)
    zbars = jnp.asarray(zbars, dtype=dtype)
    logcs = jnp.asarray(logcs, dtype=dtype)
    abar = jnp.asarray(abar, dtype=dtype)
    logcs = jnp.where(logcs == jnp.asarray(NEG, dtype=dtype), -jnp.inf, logcs)
    predictive = jnp.einsum("oi,ti->to", abar, zbars)
    residual = logcs - _safe_log(predictive)

    sample_predictive = jnp.einsum("soi,ti->sto", particles, zbars)
    fit = jnp.sum(
        jnp.where(sample_predictive > 0, sample_predictive * residual[None, :, :], 0.0),
        axis=-1,
    )
    log_particles = jnp.where(particles > 0, jnp.log(particles), 0.0)
    particle_h = -jnp.sum(
        jnp.where(particles > 0, particles * log_particles, 0.0), axis=1
    )
    entropy = jnp.einsum("ti,si->st", zbars, particle_h)
    return jnp.sum(fit - entropy, axis=-1)


def mu3_importance_update(particles, zbars, logcs, abar):
    """Self-normalized IS coordinate update for the functional μ₃ message.

    Returns ``(Abar, hbar, weights, log_Z, E_phi, KL, ESS, status)`` where
    status is 0 on a valid estimate and 2 when weights collapse or contain a
    NaN/+inf.  Low ESS is a caller-configured inference-budget outcome.
    """
    logw = mu3_log_message(particles, zbars, logcs, abar)
    bad = jnp.any(jnp.isnan(logw) | jnp.isposinf(logw))
    finite_any = jnp.any(jnp.isfinite(logw))
    safe = jnp.where(jnp.isfinite(logw), logw, -jnp.inf)
    log_norm = logsumexp(safe)
    invalid = bad | ~finite_any | ~jnp.isfinite(log_norm)
    weights = jnp.where(invalid, jnp.zeros_like(safe), jnp.exp(safe - log_norm))

    new_abar = jnp.einsum("s,soi->oi", weights, particles)
    log_particles = jnp.where(particles > 0, jnp.log(particles), 0.0)
    particle_h = -jnp.sum(
        jnp.where(particles > 0, particles * log_particles, 0.0), axis=1
    )
    new_hbar = jnp.einsum("s,si->i", weights, particle_h)
    expected_phi = jnp.sum(jnp.where(weights > 0, weights * safe, 0.0))
    sample_count = jnp.asarray(particles.shape[0], dtype=particles.dtype)
    log_z = log_norm - jnp.log(sample_count)
    kl = expected_phi - log_z
    ess = jnp.where(
        invalid, jnp.asarray(0.0, dtype=particles.dtype),
        1.0 / jnp.sum(jnp.square(weights)),
    )
    status = jnp.where(invalid, 2, 0)
    return new_abar, new_hbar, weights, log_z, expected_phi, kl, ess, status


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
#   to out:      log q_w − μ_down                    (predictive cavity)
# because the out-edge belief multiplies this message by μ_down and must
# recover q_w. The node-local FE term is evaluated directly as
#   G_tm = −H[q_w] + E_q̄[ĥ] − H[q̄].
# Unlike the leaf A-composite, −q̄ᵀρ̄ − H[q̄] is NOT the local FE:
# ρ contains the interior edge's downstream Lagrange coupling, which is
# already represented by the downstream node and the edge-entropy correction.
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
    # Only exact zeros are omitted from 0*log(0); a fixed numeric threshold
    # would silently erase valid small probabilities in a dtype-dependent way.
    h = -jnp.sum(jnp.where(b > 0, b * log_b, 0.0), axis=1)  # [U, K]
    return b, h


def _tm_dirichlet_stats(counts, novelty_weight=None):
    """(B̄, ĥ) under per-(action, column) Dirichlets, counts [U, K', K]:
    B̄ = α/α₀ and E[B log B] = (α/α₀)(ψ(α+1) − ψ(α₀+1)) in closed form, so
    ĥ ≤ H[B̄-column] with the gap = the parameter-novelty channel (unexplored
    transition columns look attractive), mirroring _dirichlet_stats.

    ``novelty_weight`` optionally scales only that gap, column by column:
    ``h_λ = H[B̄] - λ(H[B̄] - ĥ)``.  The endpoints are therefore exact:
    λ=1 is the learned GFE-TM rule and λ=0 is the fixed-parameter GFE-TM
    objective at the same posterior mean.  It is not a probability floor or
    a temperature on B̄."""
    a0 = counts.sum(axis=1, keepdims=True)
    bbar = counts / a0
    e_blog = bbar * (
        jax.scipy.special.digamma(counts + 1.0) - jax.scipy.special.digamma(a0 + 1.0)
    )
    h = -e_blog.sum(axis=1)  # [U, K]
    if novelty_weight is not None:
        mean_h = -jnp.sum(
            jnp.where(bbar > 0, bbar * _safe_log(bbar), 0.0), axis=1
        )
        h = mean_h - novelty_weight * (mean_h - h)
    return bbar, h


def _tm_joint_stats(b_param, learned, novelty_weight=None):
    """Flatten [U, K', K] stats onto the joint column index j = z·U + u,
    matching (μ_in ⊕ μ_e).reshape(-1): abar [K', K·U], h [K·U]."""
    bbar, h = (
        _tm_dirichlet_stats(b_param, novelty_weight)
        if learned
        else _tm_fixed_stats(b_param)
    )
    kp, ku = bbar.shape[1], bbar.shape[0]
    kz = bbar.shape[2]
    abar = bbar.transpose(1, 2, 0).reshape(kp, kz * ku)
    return abar, h.T.reshape(kz * ku)


def gfe_tm_update(
    mu_in,
    mu_e,
    mu_down,
    b_param,
    k_max,
    learned=False,
    novelty_weight=None,
    return_diagnostics=False,
):
    """Batched over members. mu_in [M,K], mu_e [M,U], mu_down [M,K']
    normalized log-messages; b_param is log B (fixed) or Dirichlet counts α
    (learned), [M, U, K', K]. Returns (to_out, to_in, to_sel) padded.

    ``to_out`` is the cavity ``log(q_w) - mu_down``, not ``log(q_w)``:
    multiplying it by the facing message must reconstruct the substituted
    marginal q_w. Exact-zero q_w entries stay exact zero even when the facing
    message is also zero. Positive q_w against a zero-support facing message
    is infeasible and deliberately produces NaN so the runtime cannot present
    the state as coherent (status classification is cff-j4mu)."""
    ku, kp, kz = b_param.shape[1], b_param.shape[2], b_param.shape[3]

    weights = (
        novelty_weight
        if novelty_weight is not None
        else jnp.ones((b_param.shape[0], b_param.shape[1], b_param.shape[3]))
    )

    def one(bp, weight, mi, me, md):
        abar, h = _tm_joint_stats(
            bp, learned, weight if novelty_weight is not None else None
        )
        log_d = (mi[:, None] + me[None, :]).reshape(-1)
        solved = _gfe_solve_result(abar, h, md, log_d)
        qj = solved.solution
        rho = _gfe_rho(abar, h, md, qj).reshape(kz, ku)
        to_sel = logsumexp(rho + mi[:, None], axis=0)
        to_in = logsumexp(rho + me[None, :], axis=1)
        qw = abar @ qj
        qw_support = qw > 0
        # exp(md)==0 is the dtype-aware definition of a log-message's exact
        # zero support. It covers the engine's NEG sentinel without inventing
        # a second absolute cutoff. Subtract only on common support: 0/0 is a
        # cavity zero, while positive/0 is a genuinely infeasible edge belief.
        md_support = jnp.exp(md) > 0
        feasible = jnp.all(~qw_support | md_support)
        to_out = jnp.where(qw_support & md_support, jnp.log(qw) - md, NEG)
        to_out = jnp.where(feasible, to_out, jnp.nan)
        return to_out, to_in, to_sel, solved.residual, solved.status

    to_out, to_in, to_sel, residual, status = jax.vmap(one)(
        b_param, weights, mu_in, mu_e, mu_down
    )
    messages = (
        _norm_pad(to_out, kp, k_max),
        _norm_pad(to_in, kz, k_max),
        _norm_pad(to_sel, ku, k_max),
    )
    return (*messages, residual, status) if return_diagnostics else messages


def gfe_tm_energy(
    mu_in, mu_e, mu_down, b_param, learned=False, novelty_weight=None
):
    """Node-local GFE per member.

    ``mu_down`` influences the stationary q̄ but its linear coupling is not
    itself node energy. The direct local term is
    ``sum(q_w log q_w) + q̄·ĥ + sum(q̄ log q̄)``. This distinction matters
    for an interior substituted edge, whose downstream node and Bethe edge
    correction are scored separately (cff-hp3y)."""

    weights = (
        novelty_weight
        if novelty_weight is not None
        else jnp.ones((b_param.shape[0], b_param.shape[1], b_param.shape[3]))
    )

    def one(bp, weight, mi, me, md):
        abar, h = _tm_joint_stats(
            bp, learned, weight if novelty_weight is not None else None
        )
        log_d = (mi[:, None] + me[None, :]).reshape(-1)
        qj = _gfe_solve(abar, h, md, log_d)
        qw = abar @ qj
        qw_log_qw = jnp.where(qw > 0, qw * _safe_log(qw), 0.0).sum(-1)
        qj_log_qj = jnp.where(qj > 0, qj * _safe_log(qj), 0.0).sum(-1)
        return qw_log_qw + qj @ h + qj_log_qj

    return jax.vmap(one)(b_param, weights, mu_in, mu_e, mu_down)

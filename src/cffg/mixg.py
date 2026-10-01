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
([M, K, ...]); dims and K are static per group. Non-normalizable components
fall back to a *vacuous* message mid-flooding. Numerical collapse failures
after component admission return NaN messages for runtime failure attribution;
safe-inverse placeholders inside collapse never become accepted information.
Ledger terms go +inf on
improper beliefs, while public belief/predictive reads return shape-preserving
NaNs with ``proper=False``. Guards are scale-relative per the robustness
contract: proper systems are solved without regularization, while singular
message systems use the same dtype-relative Moore-Penrose limit as the plain
Gaussian rules.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from cffg import gauss
from cffg.compiled import NEG
from cffg.gaussian_components import backward_component, forward_component

_LOG2PI = 1.8378770664093453


class LinearMixtureStats(NamedTuple):
    """D3 factor-belief responsibilities and weighted sufficient statistics."""

    responsibility: jax.Array
    syy: jax.Array
    syx: jax.Array
    sxx: jax.Array
    proper: jax.Array


class MixtureStats(NamedTuple):
    """Prior-mixture factor-belief responsibilities and weighted moments."""

    responsibility: jax.Array
    weighted_mean: jax.Array
    weighted_second: jax.Array
    proper: jax.Array


class MixturePredictive(NamedTuple):
    """Component and collapsed moments of a factor-to-output cavity."""

    weight: jax.Array
    component_mean: jax.Array
    component_covariance: jax.Array
    mean: jax.Array
    covariance: jax.Array
    proper: jax.Array


class CollapseResult(NamedTuple):
    """Checked moment projection; invalid outputs are nonfinite, never a ridge."""

    eta: jax.Array
    precision: jax.Array
    valid: jax.Array


def _log_z(eta: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
    """log ∫ exp(−½yᵀΛy + ηᵀy) dy = ½ηᵀΛ⁻¹η + d/2·log2π − ½·logdet Λ.
    Caller guarantees Λ ≻ 0 (guarded upstream); branch-safe helpers keep an
    untaken invalid path finite without perturbing an accepted matrix."""
    d = lam.shape[-1]
    sol = gauss._pd_solve(lam, eta[..., None])[..., 0]
    logdet = gauss._pd_logdet(lam)
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
    logdet_w = gauss._pd_logdet(w)
    return -0.5 * jnp.einsum("ki,kij,kj->k", b, w, b) + 0.5 * logdet_w - 0.5 * b.shape[-1] * _LOG2PI


def _selector_support(log_weights):
    return ((log_weights != jnp.asarray(NEG, log_weights.dtype))
            & ~jnp.isneginf(log_weights))


def _active_components(log_weights, *parameters):
    """Keep zero-weight components out of unstable arithmetic and its derivatives.

    Inactive components borrow the first active component only for computation;
    their probability remains exactly zero. No substituted value is published.
    """
    support = _selector_support(log_weights)
    first = jnp.argmax(support)
    return support, tuple(
        jnp.where(support.reshape((-1,) + (1,) * (value.ndim - 1)), value, value[first])
        for value in parameters
    )


def _collapse(log_w: jnp.ndarray, eta: jnp.ndarray, lam: jnp.ndarray,
              *, support=None) -> CollapseResult:
    """Moment-match supported proper components and validate every inverse result.

    Strict component propriety cannot certify a later covariance calculation.
    Inactive components are sanitized before arithmetic; exact zero support
    must not multiply an overflowing moment by zero after the fact.
    """
    support = _selector_support(log_w) if support is None else support
    weights_valid = jnp.any(support) & jnp.all(jnp.where(support, jnp.isfinite(log_w), True))
    logits = jnp.where(support, log_w, -jnp.inf)
    w_s = jnp.where(support, jax.nn.softmax(jnp.where(weights_valid, logits, 0.0)), 0.0)
    safe_eta = jnp.where(support[:, None], eta, 0.0)
    safe_lam = jnp.where(support[:, None, None], lam, jnp.eye(lam.shape[-1], dtype=lam.dtype))
    cov = gauss._pd_inverse(safe_lam)
    m = jnp.einsum("kij,kj->ki", cov, safe_eta)
    m_bar = jnp.einsum("k,ki->i", w_s, m)
    spread = jnp.where(support[:, None], m - m_bar[None], 0.0)
    c_bar = jnp.einsum("k,kij->ij", w_s, cov) + jnp.einsum("k,ki,kj->ij", w_s, spread, spread)
    lam_out = gauss._pd_inverse(c_bar)
    eta_out = lam_out @ m_bar
    valid = (
        weights_valid
        & jnp.all(jnp.isfinite(w_s))
        & jnp.all(gauss._pd_ok(safe_lam))
        & jnp.all(jnp.isfinite(m))
        & jnp.all(gauss._pd_ok(cov))
        & gauss._pd_ok(c_bar)
        & gauss._pd_ok(lam_out)
        & jnp.all(jnp.isfinite(eta_out))
    )
    return CollapseResult(_nan_unless(valid, eta_out), _nan_unless(valid, lam_out), valid)


def _exact_probability(log_message: jax.Array, log_score: jax.Array):
    """Normalize supported selector states without reviving exact zeros."""
    support = log_message != jnp.asarray(NEG, log_message.dtype)
    valid = (
        jnp.any(support)
        & jnp.all(jnp.isfinite(log_message))
        & jnp.all(jnp.where(support, jnp.isfinite(log_score), True))
    )
    supported_score = jnp.where(support, log_score, -jnp.inf)
    normalizer = jnp.where(
        valid,
        jax.scipy.special.logsumexp(supported_score),
        jnp.asarray(0.0, log_score.dtype),
    )
    probability = jnp.where(support, jnp.exp(supported_score - normalizer), 0.0)
    return jnp.where(valid, probability, 0.0), valid


def _collapsed_moments(weight, component_mean, component_covariance):
    mean = jnp.einsum("k,ki->i", weight, component_mean)
    centered = component_mean - mean[None]
    covariance = jnp.einsum("k,kij->ij", weight, component_covariance)
    covariance += jnp.einsum("k,ki,kj->ij", weight, centered, centered)
    return mean, covariance


def _nan_unless(proper: jax.Array, value: jax.Array) -> jax.Array:
    return jnp.where(proper, value, jnp.full_like(value, jnp.nan))


# --- gaussian_mixture_linear ---------------------------------------------------


def mixlin_stats(eta_o, lam_o, eta_i, lam_i, log_mu_e, a, b, w):
    """Read D3 factor-belief moments for ``gaussian_mixture_linear``.

    Sufficient statistics are weighted by the soft regime responsibility and
    use an augmented input ``[x, 1]``. An invalid local belief is represented
    by shape-preserving NaNs and ``proper=False``.
    """

    def one(eo, lo, ei, li, le, A, bb, W):
        w_ok = gauss._pd_ok(W)
        safe_w = jnp.where(
            w_ok[:, None, None],
            W,
            jnp.eye(W.shape[-1], dtype=W.dtype)[None],
        )
        eta_j, lam_j = _joint_blocks(eo, lo, ei, li, A, bb, safe_w)
        joint_ok = gauss._pd_ok(lam_j)
        safe_lam = jnp.where(
            joint_ok[:, None, None],
            lam_j,
            jnp.eye(lam_j.shape[-1], dtype=lam_j.dtype)[None],
        )
        log_score = le + _log_z(eta_j, safe_lam) + _node_const(bb, safe_w)
        responsibility, selector_ok = _exact_probability(le, log_score)
        proper = selector_ok & jnp.all(w_ok) & jnp.all(joint_ok)

        covariance = gauss._pd_inverse(safe_lam)
        mean = jnp.einsum("kij,kj->ki", covariance, eta_j)
        out_dim = bb.shape[-1]
        out_mean = mean[:, :out_dim]
        in_mean = mean[:, out_dim:]
        syy = covariance[:, :out_dim, :out_dim]
        syy += jnp.einsum("ki,kj->kij", out_mean, out_mean)
        syx = covariance[:, :out_dim, out_dim:]
        syx += jnp.einsum("ki,kj->kij", out_mean, in_mean)
        syx = jnp.concatenate([syx, out_mean[..., None]], axis=-1)
        sxx = covariance[:, out_dim:, out_dim:]
        sxx += jnp.einsum("ki,kj->kij", in_mean, in_mean)
        augmented_dim = in_mean.shape[-1] + 1
        augmented = jnp.zeros(
            (in_mean.shape[0], augmented_dim, augmented_dim), dtype=mean.dtype
        )
        augmented = augmented.at[:, :-1, :-1].set(sxx)
        augmented = augmented.at[:, :-1, -1].set(in_mean)
        augmented = augmented.at[:, -1, :-1].set(in_mean)
        augmented = augmented.at[:, -1, -1].set(1.0)

        syy *= responsibility[:, None, None]
        syx *= responsibility[:, None, None]
        augmented *= responsibility[:, None, None]
        finite = (
            jnp.all(jnp.isfinite(responsibility))
            & jnp.all(jnp.isfinite(syy))
            & jnp.all(jnp.isfinite(syx))
            & jnp.all(jnp.isfinite(augmented))
        )
        proper &= finite
        return LinearMixtureStats(
            _nan_unless(proper, responsibility),
            _nan_unless(proper, syy),
            _nan_unless(proper, syx),
            _nan_unless(proper, augmented),
            proper,
        )

    return jax.vmap(one)(eta_o, lam_o, eta_i, lam_i, log_mu_e, a, b, w)


def mixlin_predictive(eta_i, lam_i, log_mu_e, a, b, w):
    """Read the factor-to-output cavity predictive in moment form."""

    def one(ei, li, le, A, bb, W):
        input_ok = gauss._pd_ok(li)
        w_ok = gauss._pd_ok(W)
        safe_input_precision = jnp.where(
            input_ok, li, jnp.eye(li.shape[-1], dtype=li.dtype)
        )
        safe_w = jnp.where(
            w_ok[:, None, None],
            W,
            jnp.eye(W.shape[-1], dtype=W.dtype)[None],
        )
        input_covariance = gauss._pd_inverse(safe_input_precision)
        input_mean = gauss._pd_solve(safe_input_precision, ei)
        component_mean = jnp.einsum("kij,j->ki", A, input_mean) + bb
        component_covariance = gauss._pd_inverse(safe_w)
        component_covariance += jnp.einsum(
            "kij,jl,kml->kim", A, input_covariance, A
        )
        weight, selector_ok = _exact_probability(le, le)
        mean, covariance = _collapsed_moments(
            weight, component_mean, component_covariance
        )
        proper = (
            input_ok
            & selector_ok
            & jnp.all(w_ok)
            & jnp.all(jnp.isfinite(component_mean))
            & jnp.all(jnp.isfinite(component_covariance))
            & jnp.all(jnp.isfinite(covariance))
            & jnp.all(gauss._pd_ok(component_covariance))
            & gauss._pd_ok(covariance)
        )
        return MixturePredictive(
            _nan_unless(proper, weight),
            _nan_unless(proper, component_mean),
            _nan_unless(proper, component_covariance),
            _nan_unless(proper, mean),
            _nan_unless(proper, covariance),
            proper,
        )

    return jax.vmap(one)(eta_i, lam_i, log_mu_e, a, b, w)


def mixlin_to_selector(eta_o, lam_o, eta_i, lam_i, a, b, w):
    """→ selector, exact: log μ(s) = log Z(η_J, Λ_J) + node const (D3).
    Any improper joint ⇒ uniform (zeros: log-messages are shift-free)."""

    def one(eo, lo, ei, li, A, bb, W):
        eta_j, lam_j = _joint_blocks(eo, lo, ei, li, A, bb, W)
        ok = gauss._pd_ok(W).all() & gauss._pd_ok(lam_j).all()
        logmu = _log_z(eta_j, lam_j) + _node_const(bb, W)
        return jnp.where(ok, logmu - logmu.max(), jnp.zeros_like(logmu))

    return jax.vmap(one)(eta_o, lam_o, eta_i, lam_i, a, b, w)


def mixlin_forward(eta_i, lam_i, log_mu_e, a, b, w):
    """→ out: collapse of per-regime forward predictives; weights are the
    incoming selector message alone (component masses are s-independent)."""

    def one(ei, li, le, A, bb, W):
        support, (A, bb, W) = _active_components(le, A, bb, W)
        components = jax.vmap(forward_component, in_axes=(None, None, 0, 0, 0))(
            ei, li, A, bb, W,
        )
        collapsed = _collapse(le, components.eta, components.precision, support=support)
        return _component_message(components, collapsed)

    return jax.vmap(one)(eta_i, lam_i, log_mu_e, a, b, w)


def mixlin_backward(eta_o, lam_o, log_mu_e, a, b, w):
    """→ in: collapse of per-regime backward messages with the D3 weights
    w̃_s ∝ μ_e(s) · c̃_s · Z(η_b, Λ_b) — the scale factors the scale-free
    message convention drops must be rebuilt here (the weight trap)."""

    def one(eo, lo, le, A, bb, W):
        support, (A, bb, W) = _active_components(le, A, bb, W)
        def bwd(Ak, bk, Wk):
            component = backward_component(eo, lo, Ak, bk, Wk)
            # log c̃: mass of ∫ N(y; b_k, V_k) μ_o(y) dy (D3)
            log_c = _log_z(Wk @ bk + eo, Wk + lo) - 0.5 * bk @ Wk @ bk
            logdet_w = gauss._pd_logdet(Wk)
            log_c = log_c + 0.5 * logdet_w - 0.5 * bk.shape[-1] * _LOG2PI
            return component, log_c

        components, log_c = jax.vmap(bwd)(A, bb, W)
        ok = jnp.all(components.valid & components.normalizable)
        log_w = le + log_c + jnp.where(
            ok, _log_z(components.eta, components.precision), 0.0,
        )
        collapsed = _collapse(log_w, components.eta, components.precision, support=support)
        return _component_message(components, collapsed)

    return jax.vmap(one)(eta_o, lam_o, log_mu_e, a, b, w)


def _component_message(components, collapsed):
    normalizable = jnp.all(components.normalizable)
    valid = jnp.all(components.valid)
    return (
        _nan_unless(valid, jnp.where(normalizable, collapsed.eta, 0.0)),
        _nan_unless(valid, jnp.where(normalizable, collapsed.precision, 0.0)),
    )


def mixlin_energy(eta_o, lam_o, eta_i, lam_i, log_mu_e, a, b, w):
    """Node energy, exact for the held beliefs (D3): mixture belief over
    (y, x, s) with regime weights ρ_s ∝ μ_e(s)·Z_J(s); joint entropy by the
    chain rule; per-regime E[log f] via the linear_energy residual algebra.
    Improper ⇒ +inf (loud)."""

    def one(eo, lo, ei, li, le, A, bb, W):
        eta_j, lam_j = _joint_blocks(eo, lo, ei, li, A, bb, W)
        ok = gauss._pd_ok(W).all() & gauss._pd_ok(lam_j).all()
        d = lam_j.shape[-1]
        log_rho = le + _log_z(eta_j, lam_j) + _node_const(bb, W)
        rho = jax.nn.softmax(log_rho)
        log_rho_n = jax.nn.log_softmax(log_rho)
        h_s = -(rho * log_rho_n).sum()
        logdet_j = gauss._pd_logdet(lam_j)
        h_gauss = 0.5 * d * (1.0 + _LOG2PI) - 0.5 * logdet_j  # [K]

        sig_j = gauss._pd_inverse(lam_j)
        m_j = jnp.einsum("kij,kj->ki", sig_j, eta_j)
        do = bb.shape[-1]
        jm = jnp.concatenate([jnp.eye(do, dtype=A.dtype)[None].repeat(A.shape[0], 0), -A], axis=-1)
        r = jnp.einsum("kij,kj->ki", jm, m_j) - bb  # [K, do]
        sig_r = jnp.einsum("kij,kjl,kml->kim", jm, sig_j, jm)
        logdet_w = gauss._pd_logdet(W)
        e_logf = (
            0.5 * logdet_w
            - 0.5 * do * _LOG2PI
            - 0.5 * (jnp.einsum("ki,kij,kj->k", r, W, r) + jnp.einsum("kij,kji->k", W, sig_r))
        )
        f = -h_s - (rho * h_gauss).sum() - (rho * e_logf).sum()
        return jnp.where(ok, f, jnp.inf)

    return jax.vmap(one)(eta_o, lam_o, eta_i, lam_i, log_mu_e, a, b, w)


# --- gaussian_mixture (prior variant, f_s(y) = N(y; m_s, V_s)) -------------------


def mix_stats(eta_o, lam_o, log_mu_e, mean, w):
    """Read prior-mixture factor-belief responsibilities and moments."""

    def one(eo, lo, le, component_mean, W):
        w_ok = gauss._pd_ok(W)
        safe_w = jnp.where(
            w_ok[:, None, None],
            W,
            jnp.eye(W.shape[-1], dtype=W.dtype)[None],
        )
        eta_q = jnp.einsum("kij,kj->ki", safe_w, component_mean) + eo[None]
        lam_q = safe_w + lo[None]
        belief_ok = gauss._pd_ok(lam_q)
        safe_lam = jnp.where(
            belief_ok[:, None, None],
            lam_q,
            jnp.eye(lam_q.shape[-1], dtype=lam_q.dtype)[None],
        )
        log_score = (
            le
            + _log_z(eta_q, safe_lam)
            - 0.5
            * jnp.einsum(
                "ki,kij,kj->k", component_mean, safe_w, component_mean
            )
            + 0.5 * gauss._pd_logdet(safe_w)
            - 0.5 * component_mean.shape[-1] * _LOG2PI
        )
        responsibility, selector_ok = _exact_probability(le, log_score)
        covariance = gauss._pd_inverse(safe_lam)
        belief_mean = jnp.einsum("kij,kj->ki", covariance, eta_q)
        second = covariance + jnp.einsum(
            "ki,kj->kij", belief_mean, belief_mean
        )
        weighted_mean = responsibility[:, None] * belief_mean
        weighted_second = responsibility[:, None, None] * second
        proper = (
            selector_ok
            & jnp.all(w_ok)
            & jnp.all(belief_ok)
            & jnp.all(jnp.isfinite(weighted_mean))
            & jnp.all(jnp.isfinite(weighted_second))
        )
        return MixtureStats(
            _nan_unless(proper, responsibility),
            _nan_unless(proper, weighted_mean),
            _nan_unless(proper, weighted_second),
            proper,
        )

    return jax.vmap(one)(eta_o, lam_o, log_mu_e, mean, w)


def mix_predictive(log_mu_e, mean, w):
    """Read a prior mixture's factor-to-output cavity predictive."""

    def one(le, component_mean, W):
        w_ok = gauss._pd_ok(W)
        safe_w = jnp.where(
            w_ok[:, None, None],
            W,
            jnp.eye(W.shape[-1], dtype=W.dtype)[None],
        )
        component_covariance = gauss._pd_inverse(safe_w)
        weight, selector_ok = _exact_probability(le, le)
        collapsed_mean, covariance = _collapsed_moments(
            weight, component_mean, component_covariance
        )
        proper = (
            selector_ok
            & jnp.all(w_ok)
            & jnp.all(jnp.isfinite(component_mean))
            & jnp.all(jnp.isfinite(component_covariance))
            & jnp.all(jnp.isfinite(covariance))
            & jnp.all(gauss._pd_ok(component_covariance))
            & gauss._pd_ok(covariance)
        )
        return MixturePredictive(
            _nan_unless(proper, weight),
            _nan_unless(proper, component_mean),
            _nan_unless(proper, component_covariance),
            _nan_unless(proper, collapsed_mean),
            _nan_unless(proper, covariance),
            proper,
        )

    return jax.vmap(one)(log_mu_e, mean, w)


def mix_to_selector(eta_o, lam_o, mean, w):
    """→ selector, exact and always proper: W_s + Λ_o ≻ 0."""

    def one(eo, lo, m, W):
        wm = jnp.einsum("kij,kj->ki", W, m)
        logmu = (
            _log_z(wm + eo[None], W + lo[None])
            - 0.5 * jnp.einsum("ki,kij,kj->k", m, W, m)
            + 0.5 * gauss._pd_logdet(W)
            - 0.5 * m.shape[-1] * _LOG2PI
        )
        ok = gauss._pd_ok(W).all() & gauss._pd_ok(W + lo[None]).all()
        return jnp.where(ok, logmu - logmu.max(), jnp.zeros_like(logmu))

    return jax.vmap(one)(eta_o, lam_o, mean, w)


def mix_forward(log_mu_e, mean, w):
    """→ out: collapse of the component priors N(m_s, V_s) with selector
    weights (masses are all 1 — no trap in this direction)."""

    def one(le, m, W):
        support, (m, W) = _active_components(le, m, W)
        collapsed = _collapse(le, jnp.einsum("kij,kj->ki", W, m), W, support=support)
        ok = gauss._pd_ok(W).all()
        return jnp.where(ok, collapsed.eta, 0.0), jnp.where(ok, collapsed.precision, 0.0)

    return jax.vmap(one)(log_mu_e, mean, w)


def mix_energy(eta_o, lam_o, log_mu_e, mean, w):
    """Node energy of the prior mixture: same chain-rule structure as
    mixlin_energy with the joint being just y."""

    def one(eo, lo, le, m, W):
        eta_q = jnp.einsum("kij,kj->ki", W, m) + eo[None]
        lam_q = W + lo[None]
        d = lam_q.shape[-1]
        log_rho = (
            le + _log_z(eta_q, lam_q)
            - 0.5 * jnp.einsum("ki,kij,kj->k", m, W, m)
            + 0.5 * gauss._pd_logdet(W)
            - 0.5 * d * _LOG2PI
        )
        rho = jax.nn.softmax(log_rho)
        h_s = -(rho * jax.nn.log_softmax(log_rho)).sum()
        logdet_q = gauss._pd_logdet(lam_q)
        h_gauss = 0.5 * d * (1.0 + _LOG2PI) - 0.5 * logdet_q
        sig_q = gauss._pd_inverse(lam_q)
        m_q = jnp.einsum("kij,kj->ki", sig_q, eta_q)
        r = m_q - m
        logdet_w = gauss._pd_logdet(W)
        e_logf = (
            0.5 * logdet_w
            - 0.5 * d * _LOG2PI
            - 0.5 * (jnp.einsum("ki,kij,kj->k", r, W, r) + jnp.einsum("kij,kji->k", W, sig_q))
        )
        ok = gauss._pd_ok(W).all() & gauss._pd_ok(lam_q).all()
        f = -h_s - (rho * h_gauss).sum() - (rho * e_logf).sum()
        return jnp.where(ok, f, jnp.inf)

    return jax.vmap(one)(eta_o, lam_o, log_mu_e, mean, w)

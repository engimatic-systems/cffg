"""Gaussian message-update and node-energy rules (cff-4e5z).

Messages on Gaussian edges live in information (canonical) form,

    mu(x)  ∝  exp(−½ xᵀ Λ x + ηᵀ x),           parameters (η [d], Λ [d,d]),

because the two operations message passing needs are trivial there: products
of messages add their (η, Λ), and the *vacuous* message (a half-edge, an
unbonded glue port, an uninitialized slot) is exactly (0, 0) — no infinities,
no special cases. Λ may be singular (an improper, partially-informative
message); that is a feature, and no rule requires an inverse of an arbitrary
incoming Λ. Affine rules factor the incoming likelihood in unit-scaled coordinates.
The outgoing precision is a Gram matrix, including for partial observations.

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
    still forming (vacuous Λ_i, rank-deficient A), so the implementation
    eliminates the input in a balanced joint least-squares system using
    the left null space of its design. For invertible square A a change of
    variables z=Ax reduces prediction to the same convolution as below.

BACKWARD (marginalize y against mu_out = (η_o, Λ_o)):
    mu(x) = ∫ f(y,x) mu_out(y) dy.  Quadratic in y with curvature W + Λ_o
    (always PD) and linear term W(Ax+b) + η_o; integrating leaves
        G     = W − W (W + Λ_o)⁻¹ W          (= (V + Λ_o⁻¹)⁻¹ when Λ_o ≻ 0)
        Λ_in  = Aᵀ G A
        η_in  = Aᵀ ( W (W + Λ_o)⁻¹ η_o − G b )
    A likelihood with square root B has G = Bᵀ(I+BVBᵀ)⁻¹B. A stacked
    QR gives the square root of I+BVBᵀ without forming that sum.
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
    (mask 0) gives a vacuous message to in and node energy −H[q_in],
    cancelling the interior input-edge entropy in the Bethe ledger.

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
axis) via vmap; dims are static per group; the engine gathers shape-local
blocks before calling and scatters exact segments after. Energies
are only meaningful at (near-)fixed points with proper beliefs — improper
beliefs surface as +inf, deliberately loud.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from cffg import efrules
from cffg._cavities import sum_other_legs

_LOG2PI = efrules.LOG2PI
GAUSSIAN = efrules.GAUSSIAN
_matrix_scale = efrules._matrix_scale
_symmetry_ok = efrules._symmetry_ok
_pd_ok = efrules._pd_ok
_safe_pd = efrules._safe_pd
_pd_solve = efrules._pd_solve
_pd_inverse = efrules._pd_inverse
_pd_logdet = efrules._pd_logdet


def is_proper_precision(mat: jnp.ndarray) -> jnp.ndarray:
    """Public propriety predicate used by Gaussian engine readouts."""
    return GAUSSIAN.is_proper_precision(mat)


def _psd_ok(mat: jnp.ndarray) -> jnp.ndarray:
    """Scale-relative PSD predicate for possibly-vacuous messages."""
    sym = 0.5 * (mat + mat.swapaxes(-1, -2))
    eig = jnp.linalg.eigvalsh(sym)
    tol = 16.0 * mat.shape[-1] * jnp.finfo(mat.dtype).eps * _matrix_scale(mat)
    return _symmetry_ok(mat) & jnp.isfinite(eig).all(axis=-1) & (eig[..., 0] >= -tol)


def _psd_pinv(mat: jnp.ndarray) -> jnp.ndarray:
    """Moore-Penrose inverse with dtype/dimension-relative numerical rank."""
    sym = 0.5 * (mat + mat.swapaxes(-1, -2))
    rtol = 16.0 * mat.shape[-1] * jnp.finfo(mat.dtype).eps
    return jnp.linalg.pinv(sym, rtol=rtol, hermitian=True)


class _PSDProjection(NamedTuple):
    matrix: jax.Array
    eigenvalues: jax.Array
    eigenvectors: jax.Array
    valid: jax.Array

    @property
    def full_rank(self):
        return jnp.all(self.eigenvalues > 0, axis=-1)


def _project_psd_with_rank(mat: jnp.ndarray, reference: jnp.ndarray) -> _PSDProjection:
    """Remove only roundoff-scale Schur-complement eigenvalues.

    Invariant protected: algebraically vacuous messages are represented by
    exact zero even after subtractive cancellation.  The cutoff is relative
    to the parent precision, so changing units cannot create information.
    """
    sym = 0.5 * (mat + mat.swapaxes(-1, -2))
    eig, vec = jnp.linalg.eigh(sym)
    tol = 4.0 * mat.shape[-1] * jnp.finfo(mat.dtype).eps * _matrix_scale(reference)
    valid = jnp.all(jnp.isfinite(eig), axis=-1) & (eig[..., 0] >= -tol)
    eig = jnp.where(eig > tol[..., None], eig, 0.0)
    matrix = (vec * eig[..., None, :]) @ vec.swapaxes(-1, -2)
    return _PSDProjection(matrix, eig, vec, valid)


def _project_psd(mat: jnp.ndarray, reference: jnp.ndarray) -> jnp.ndarray:
    """Preserve the ordinary Gaussian matrix API; mixture admission also uses rank."""
    return _project_psd_with_rank(mat, reference).matrix


def _likelihood_axes(eta, precision):
    """In x = diag(1/scale) U v the incoming precision is diagonal."""
    scale = jnp.sqrt(jnp.where(jnp.diag(precision) > 0, jnp.diag(precision), 1.))
    corr = (precision / scale[:, None]) / scale[None, :]
    values, axes = jnp.linalg.eigh(corr)
    roots = jnp.sqrt(jnp.maximum(values, 0.))
    scaled_eta = eta / scale
    natural = axes.T @ scaled_eta
    # A resolvable null-space linear tilt survives. Rounding the dot product
    # for a strong observation in its supported direction can mimic a tilt.
    error = len(eta) * jnp.finfo(eta.dtype).eps * (abs(axes.T) @ abs(scaled_eta))
    natural = jnp.where((roots == 0) & (abs(natural) <= error), 0., natural)
    return roots, axes, scale, natural


def _observed_affine(channel, target, flat, a, b, w):
    """Integrate those channels against N(y; Ax+b, W⁻¹), returning a Gram precision."""
    chol = jnp.linalg.cholesky(w)
    noise = jax.lax.linalg.triangular_solve(chol, channel.T, left_side=True, lower=True)
    # QR obtains a square root of I + B V Bᵀ without adding the identity
    # to a large squared singular value. No inverse of Λo is taken.
    r = jnp.linalg.qr(jnp.concatenate((noise, jnp.eye(len(w), dtype=w.dtype))), mode="r")
    shift = jax.lax.linalg.triangular_solve(chol, flat[:, None], left_side=True, lower=True)
    rhs = target - channel @ b - (noise.T @ shift)[:, 0]
    design = channel @ a
    whitened = jax.lax.linalg.triangular_solve(
        r, jnp.column_stack((design, rhs)), left_side=True, transpose_a=True,
    )
    design, rhs = whitened[:, :-1], whitened[:, -1]
    lam = design.T @ design
    eta = design.T @ rhs + a.T @ flat
    return eta, lam


def _neg_entropy(lam_q: jnp.ndarray) -> jnp.ndarray:
    """−H[q], or +inf when Λ is nonsymmetric/non-PD/non-finite."""
    return GAUSSIAN.negative_entropy_from_precision(lam_q)


# --- BP messages -------------------------------------------------------------


def prior_message(eta_p, w):
    """gaussian node → out: the constant message (η, Λ) = (Wm, W)."""
    return eta_p, w


def _valid_affine(e, l, a, b, w):
    return _pd_ok(w) & _psd_ok(l) & jnp.isfinite(a).all() & jnp.isfinite(b).all() & jnp.isfinite(e).all()


def _channel_data(e, l):
    roots, axes, scale, natural = _likelihood_axes(e, l)
    channel = (roots[:, None] * axes.T) * scale[None, :]
    flat = scale * (axes @ jnp.where(roots == 0, natural, 0.))
    target = jnp.where(roots > 0, natural / jnp.where(roots > 0, roots, 1.), 0.)
    return channel, target, flat


def _inverse_map_forward(e, l, a, b, w):
    channel, target, flat = _channel_data(e, l)
    transformed = jnp.linalg.solve(a.T, jnp.column_stack((channel.T, flat)))
    result = _observed_affine(transformed[:, :-1].T, target, transformed[:, -1],
                             jnp.eye(len(e), dtype=e.dtype), -b, w)
    return tuple(jnp.where(_valid_affine(e, l, a, b, w), out, 0.) for out in result)


@jax.custom_jvp
def linear_forward(eta_i, lam_i, a, b, w):
    """gaussian_linear → out. [M,·] batched; see module docstring."""

    def one(ei, li, A, bb, W):
        roots, axes, scale, natural = _likelihood_axes(ei, li)
        noise = jnp.linalg.cholesky(W).T
        model = noise @ ((A / scale) @ axes)
        # In the joint least-squares system the input design has orthogonal
        # likelihood rows. Balance its columns before eliminating the input.
        joint = jnp.vstack((model, jnp.diag(roots)))
        norms = jnp.max(abs(joint), axis=0)
        norms = jnp.where(norms > 0, norms, 1.)
        left, singular, right = jnp.linalg.svd(joint / norms, full_matrices=True)
        retained = singular > max(joint.shape) * jnp.finfo(ei.dtype).eps * singular[0]
        null = jnp.concatenate((~retained, jnp.ones(len(bb), dtype=bool)))
        # The rows left orthogonal to the input give an output square root.
        residual = (left[:len(bb)].T @ noise) * null[:, None]
        lam = residual.T @ residual
        reciprocal = jnp.where(retained, 1 / jnp.where(retained, singular, 1.), 0.)
        inverse = (right.T * reciprocal**2) @ right
        gain = (model / norms) @ inverse
        eta = lam @ bb + noise.T @ (gain @ (natural / norms))
        if A.shape[0] == A.shape[1]:
            sign, logdet = jnp.linalg.slogdet(A)
            vacuous = (li == 0).all() & (ei == 0).all() & (sign != 0) & jnp.isfinite(logdet)
            eta = jnp.where(vacuous, 0., eta)
            lam = jnp.where(vacuous, 0., lam)
        ok = _valid_affine(ei, li, A, bb, W)
        return jnp.where(ok, eta, 0.0), jnp.where(ok, lam, 0.0)

    args = (eta_i, lam_i, a, b, w)
    if a.shape[-1] == a.shape[-2]:
        sign, logdet = jnp.linalg.slogdet(a)
        invertible = jnp.all((sign != 0) & jnp.isfinite(logdet))
        return jax.lax.cond(invertible, lambda xs: jax.vmap(_inverse_map_forward)(*xs),
                            lambda xs: jax.vmap(one)(*xs), args)
    return jax.vmap(one)(*args)


@jax.custom_jvp
def linear_backward(eta_o, lam_o, a, b, w):
    """gaussian_linear → in; invalid incoming precisions yield vacuous."""

    def one(eo, lo, A, bb, W):
        # Factor Λo = BᵀB in unit-scaled coordinates. A singular likelihood
        # contributes ordinary scalar observations on its supported directions.
        channel, target, flat = _channel_data(eo, lo)
        eta, lam = _observed_affine(channel, target, flat, A, bb, W)
        exact_vacuous = (lo == 0).all() & (eo == 0).all()
        lam = jnp.where(exact_vacuous, jnp.zeros_like(lam), lam)
        eta = jnp.where(exact_vacuous, jnp.zeros_like(eta), eta)
        ok = _valid_affine(eo, lo, A, bb, W)
        return jnp.where(ok, eta, 0.0), jnp.where(ok, lam, 0.0)

    return jax.vmap(one)(eta_o, lam_o, a, b, w)


# The eigenvector, QR and SVD bases are nonunique (e.g. an isotropic
# likelihood). Differentiate the basis-independent Gaussian integral.
@linear_forward.defjvp
def _forward_jvp(primals, tangents):
    def canonical(e, l, a, b, w):
        cross = w @ a
        system = cross.T @ a + l
        # Apply the fixed-rank derivative in local units so independent
        # coordinates survive even when their raw precisions span many decades.
        scale = jnp.sqrt(jnp.where(jnp.diag(system) > 0, jnp.diag(system), 1.))
        normalized = (system / scale[:, None]) / scale[None, :]
        inverse = (_psd_pinv(normalized) / scale[:, None]) / scale[None, :]
        gain = cross @ inverse
        precision = w - gain @ cross.T
        return gain @ e + precision @ b, precision

    return linear_forward(*primals), jax.jvp(jax.vmap(canonical), primals, tangents)[1]


@linear_backward.defjvp
def _backward_jvp(primals, tangents):
    def canonical(e, l, a, b, w):
        gain = _pd_solve(w + l, w).T
        precision = gain @ l
        return a.T @ (gain @ e - precision @ b), a.T @ precision @ a

    return linear_backward(*primals), jax.jvp(jax.vmap(canonical), primals, tangents)[1]


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
    return sum_other_legs(eta), sum_other_legs(lam)


# --- Node energies F[q_a] ----------------------------------------------------


def prior_energy(eta_i, lam_i, mean, w):
    """gaussian node: q ∝ N(m,V)·mu_in with (η_q, Λ_q) = (Wm+η_i, W+Λ_i)."""

    def one(ei, li, m, W):
        eta_q, lam_q = GAUSSIAN.product((W @ m, W), (ei, li))
        m_q = _pd_solve(lam_q, eta_q)
        sig_q = _pd_inverse(lam_q)
        logdet_w = _pd_logdet(W)
        d = m - m_q
        e_logf = 0.5 * logdet_w - 0.5 * W.shape[0] * _LOG2PI - 0.5 * (
            d @ W @ d + jnp.trace(W @ sig_q)
        )
        val = _neg_entropy(lam_q) - e_logf
        ok = _pd_ok(W) & _pd_ok(lam_q) & jnp.isfinite(val)
        return jnp.where(ok, val, jnp.inf)

    return jax.vmap(one)(eta_i, lam_i, mean, w)


def linear_energy(eta_o, lam_o, eta_i, lam_i, a, b, w):
    """gaussian_linear (unobserved): joint belief over (out, in)."""

    def one(eo, lo, ei, li, A, bb, W):
        do, di = A.shape
        lam_j = jnp.block([[W + lo, -W @ A], [-A.T @ W, A.T @ W @ A + li]])
        eta_j = jnp.concatenate([W @ bb + eo, -A.T @ W @ bb + ei])
        sig_j = _pd_inverse(lam_j)
        m_j = _pd_solve(lam_j, eta_j)
        j = jnp.concatenate([jnp.eye(do, dtype=W.dtype), -A], axis=1)  # r = J[y;x] − b
        r = j @ m_j - bb
        sig_r = j @ sig_j @ j.T
        logdet_w = _pd_logdet(W)
        e_logf = 0.5 * logdet_w - 0.5 * do * _LOG2PI - 0.5 * (
            r @ W @ r + jnp.trace(W @ sig_r)
        )
        val = _neg_entropy(lam_j) - e_logf
        ok = _pd_ok(W) & _pd_ok(lam_j) & jnp.isfinite(val)
        return jnp.where(ok, val, jnp.inf)

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
        m_q = _pd_solve(lam_q, mk * (A.T @ (W @ (yy - bb))) + ei)
        sig_q = _pd_inverse(lam_q)
        r = yy - A @ m_q - bb
        logdet_w = _pd_logdet(W)
        e_logf = 0.5 * logdet_w - 0.5 * do * _LOG2PI - 0.5 * (
            r @ W @ r + jnp.trace(W @ (A @ sig_q @ A.T))
        )
        val = _neg_entropy(lam_q) - mk * e_logf
        ok = _pd_ok(W) & _pd_ok(lam_q) & jnp.isfinite(val)
        return jnp.where(ok, val, jnp.inf)

    return jax.vmap(one)(eta_i, lam_i, y, mask, a, b, w)


def equality_energy(eta, lam):
    """equality: −H[q] from the product of all legs (log f = 0 on support)."""
    natural_q = GAUSSIAN.product_axis((eta, lam), axis=1)
    return -GAUSSIAN.entropy(natural_q)


def edge_entropy(lam0, lam1):
    """+H[q_e] per interior edge (Bethe counting number d_e − 1 = 1)."""
    lam_q = GAUSSIAN.precision_product(lam0, lam1)
    val = GAUSSIAN.entropy_from_precision(lam_q)
    return jnp.where(_pd_ok(lam_q), val, jnp.inf)


# --- gaussian_prec VMP (cff-7c5o, DERIVATIONS.md D4) ---------------------------
#
# f(x; m, τ) = N(x; m, τ⁻¹): scalar Gaussian node whose precision lives on an
# ef:gamma edge — the EF abstraction's first mixed client. Mean-field
# q(x)q(τ); both messages read *marginals* (VMP convention), all closed form.
# Batched [M] without vmap (everything is scalar). Guards: improper marginals
# ⇒ vacuous message / +inf energy (engine-wide loudness convention).


def _gamma_ab(eta_tau):
    """(α, β) from a gamma-natural marginal [M, 2] (η = (α−1, −β))."""
    return eta_tau[..., 0] + 1.0, -eta_tau[..., 1]


def prec_to_x(eta_tau_q, m):
    """→ x edge: N(x; m, E[τ]⁻¹) in info form ((E[τ]·m, E[τ])); E[τ] = α/β.
    eta_tau_q [M, 2] is the τ-edge *marginal*; m [M, 1] the mean blob."""
    a, b = _gamma_ab(eta_tau_q)
    ok = (a > 0) & (b > 0)
    etau = jnp.where(ok, a / jnp.where(ok, b, 1.0), 0.0)  # vacuous when improper
    return etau[:, None] * m, etau[:, None, None] * jnp.ones_like(m)[..., None]


def _e_sq_dev(eta_x_q, lam_x_q, m):
    """E[(x−m)²] = (x̄−m)² + var from the x-edge marginal; (value, proper)."""
    lam = lam_x_q[..., 0, 0]
    ok = lam > 0
    safe = jnp.where(ok, lam, 1.0)
    xbar = eta_x_q[..., 0] / safe
    return (xbar - m[..., 0]) ** 2 + 1.0 / safe, ok


def prec_to_tau(eta_x_q, lam_x_q, m):
    """→ τ edge: gamma-natural increment (1/2, −E[(x−m)²]/2) [M, 2]."""
    s, ok = _e_sq_dev(eta_x_q, lam_x_q, m)
    msg = jnp.stack([0.5 * jnp.ones_like(s), -0.5 * s], axis=-1)
    return jnp.where(ok[:, None], msg, 0.0)


def prec_energy(eta_tau_q, eta_x_q, lam_x_q, m):
    """Node energy −E_q[log f] = −½(ψ(α) − ln β) + ½ ln 2π + ½ (α/β) E[(x−m)²].
    The ψ(α) is the digamma correction that makes the mean-field VFE exact."""
    from jax.scipy.special import digamma

    a, b = _gamma_ab(eta_tau_q)
    s, ok_x = _e_sq_dev(eta_x_q, lam_x_q, m)
    ok = ok_x & (a > 0) & (b > 0)
    safe_b = jnp.where(ok, b, 1.0)
    val = (
        -0.5 * (digamma(jnp.where(ok, a, 1.0)) - jnp.log(safe_b))
        + 0.5 * _LOG2PI
        + 0.5 * (jnp.where(ok, a, 1.0) / safe_b) * s
    )
    return jnp.where(ok, val, jnp.inf)


# --- Readout -----------------------------------------------------------------


def moments(eta, lam):
    """(mean, cov) from information form; improper beliefs are all-NaN.

    NaNs are an explicit readout status: they cannot be mistaken for the
    negative covariance previously returned for an indefinite precision.
    Use :func:`is_proper_precision` when a boolean status is needed.
    """
    return GAUSSIAN.moments((eta, lam))

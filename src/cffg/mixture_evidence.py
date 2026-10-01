"""Gaussian-mixture joint construction and component evidence."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from cffg import gauss

_LOG2PI = 1.8378770664093453


class RelativeEvidence(NamedTuple):
    score: jax.Array
    valid: jax.Array


def log_z(eta, precision):
    """Absolute Gaussian log partition; callers own propriety admission."""
    solution = gauss._pd_solve(precision, eta[..., None])[..., 0]
    return (.5 * (eta * solution).sum(-1) + .5 * precision.shape[-1] * _LOG2PI
            - .5 * gauss._pd_logdet(precision))


def joint_blocks(eta_o, precision_o, eta_i, precision_i, matrix, offset, noise_precision):
    """Per-component joint information over (out, in)."""
    weighted = jnp.einsum("kij,kjl->kil", noise_precision, matrix)
    top = jnp.concatenate([noise_precision + precision_o[None], -weighted], axis=-1)
    bottom = jnp.concatenate([
        -weighted.swapaxes(-1, -2),
        jnp.einsum("kji,kjl->kil", matrix, weighted) + precision_i[None],
    ], axis=-1)
    precision = jnp.concatenate([top, bottom], axis=-2)
    weighted_offset = jnp.einsum("kij,kj->ki", noise_precision, offset)
    eta = jnp.concatenate([
        weighted_offset + eta_o[None],
        -jnp.einsum("kji,kj->ki", matrix, weighted_offset) + eta_i[None],
    ], axis=-1)
    return eta, precision


def node_constant(offset, noise_precision):
    """Component-dependent constant in a normalized Gaussian transition."""
    return (-.5 * jnp.einsum("ki,kij,kj->k", offset, noise_precision, offset)
            + .5 * gauss._pd_logdet(noise_precision) - .5 * offset.shape[-1] * _LOG2PI)


def _anchor(eta, precision, means):
    """A bounded common reference changes only a discarded score constant."""
    eta, precision, means = jax.lax.stop_gradient((eta, precision, means))
    finite_precision = jnp.where(jnp.isfinite(precision), precision, 0.)
    mode = gauss._psd_pinv(finite_precision) @ eta
    lower, upper = means.min(axis=0), means.max(axis=0)
    mode = jnp.where(jnp.isnan(mode), lower / 2 + upper / 2, mode)
    return jnp.clip(mode, lower, upper)


def predictive_evidence(eta, precision, means, covariances):
    """Relative log integral of each normalized predictive times one potential.

    For C=L L.T, shared anchor a and r=eta-precision*a, solve
    G u=L^-1(m-a)+L.T r, G=I+L.T precision L, and let p=L u.
    Stationarity gives the mode objective without subtracting quadratic
    Gaussian normalizers. The omitted potential(a) is common to components.
    Symmetric indefinite potentials remain admissible when G is positive.
    """
    anchor = _anchor(eta, precision, means)
    residual = eta - precision @ anchor

    def one(mean, covariance):
        covariance_ok, safe_covariance = gauss._safe_pd(covariance)
        lower = jnp.linalg.cholesky(safe_covariance)
        system = jnp.eye(len(mean), dtype=mean.dtype) + lower.T @ precision @ lower
        delta = mean - anchor
        coordinates = solve_triangular(lower, delta, lower=True)
        solution = gauss._pd_solve(system, coordinates + lower.T @ residual)
        posterior = lower @ solution
        score = (.5 * residual @ posterior + .5 * residual @ delta
                 - .5 * delta @ precision @ posterior - .5 * gauss._pd_logdet(system))
        valid = (covariance_ok & gauss._pd_ok(system) & gauss._symmetry_ok(precision)
                 & jnp.all(jnp.isfinite(mean)) & jnp.all(jnp.isfinite(residual))
                 & jnp.all(jnp.isfinite(coordinates)) & jnp.all(jnp.isfinite(posterior))
                 & jnp.isfinite(score))
        return score, valid

    scores, valid = jax.vmap(one)(means, covariances)
    return RelativeEvidence(scores, jnp.all(valid))


def prior_evidence(eta, precision, means, noise_precision):
    covariance = gauss._pd_inverse(noise_precision)
    result = predictive_evidence(eta, precision, means, covariance)
    return RelativeEvidence(result.score, result.valid & jnp.all(gauss._pd_ok(noise_precision)))


def linear_evidence(eta_o, precision_o, eta_i, precision_i, matrix, offset, noise_precision):
    """Use predictive evidence for proper input, preserving the joint fallback.

    An improper input can still give a proper joint through output information.
    Its component masses include Jacobians that cannot be discarded. Inactive
    branches receive finite dummy inputs before products or decompositions.
    """
    proper_input = gauss._pd_ok(precision_i)
    input_identity = jnp.eye(len(eta_i), dtype=eta_i.dtype)
    output_identity = jnp.eye(len(eta_o), dtype=eta_o.dtype)

    def inputs(active):
        return (
            jnp.where(active, eta_o, 0.),
            jnp.where(active, precision_o, output_identity),
            jnp.where(active, eta_i, 0.),
            jnp.where(active, precision_i, input_identity),
            jnp.where(active, matrix, 0.),
            jnp.where(active, offset, 0.),
            jnp.where(active, noise_precision, output_identity[None]),
        )

    eo, lo, ei, li, a, b, w = inputs(proper_input)
    covariance_i = gauss._pd_inverse(li)
    mean_i = gauss._pd_solve(li, ei)
    means = jnp.einsum("kij,j->ki", a, mean_i) + b
    noise_covariance = gauss._pd_inverse(w)
    covariances = noise_covariance + a @ covariance_i @ a.swapaxes(-1, -2)
    stable = predictive_evidence(eo, lo, means, covariances)
    stable_valid = (stable.valid & gauss._pd_ok(covariance_i)
                    & jnp.all(jnp.isfinite(mean_i)) & jnp.all(gauss._pd_ok(w))
                    & jnp.all(gauss._pd_ok(noise_covariance)))

    eo, lo, ei, li, a, b, w = inputs(~proper_input)
    eta_j, precision_j = joint_blocks(eo, lo, ei, li, a, b, w)
    legacy_score = log_z(eta_j, precision_j) + node_constant(b, w)
    legacy_valid = (jnp.all(gauss._pd_ok(w)) & jnp.all(gauss._pd_ok(precision_j))
                    & jnp.all(jnp.isfinite(legacy_score)))
    return RelativeEvidence(jnp.where(proper_input, stable.score, legacy_score),
                            jnp.where(proper_input, stable_valid, legacy_valid))

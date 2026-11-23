"""Rank and arithmetic certificates for normalized linear-mixture components."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from cffg import gauss


class GaussianComponent(NamedTuple):
    eta: jax.Array
    precision: jax.Array
    normalizable: jax.Array
    valid: jax.Array


def _full_column_rank(generator, matrix, spectral_error):
    """Certify rank before rounding a Gram matrix, independently of coordinate units."""
    generator, matrix, spectral_error = jax.lax.stop_gradient((generator, matrix, spectral_error))
    factor = generator @ matrix
    product_scale = jnp.abs(generator) @ jnp.abs(matrix)
    finite = jnp.all(jnp.isfinite(factor)) & jnp.all(jnp.isfinite(product_scale))
    if factor.shape[0] < factor.shape[1]:
        return jnp.asarray(False), finite
    # Equilibration must not amplify cancellation in a nullspace projection.
    error = 16 * matrix.shape[0] * jnp.finfo(factor.dtype).eps * product_scale + spectral_error
    factor = jnp.where(jnp.abs(factor) > error, factor, 0.0)
    scaled = jnp.where(jnp.isfinite(factor), factor, 0.0)
    for axis in (0, 1):
        scale = jnp.max(jnp.abs(scaled), axis=axis, keepdims=True)
        # A zero row/column stays zero; the dummy divisor adds no information.
        scaled = scaled / jnp.where(scale > 0, scale, 1.0)
    singular = jnp.linalg.svd(scaled, compute_uv=False, full_matrices=False)
    # Only numerically independent directions may define a normalized density.
    tolerance = 16 * max(factor.shape) * jnp.finfo(factor.dtype).eps * singular[0]
    return finite & (singular[-1] > tolerance), finite


def _backward_rank(precision, matrix, proper):
    """For positive noise precision, null(G) = null(incoming precision).

    Use that unchanged subspace rather than eigenvectors of a subtractively
    computed G. Proper incoming information reduces the rank question to A.
    """
    diagonal = jnp.diag(precision)
    units = jnp.sqrt(jnp.where(diagonal > 0, diagonal, 1.0))
    equilibrated = precision / units[:, None] / units[None, :]
    # L = D C D: test C's informed subspace against D A, without overflow.
    column_scale = jnp.max(jnp.abs(matrix), axis=0)
    normalized = matrix / jnp.where(column_scale > 0, column_scale, 1.0)
    transformed = (units / jnp.max(units))[:, None] * normalized
    matrix = jnp.where(proper, matrix, transformed)
    projection = _schur_projection(equilibrated, equilibrated, proper)
    retained = projection.eigenvalues > 0
    basis = jnp.where(retained[:, None], projection.eigenvectors.T, 0.0)
    basis = jnp.where(proper, jnp.eye(matrix.shape[0], dtype=matrix.dtype), basis)
    # Eigenspace error scales with the separation from the discarded zero space.
    eigenvalue = jnp.where(retained, projection.eigenvalues, 1.0)
    uncertainty = 16 * matrix.shape[0] * jnp.finfo(matrix.dtype).eps
    uncertainty = uncertainty * gauss._matrix_scale(equilibrated) / eigenvalue
    column_scale = jnp.max(jnp.abs(matrix), axis=0)
    normalized = matrix / jnp.where(column_scale > 0, column_scale, 1.0)
    column_norm = jnp.linalg.norm(normalized, axis=0) * column_scale
    error = jnp.where(proper, 0.0, uncertainty[:, None] * column_norm[None, :])
    rank, valid = _full_column_rank(basis, matrix, error)
    return rank, valid & jnp.all(jnp.isfinite(error)) & jnp.where(proper, True, projection.valid)


def forward_component(eta_i, precision_i, matrix, offset, noise_precision):
    """Forward Schur rank remains authoritative after matrix reconstruction."""
    proper_input = gauss._pd_ok(precision_i)
    weighted = noise_precision @ matrix
    system = matrix.T @ weighted + precision_i
    inverse = gauss._psd_pinv(jnp.where(proper_input, _distinct_diagonal(system), system))
    projection = _schur_projection(
        noise_precision - weighted @ inverse @ weighted.T, noise_precision, proper_input,
    )
    eta_schur = projection.matrix @ offset + weighted @ (inverse @ eta_i)

    covariance_i = gauss._pd_inverse(precision_i)
    mean_i = gauss._pd_solve(precision_i, eta_i)
    covariance_noise = gauss._pd_inverse(noise_precision)
    covariance = covariance_noise + matrix @ covariance_i @ matrix.T
    precision = gauss._pd_inverse(covariance)
    eta = precision @ (offset + matrix @ mean_i)
    covariance_valid = (
        gauss._pd_ok(covariance_i) & gauss._pd_ok(covariance_noise)
        & gauss._pd_ok(covariance) & gauss._pd_ok(precision)
    )
    normalizable = proper_input | projection.full_rank
    precision = jnp.where(proper_input, precision, projection.matrix)
    eta = jnp.where(proper_input, eta, eta_schur)
    valid = (
        gauss._pd_ok(noise_precision) & gauss._psd_ok(precision_i)
        & jnp.all(jnp.isfinite(matrix)) & jnp.all(jnp.isfinite(offset))
        & jnp.all(jnp.isfinite(eta_i)) & jnp.all(jnp.isfinite(eta))
        & jnp.all(jnp.isfinite(precision))
        & jnp.where(proper_input, covariance_valid, projection.valid)
        & jnp.where(normalizable, gauss._pd_ok(precision), True)
    )
    return GaussianComponent(eta, precision, normalizable, valid)


def backward_component(eta_o, precision_o, matrix, offset, noise_precision):
    """Check the generating factor's input rank, not its rounded Gram matrix."""
    proper_output = gauss._pd_ok(precision_o)
    system = noise_precision + precision_o
    solved = gauss._pd_solve(system, noise_precision)
    projection = _schur_projection(
        noise_precision - noise_precision @ solved, noise_precision, proper_output,
    )
    eta_schur = matrix.T @ (solved.T @ eta_o - projection.matrix @ offset)

    covariance_o = gauss._pd_inverse(precision_o)
    mean_o = gauss._pd_solve(precision_o, eta_o)
    covariance_noise = gauss._pd_inverse(noise_precision)
    covariance = covariance_noise + covariance_o
    combined_precision = gauss._pd_inverse(covariance)
    covariance_valid = (
        gauss._pd_ok(covariance_o) & gauss._pd_ok(covariance_noise)
        & gauss._pd_ok(covariance) & gauss._pd_ok(combined_precision)
    )
    normalizable, rank_valid = _backward_rank(precision_o, matrix, proper_output)
    g = jnp.where(proper_output, combined_precision, projection.matrix)
    precision = matrix.T @ g @ matrix
    eta = jnp.where(
        proper_output, matrix.T @ (combined_precision @ (mean_o - offset)), eta_schur,
    )
    valid = (
        gauss._pd_ok(noise_precision) & gauss._psd_ok(precision_o)
        & gauss._pd_ok(system)
        & jnp.all(jnp.isfinite(matrix)) & jnp.all(jnp.isfinite(offset))
        & jnp.all(jnp.isfinite(eta_o)) & jnp.all(jnp.isfinite(eta))
        & jnp.all(jnp.isfinite(precision)) & rank_valid
        & jnp.where(proper_output, covariance_valid, projection.valid)
        & jnp.where(normalizable, gauss._pd_ok(precision), True)
    )
    return GaussianComponent(eta, precision, normalizable, valid)


def _distinct_diagonal(matrix):
    return jnp.diag(jnp.arange(1, matrix.shape[-1] + 1, dtype=matrix.dtype))


def _schur_projection(matrix, reference, unused):
    # An unselected repeated eigenspace otherwise poisons proper-branch VJPs.
    dummy = _distinct_diagonal(matrix)
    return gauss._project_psd_with_rank(
        jnp.where(unused, dummy, matrix), jnp.where(unused, dummy, reference),
    )

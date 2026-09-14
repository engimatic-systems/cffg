"""Compensated local covariance algebra for D12 moment projection.

A covariance is represented by a rounded matrix and a retained additive
correction. Neither word is a new stationarity coordinate. Matrix products
accumulate one outer product at a time, using only matrix-sized temporaries.
"""

import jax
import jax.numpy as jnp

from cffg.hybrid_affine import _add, _two_product, _two_sum, affine


def supported(value):
    """Normal finite words or exact zero; subnormal inputs are unsupported."""
    return jnp.all(jnp.isfinite(_add(value, jnp.zeros_like(value))))


def symmetric(matrix):
    """Share one computed triangle of a mathematically symmetric matrix."""
    indices = jnp.arange(matrix.shape[-1])
    return jnp.where(indices[:, None] <= indices[None, :], matrix, matrix.swapaxes(-1, -2))


@jax.custom_jvp
def product(left, right, *offsets):
    """Compensated matrix product plus separately supplied matrix offsets."""
    shape = jnp.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    shape += (left.shape[-2], right.shape[-1])
    initial = (jnp.zeros(shape, left.dtype),) * 2

    def accumulate(k, state):
        total, correction = state
        high, low = _two_product(left[..., :, k, None], right[..., None, k, :])
        total, remainder = _two_sum(total, high)
        return total, _add(correction, _add(remainder, low))

    total, correction = jax.lax.fori_loop(0, left.shape[-1], accumulate, initial)
    for offset in offsets:
        total, remainder = _two_sum(total, offset)
        correction = _add(correction, remainder)
    return _add(total, correction)


@product.defjvp
def _product_jvp(primals, tangents):
    left, right, *offsets = primals
    dl, dr, *do = tangents
    tangent = dl @ right + left @ dr
    for offset in do:
        tangent += offset
    return product(left, right, *offsets), tangent


def combine(base, increment, *corrections):
    """Round base+increment once and retain its signed rounding remainder."""
    high = affine(jnp.ones_like(base)[..., None], increment[..., None], base, *corrections)
    low = affine(jnp.ones_like(base)[..., None], increment[..., None], base, *corrections, -high)
    return high, low


def residual(base, correction, value, target):
    """target - (base+correction) @ value, without combining the words."""
    return product(-base, value, product(-correction, value), target)


def _refined_solve(base, correction, rhs):
    chol = jnp.linalg.cholesky(base)
    initial = jax.scipy.linalg.cho_solve((chol, True), rhs)
    remainder = residual(base, correction, initial, rhs)
    increment = jax.scipy.linalg.cho_solve((chol, True), remainder)
    value = affine(jnp.ones_like(initial)[..., None], increment[..., None], initial)
    final_residual = residual(base, correction, value, rhs)
    scale = abs(rhs) + abs(base) @ abs(value) + abs(correction) @ abs(value)
    # A relative backward error must not hide a normal RHS behind a unit floor.
    allowance = 32 * base.shape[-1] * jnp.finfo(base.dtype).eps * scale
    zero_rhs = jnp.all(rhs == 0, axis=-2, keepdims=True)
    exact_zero = jnp.all(value == 0, axis=-2, keepdims=True)
    resolved = jnp.where(zero_rhs, exact_zero, abs(final_residual) <= allowance)
    finite = jnp.isfinite(value) & jnp.isfinite(final_residual) & jnp.isfinite(scale)
    accepted = jnp.all(resolved & finite, axis=(-2, -1))
    return jnp.where(accepted[..., None, None], value, jnp.nan)


def solve(base, correction, rhs):
    """Solve the represented covariance with one compensated refinement.

    The correction is a rounding remainder, not a second large covariance.
    Cholesky supplies the solve; a matrix-sized residual restores its lost
    low bits. This is a fixed arithmetic refinement, not a convergence claim.
    A dtype/dimension-scaled backward-error certificate rejects unresolved
    solves, including a nonzero normal RHS whose solution flushes to zero.
    The linear-solve primitive supplies dX=C^-1(dR-dC X), including reverse
    and higher derivatives, without differentiating integer arithmetic.
    """
    return jax.lax.custom_linear_solve(
        lambda value: base @ value + correction @ value, rhs,
        solve=lambda _, value: _refined_solve(base, correction, value), symmetric=True)


@jax.custom_jvp
def inverse(base, correction):
    """Retain inverse rounding information before a strong-cavity quotient.

    Solve the compensated inverse residual for its low word. Both returned
    words represent the same inverse; the high word carries its mathematical
    differential and the derived remainder carries zero differential.
    """
    eye = jnp.broadcast_to(jnp.eye(base.shape[-1], dtype=base.dtype), base.shape)
    chol = jnp.linalg.cholesky(base)
    high = symmetric(jax.scipy.linalg.cho_solve((chol, True), eye))
    low = symmetric(jax.scipy.linalg.cho_solve(
        (chol, True), residual(base, correction, high, eye)))
    high, low = (symmetric(word) for word in combine(high, low))
    remainder = product(-base, high, product(-correction, high),
                        product(-base, low), product(-correction, low), eye)
    scale = abs(eye) + abs(base) @ abs(high) + abs(correction) @ abs(high)
    # A once-rounded inverse cannot certify the low bits used by a quotient.
    allowance = 64 * base.shape[-1] * jnp.finfo(base.dtype).eps ** 2 * scale
    resolved = jnp.all(jnp.isfinite(remainder) & jnp.isfinite(scale)
                       & (abs(remainder) <= allowance), axis=(-2, -1))
    return tuple(jnp.where(resolved[..., None, None], word, jnp.nan) for word in (high, low))


@inverse.defjvp
def _inverse_jvp(primals, tangents):
    base, correction = primals
    db, dc = tangents
    high, low = inverse(base, correction)
    derivative = -solve(base, correction, solve(base, correction, db + dc).swapaxes(-1, -2))
    return (high, low), (symmetric(derivative), jnp.zeros_like(low))


@jax.custom_jvp
def logdet(base, correction):
    """logdet(base+correction), preserving the small logdet correction."""
    chol = jnp.linalg.cholesky(base)
    whitened = jax.scipy.linalg.solve_triangular(chol, correction, lower=True)
    whitened = jax.scipy.linalg.solve_triangular(chol, whitened.swapaxes(-1, -2), lower=True)
    eigenvalues = jnp.linalg.eigvalsh((whitened + whitened.swapaxes(-1, -2)) / 2)
    return (2 * jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=-1)
            + jnp.log1p(eigenvalues).sum(axis=-1))


@logdet.defjvp
def _logdet_jvp(primals, tangents):
    base, correction = primals
    db, dc = tangents
    derivative = solve(base, correction, db + dc)
    return logdet(base, correction), jnp.trace(derivative, axis1=-2, axis2=-1)

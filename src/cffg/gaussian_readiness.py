"""Numerical input-admission metadata, never a substitute Gaussian belief."""

from enum import IntEnum

import jax
import jax.numpy as jnp

from cffg import gauss


class InputReadiness(IntEnum):
    READY = 0
    PARTIAL = 1
    INVALID = 2


def _has_subnormal(value):
    """Detect nonzero subnormals without floating arithmetic that can flush them."""
    info = jnp.finfo(value.dtype)
    unsigned = getattr(jnp, f"uint{info.bits}")
    bits = jax.lax.bitcast_convert_type(value, unsigned)
    magnitude = bits & jnp.asarray((1 << (info.bits - 1)) - 1, unsigned)
    return jnp.any((magnitude != 0) & (magnitude < (1 << info.nmant)))


def _information_direction(eta, units):
    """Normalize eta/units without overflowing or subtracting large logarithms."""
    mantissa, exponent = jnp.frexp(eta)
    unit_mantissa, unit_exponent = jnp.frexp(units)
    exponent = exponent - unit_exponent
    present = eta != 0
    largest = jnp.max(jnp.where(present, exponent, jnp.iinfo(exponent.dtype).min))
    largest = jnp.where(jnp.any(present), largest, 0)
    shift = jnp.where(present, exponent, largest) - largest
    return jnp.where(present, jnp.ldexp(mantissa / unit_mantissa, shift), 0.)


def _resolved_partial(matrix, information):
    """Numerical PSD/range compatibility with a bounded eigenspace uncertainty."""
    dimension, dtype = matrix.shape[0], matrix.dtype
    eps = jnp.finfo(dtype).eps
    symmetric = (matrix + matrix.T) / 2
    eigenvalues, basis = jnp.linalg.eigh(symmetric)
    scale = jnp.max(jnp.sum(jnp.abs(symmetric), axis=1))
    residual = symmetric @ basis - basis * eigenvalues[None, :]
    # Include eigensolve residual and congruence/symmetrization rounding.
    orthogonality = jnp.linalg.norm(basis.T @ basis - jnp.eye(dimension, dtype=dtype))
    envelope = jnp.linalg.norm(residual) + scale * (orthogonality + 8 * dimension * eps)
    retained = eigenvalues > envelope
    discarded = ~retained
    rank = jnp.sum(retained)
    positive = jnp.min(jnp.where(retained, eigenvalues, jnp.inf))
    null_upper = jnp.max(jnp.where(discarded, eigenvalues, -jnp.inf))
    gap = positive - null_upper
    safe_gap = jnp.where(gap > 2 * envelope, gap - 2 * envelope, 1.)
    uncertainty = envelope / safe_gap
    # Never turn a poorly separated eigenspace into a broad compatibility band.
    resolved = (gap > 2 * envelope) & (uncertainty <= jnp.sqrt(eps))
    coordinates = basis.T @ information
    product_scale = jnp.abs(basis.T) @ jnp.abs(information)
    # Protect null-dot rounding, transformed-eta division and basis uncertainty.
    tolerance = 16 * dimension * eps * product_scale
    tolerance += (uncertainty + 4 * eps) * jnp.linalg.norm(information)
    compatible = jnp.all(jnp.where(discarded, jnp.abs(coordinates) <= tolerance, True))
    return (gauss._symmetry_ok(matrix) & jnp.all(jnp.isfinite(eigenvalues))
            & (eigenvalues[0] >= -envelope) & (rank > 0) & (rank < dimension)
            & resolved & compatible)


def input_readiness(eta, precision):
    """Classify one information potential as READY, PARTIAL or INVALID.

    READY preserves the existing strict propriety predicate. PARTIAL means
    numerically consistent with flat uninformed directions and compatible
    linear information; it does not authorize function evaluation or moments.
    Unresolved fallback subspaces are INVALID. Tiny signed perturbations below
    the stated rounding envelope are not an exact-arithmetic PSD/range proof.
    Fallback subnormal entries are INVALID because backend arithmetic may
    flush them to zero. Exact signed zeros remain valid partial information.
    Noise, parameters and output potentials require separate factor validation.
    """
    eta, precision = jnp.asarray(eta), jnp.asarray(precision)
    if eta.ndim != 1 or not len(eta) or precision.shape != (len(eta), len(eta)):
        raise ValueError("input readiness requires a nonempty vector and matching square precision")
    if eta.dtype != precision.dtype or not jnp.issubdtype(eta.dtype, jnp.floating):
        raise TypeError("input readiness requires matching floating-point dtypes")
    eta, precision = jax.lax.stop_gradient((eta, precision))
    finite = jnp.all(jnp.isfinite(eta)) & jnp.all(jnp.isfinite(precision))
    ready = finite & gauss._pd_ok(precision)
    diagonal = jnp.diag(precision)
    missing = diagonal == 0
    structural = jnp.all(jnp.where(missing[:, None] | missing[None, :], precision == 0, True))
    structural &= jnp.all(jnp.where(missing, eta == 0, True))
    candidate = finite & gauss._symmetry_ok(precision) & jnp.all(diagonal >= 0) & structural
    # Fallback arithmetic cannot certify values the backend may flush to zero.
    candidate &= ~(_has_subnormal(eta) | _has_subnormal(precision))
    # Invalid/ready branches receive finite operands; these are not model data.
    working = jnp.where(candidate & ~ready, precision, jnp.eye(len(eta), dtype=eta.dtype))
    information = jnp.where(candidate & ~ready, eta, 0.)
    diagonal = jnp.diag(working)
    units = jnp.sqrt(jnp.where(diagonal > 0, diagonal, 1.))
    equilibrated = working / units[:, None] / units[None, :]
    direction = _information_direction(information, units)
    finite_scaled = jnp.all(jnp.isfinite(equilibrated)) & jnp.all(jnp.isfinite(direction))
    safe = jnp.where(finite_scaled, equilibrated, jnp.eye(len(eta), dtype=eta.dtype))
    # Exact zero coordinates need no spectral cutoff on the informed block.
    completion = safe + jnp.diag((diagonal == 0).astype(eta.dtype))
    coordinate_partial = jnp.any(diagonal == 0) & gauss._pd_ok(completion)
    partial = candidate & finite_scaled & (coordinate_partial | _resolved_partial(safe, direction))
    return jnp.where(ready, jnp.int32(InputReadiness.READY),
                      jnp.where(partial, jnp.int32(InputReadiness.PARTIAL), jnp.int32(InputReadiness.INVALID)))

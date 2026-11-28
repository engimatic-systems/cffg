"""Host-side joint draws for the Gaussian conjugate families."""

from operator import index

import numpy as np

from cffg import ef
from cffg._column_precision import scaled_root


def _sample_shape(shape):
    dims = shape if isinstance(shape, tuple) else (shape,)
    if any(isinstance(n, (bool, np.bool_)) for n in dims):
        raise TypeError("sample_shape requires integer dimensions")
    dims = tuple(index(n) for n in dims)
    if any(n < 0 for n in dims):
        raise ValueError("sample_shape dimensions must be nonnegative")
    return dims


def _scalar(name, value):
    value = np.asarray(value)
    if value.ndim or value.dtype.kind not in "fiu" or not np.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


def _scale_factor(name, value, dim):
    matrix = np.asarray(value)
    if matrix.shape != (dim, dim) or matrix.dtype not in (np.float32, np.float64):
        raise ValueError(f"{name} must be a {dim} by {dim} float32 or float64 matrix")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            corr, scale = ef.resolved_correlation(matrix)
            return np.linalg.cholesky(corr), scale
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"{name} must be finite and positive definite at stored precision") from exc


def sample_joint(key, sample_shape, center, psi, nu, *, column_precision=None, column_root=None, kappa=None):
    from jax import random

    shape = _sample_shape(sample_shape)
    center = np.asarray(center)
    regression = column_precision is not None or column_root is not None
    if center.ndim != (2 if regression else 1) or 0 in center.shape:
        raise ValueError("posterior location has the wrong shape")
    if center.dtype.kind not in "fiu" or not np.isfinite(center).all():
        raise ValueError("posterior location must be finite and real")
    d = center.shape[0]
    nu = _scalar("nu", nu)
    if nu <= d - 1:
        raise ValueError("nu must exceed output dimension minus one")
    row_chol, row_scale = _scale_factor("psi", psi, d)
    dtype = np.result_type(center.dtype, psi.dtype)
    if column_root is not None:
        root, col_scale = scaled_root(column_root, center.shape[1])
        col_chol = root.T
        dtype = np.result_type(dtype, column_root.dtype)
    elif regression:
        col_chol, col_scale = _scale_factor("lam", column_precision, center.shape[1])
        dtype = np.result_type(dtype, column_precision.dtype)
    else:
        kappa = _scalar("kappa", kappa)
        if kappa <= 0:
            raise ValueError("kappa must be positive")
    if dtype not in (np.float32, np.float64):
        raise ValueError("joint draws require a float32 or float64 result dtype")

    # The key seeds the same host generator whether JAX x64 is enabled or disabled.
    seed = np.asarray(random.bits(key, (4,), dtype=np.uint32))
    rng = np.random.Generator(np.random.PCG64(seed))
    # Reverse Bartlett factor (Axen, 2023). For L L' = psi, S = L inv(T)
    # is a lower Cholesky factor of Sigma ~ IW(psi, nu).
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        t = np.zeros((*shape, d, d))
        i, j = np.tril_indices(d, -1)
        t[..., i, j] = rng.standard_normal((*shape, len(i)))
        diag = np.arange(d)
        t[..., diag, diag] = np.sqrt(rng.chisquare((nu - (d - 1)) + diag, size=(*shape, d)))
        if np.any(t[..., diag, diag] == 0):
            raise FloatingPointError("inverse-Wishart draw has an unrepresentable chi diagonal")
        root = np.linalg.solve(t.swapaxes(-1, -2), row_chol.T).swapaxes(-1, -2)
        root = root * row_scale[:, None]
        cov = (root @ root.swapaxes(-1, -2)).astype(dtype)
        z = rng.standard_normal((*shape, *center.shape))
        if regression:
            # Z inv(chol(lam)) has column covariance inv(lam).
            z = np.linalg.solve(col_chol.T, z.swapaxes(-1, -2)).swapaxes(-1, -2) / col_scale
            draw = center + root @ z
        else:
            draw = center + (root @ (z / np.sqrt(kappa))[..., None])[..., 0]
        draw = draw.astype(dtype)
        if not np.isfinite(draw).all():
            raise FloatingPointError("conditional Gaussian draw is outside the output dtype")
        try:
            ef.resolved_correlation(cov)
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise FloatingPointError("drawn covariance is unresolved at output precision") from exc
    return draw, cov

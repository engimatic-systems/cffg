"""Column-scaled validation and solves for upper precision roots."""

import numpy as np

from cffg import ef


def scaled_root(root, dimension):
    root = np.asarray(root)
    if root.shape != (dimension, dimension) or root.dtype not in (
        np.float32,
        np.float64,
    ):
        raise ValueError(
            "column precision root must be a square float32 or float64 matrix"
        )
    if (
        not np.isfinite(root).all()
        or np.any(np.tril(root, -1) != 0)
        or np.any(np.diag(root) <= 0)
    ):
        raise ValueError(
            "column precision root must be finite and upper triangular with positive diagonal"
        )
    scale = np.max(np.abs(root), axis=0).astype(np.float64)
    normalized = root / scale
    singular = np.linalg.svd(normalized, compute_uv=False)
    # Rank admission concerns the stored root, before squaring its condition number.
    if singular[-1] <= dimension * np.finfo(root.dtype).eps * singular[0]:
        raise ValueError("column precision root unresolved at stored precision")
    return normalized, scale


def dense_precision(root):
    """Check the Gram actually returned, including its weak directions."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            wide = np.asarray(root, np.float64)
            dense = (wide.T @ wide).astype(root.dtype)
            corr, scale = ef.resolved_correlation(dense)
            # Whiten the rounded Gram's Cholesky factor with the original root.
            ratio = np.linalg.solve((wide / scale).T, np.linalg.cholesky(corr))
            error = np.linalg.norm(ratio @ ratio.T - np.eye(len(root)), ord=2)
            # Keep at least half the source significand in every quadratic form.
            if error <= np.sqrt(np.finfo(root.dtype).eps):
                return dense
    except (FloatingPointError, np.linalg.LinAlgError):
        pass
    raise ValueError(
        "dense column precision unresolved; use lam_root, precision_quadratic or solve_precision"
    )

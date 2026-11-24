"""Batched NIW and Dirichlet expected log likelihoods for mixed observations."""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import digamma

from cffg.efrules import LOG2PI, _symmetry_ok


def _array(name, value, shape, dtype):
    if not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be an array")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")


def _check_bank_shapes(bank):
    if not hasattr(bank.mu, "dtype") or bank.mu.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("mu must be a float32 or float64 array")
    if jax.dtypes.canonicalize_dtype(bank.mu.dtype) != bank.mu.dtype:
        raise TypeError(f"bank dtype {bank.mu.dtype} is disabled in JAX")
    if bank.mu.ndim != 2 or min(bank.mu.shape) < 1:
        raise ValueError("mu must have shape [K,d] with positive dimensions")
    k, d = bank.mu.shape
    for field in ("kappa", "nu", "alpha"):
        _array(field, getattr(bank, field), (k,), bank.mu.dtype)
    _array("psi", bank.psi, (k, d, d), bank.mu.dtype)
    _array("active", bank.active, (k,), jnp.bool_)
    if not isinstance(bank.categories, tuple):
        raise TypeError("bank.categories must be a tuple")
    for head in bank.categories:
        if not hasattr(head, "dtype"):
            raise TypeError("each categorical prior must be an array")
        if head.ndim != 2 or head.shape[-1] < 1:
            raise ValueError("each categorical prior must have shape [K,C] with C >= 1")
        _array("categorical prior", head, (k, head.shape[1]), bank.mu.dtype)


def _check_batch_shapes(bank, batch):
    if not hasattr(batch.mean, "dtype"):
        raise TypeError("batch.mean must be an array")
    if batch.mean.ndim != 2:
        raise ValueError("mean must have shape [N,d]")
    n, d = batch.mean.shape[0], bank.mu.shape[1]
    _array("mean", batch.mean, (n, d), bank.mu.dtype)
    _array("covariance", batch.covariance, (n, d, d), bank.mu.dtype)
    _array("gaussian_observed", batch.gaussian_observed, (n,), jnp.bool_)
    if not isinstance(batch.categories, tuple):
        raise TypeError("batch.categories must be a tuple")
    if len(bank.categories) != len(batch.categories):
        raise ValueError("batch and bank must have the same number of categorical fields")
    for index in batch.categories:
        if not hasattr(index, "dtype") or not jnp.issubdtype(index.dtype, jnp.integer):
            raise TypeError("category indices must have integer dtype")
        _array("category indices", index, (n,), index.dtype)


def _correlation(matrices):
    diag = jnp.diagonal(matrices, axis1=-2, axis2=-1)
    sd = jnp.sqrt(jnp.where(diag > 0, diag, 1))
    return (matrices / sd[..., :, None]) / sd[..., None, :], sd


def _matrix_valid(matrices, *, strict):
    """Test definiteness in correlation units, including deterministic coordinates."""
    diagonal = jnp.diagonal(matrices, axis1=-2, axis2=-1)
    positive = diagonal > 0
    if strict:
        valid = jnp.all(positive)
    else:
        variable = positive[..., :, None] & positive[..., None, :]
        valid = jnp.all(diagonal >= 0) & jnp.all(variable | (matrices == 0))
    if matrices.shape[-1] == 1:
        return valid & jnp.all(jnp.isfinite(diagonal))
    corr, _ = _correlation(matrices)
    d = matrices.shape[-1]
    # A strict pivot must resolve its sign; a semidefinite observation may
    # have an exact zero eigenvalue up to source-precision roundoff.
    guard = d * jnp.finfo(matrices.dtype).eps * jnp.eye(d, dtype=matrices.dtype)
    guarded = corr - guard if strict else corr + 16 * guard
    return valid & jnp.all(_symmetry_ok(corr)) & jnp.all(jnp.isfinite(jnp.linalg.cholesky(guarded)))


def validate_bank(bank):
    """Check every component, including dormant slots. Return a scalar JAX bool."""
    _check_bank_shapes(bank)
    valid = (jnp.any(bank.active) & jnp.all(jnp.isfinite(bank.mu))
             & _matrix_valid(bank.psi, strict=True))
    for values in (bank.kappa, bank.alpha, *bank.categories):
        valid &= jnp.all(jnp.isfinite(values) & (values > 0))
    return valid & jnp.all(jnp.isfinite(bank.nu) & (bank.nu > bank.mu.shape[1] - 1))


def validate_inputs(bank, batch):
    """Raise for invalid layouts and return a scalar JAX bool for value checks."""
    valid = validate_bank(bank)
    _check_batch_shapes(bank, batch)
    cov = jnp.where(batch.gaussian_observed[:, None, None], batch.covariance, 0)
    valid &= _matrix_valid(cov, strict=False)
    valid &= jnp.all(~batch.gaussian_observed[:, None] | jnp.isfinite(batch.mean))
    for index, head in zip(batch.categories, bank.categories, strict=True):
        if jnp.issubdtype(index.dtype, jnp.signedinteger):
            valid &= jnp.all(index >= -1)
        if head.shape[1] <= jnp.iinfo(index.dtype).max:
            valid &= jnp.all(index < head.shape[1])
    return valid


def _dirichlet_log(alpha, active):
    """Compute psi(a) - psi(S), retaining a small complement S-a.

    Shift both arguments by eight using the digamma recurrence. Thereafter
    the asymptotic series is accurate to about 3e-13 in absolute log score.
    Factoring each difference by the complement preserves near-unit events.
    """
    if alpha.shape[-1] == 1:
        return jnp.zeros_like(alpha)
    support = jnp.where(active, alpha, 0)
    peak = jnp.max(support, axis=-1, keepdims=True)
    relative = support / peak
    total = jnp.sum(relative, axis=-1, keepdims=True)
    # Only the largest concentration can lose its complement in subtraction.
    largest = jnp.arange(alpha.shape[-1]) == jnp.argmax(support, axis=-1, keepdims=True)
    rest = jnp.where(largest, jnp.sum(jnp.where(largest, 0, relative), axis=-1,
                                      keepdims=True), total - relative)

    def complement(shift):
        scale = jnp.maximum(peak, shift)
        scaled_peak = peak / scale
        denominator = total * scaled_peak + shift / scale
        return rest * scaled_peak / denominator, scale, denominator

    shift = jnp.asarray(8, dtype=alpha.dtype)
    share, scale, denominator = complement(shift)
    shifted = alpha + shift
    log_ratio = jnp.where(share < 0.5, jnp.log1p(-share),
                          jnp.log(shifted / scale) - jnp.log(denominator))
    inverse = 1 / shifted
    result = log_ratio - 0.5 * inverse * share
    for power, coefficient in ((2, -1/12), (4, 1/120), (6, -1/252),
                               (8, 1/240), (10, -1/132)):
        result -= coefficient * inverse**power * jnp.expm1(power * log_ratio)

    def subtract_term(i, value):
        shift = jnp.asarray(i, dtype=alpha.dtype)
        fraction, _, _ = complement(shift)
        return value - fraction / (alpha + shift)

    return jax.lax.fori_loop(0, 8, subtract_term, result)


def _covariance_trace(root, covariance, active):
    """Return tr(Psi^-1 C) without cancellation for nearly orthogonal directions."""
    k, d, _ = root.shape
    eye = jnp.broadcast_to(jnp.eye(d, dtype=root.dtype), (k, d, d))
    inverse_root = solve_triangular(root, eye, lower=True)
    dot = jax.lax.Precision.HIGHEST
    precision = jnp.einsum("kli,klj->kij", inverse_root, inverse_root, precision=dot)
    trace = jnp.einsum("kij,nji->nk", precision, covariance, precision=dot)
    magnitude = jnp.einsum("kij,nji->nk", abs(precision), abs(covariance), precision=dot)
    # Limit cancellation in the fast contraction to a factor of eight.
    unstable = ~jnp.isfinite(trace) | (trace < magnitude / 8)

    def whitened_trace():
        corr, sd = _correlation(covariance)
        values, vectors = jnp.linalg.eigh(corr)
        factor = sd[:, :, None] * vectors * jnp.sqrt(jnp.maximum(values, 0))[:, None, :]
        rhs = factor.transpose(1, 0, 2).reshape(d, -1)

        def one(chol):
            whitened = solve_triangular(chol, rhs, lower=True)
            return jnp.sum(whitened.reshape(d, -1, d)**2, axis=(0, 2))

        # Map over components to bound workspace for a large bank.
        return jnp.where(unstable, jax.lax.map(one, root).T, trace)

    return jax.lax.cond(jnp.any(unstable & active), whitened_trace, lambda: trace)


def _gaussian_log(bank, batch):
    """E log N(x | location, precision) under independent observation and NIW."""
    d = bank.mu.shape[1]
    corr, scale = _correlation(bank.psi)
    root = scale[:, :, None] * jnp.linalg.cholesky(corr)
    logdet = 2 * jnp.sum(jnp.log(jnp.diagonal(root, axis1=-2, axis2=-1)), axis=-1)
    orders = (bank.nu[:, None] - jnp.arange(d, dtype=bank.nu.dtype)) / 2
    elogdet = jnp.sum(digamma(orders), axis=-1) + d * jnp.log(2.) - logdet

    mean = jnp.where(batch.gaussian_observed[:, None], batch.mean, 0)
    cov = jnp.where(batch.gaussian_observed[:, None, None], batch.covariance, 0)
    difference = mean.T[None, :, :] - bank.mu[:, :, None]
    whitened = solve_triangular(root, difference, lower=True)
    squared = jnp.sum(jnp.square(whitened), axis=1).T
    trace = _covariance_trace(root, cov, bank.active)
    loglik = 0.5 * (elogdet - d * LOG2PI - d / bank.kappa - bank.nu * (squared + trace))
    return jnp.where(batch.gaussian_observed[:, None], loglik, 0)


def score(bank, batch, *, include_mixing=False):
    """Return absolute likelihood scores [N,K], assignments [N,K], and scalar validity."""
    valid = validate_inputs(bank, batch)
    mixing = jnp.asarray(include_mixing)
    _array("include_mixing", mixing, (), jnp.bool_)
    shape = (batch.mean.shape[0], bank.mu.shape[0])
    invalid = (jnp.full(shape, -jnp.inf, dtype=bank.mu.dtype),
               jnp.zeros(shape, dtype=bank.mu.dtype), jnp.asarray(False))
    if shape[0] == 0:
        return invalid[0], invalid[1], valid

    def relative(term):
        supported = jnp.where(bank.active, term, -jnp.inf)
        return supported - jnp.max(supported, axis=-1, keepdims=True)

    def evaluate():
        scores = _gaussian_log(bank, batch)
        logits = relative(scores)
        for index, head in zip(batch.categories, bank.categories, strict=True):
            table = _dirichlet_log(head, True)
            term = jnp.where(index[:, None] >= 0, table.T[jnp.maximum(index, 0)], 0)
            scores += term
            # The absolute sum may round away a categorical log odds when
            # every component agrees on a very unlikely continuous field.
            logits = relative(logits + relative(term))
        scores = jnp.where(bank.active, scores, -jnp.inf)
        mixing_log = jax.lax.cond(mixing, lambda: _dirichlet_log(bank.alpha, bank.active),
                                  lambda: jnp.zeros_like(bank.alpha))
        weights = jax.nn.softmax(logits + relative(mixing_log), axis=-1)
        finite = (jnp.all(~bank.active | jnp.isfinite(scores))
                  & jnp.all(~bank.active | jnp.isfinite(mixing_log))
                  & jnp.all(jnp.isfinite(weights)))
        return jnp.where(finite, scores, invalid[0]), jnp.where(finite, weights, 0), finite

    return jax.lax.cond(valid, evaluate, lambda: invalid)

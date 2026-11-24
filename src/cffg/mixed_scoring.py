"""Batched NIW and Dirichlet expected log likelihoods for mixed observations."""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import digamma

from cffg import _compensated
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
    carry = getattr(bank, "carry", None)
    if carry is not None:
        for name in ("mu", "kappa", "psi", "nu", "alpha"):
            high = getattr(bank, name)
            _array(f"carry.{name}", getattr(carry, name), high.shape, high.dtype)
        if not isinstance(carry.categories, tuple) or len(carry.categories) != len(bank.categories):
            raise ValueError("carry must contain one low array for each categorical field")
        for high, low in zip(bank.categories, carry.categories, strict=True):
            _array("categorical carry", low, high.shape, high.dtype)


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


def category_indices(index):
    """Use an integer wide enough for JAX to address the entire category axis."""
    return index.astype(jnp.int32) if index.dtype.itemsize < 4 else index


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


def _parameters_valid(bank, scale_valid):
    valid = jnp.any(bank.active) & jnp.all(jnp.isfinite(bank.mu)) & scale_valid
    for values in (bank.kappa, bank.alpha, *bank.categories):
        valid &= jnp.all(jnp.isfinite(values) & (values > 0))
    carry = getattr(bank, "carry", None)
    if carry is not None:
        for name in ("mu", "kappa", "psi", "nu", "alpha"):
            valid &= _compensated.canonical(getattr(bank, name), getattr(carry, name))
        for high, low in zip(bank.categories, carry.categories, strict=True):
            valid &= _compensated.canonical(high, low)
    return valid & jnp.all(jnp.isfinite(bank.nu) & (bank.nu > bank.mu.shape[1] - 1))


_compiled_parameters_valid = jax.jit(_parameters_valid)


def validate_bank(bank, *, for_checkpoint=False):
    """Check every component, including dormant slots. Return a scalar JAX bool."""
    # Check source dtypes before JAX can narrow a host array on dispatch.
    _check_bank_shapes(bank)
    # Keep the checkpoint's matrix operations in their original precision
    # and sequence: fusion can change a Cholesky pivot at the rank boundary.
    scale_valid = _matrix_valid(bank.psi, strict=True)
    parameters_valid = _compiled_parameters_valid if for_checkpoint else _parameters_valid
    return parameters_valid(bank, scale_valid)


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

    Shift both arguments by eight using the digamma recurrence. The first
    omitted asymptotic term is then below machine epsilon in each dtype.
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
    terms = ((2, -1/12), (4, 1/120), (6, -1/252), (8, 1/240), (10, -1/132))
    if alpha.dtype == jnp.float64:
        terms += ((12, 691/32760), (14, -1/12), (16, 3617/8160))
    for power, coefficient in terms:
        result -= coefficient * inverse**power * jnp.expm1(power * log_ratio)

    def subtract_term(i, value):
        shift = jnp.asarray(i, dtype=alpha.dtype)
        fraction, _, _ = complement(shift)
        return value - fraction / (alpha + shift)

    return jax.lax.fori_loop(0, 8, subtract_term, result)


def _digamma_adjustment(x, relative):
    """Change from x to x*(1+relative), with a sub-ULP relative perturbation."""
    # At small x the pole can amplify a low word into observable log odds.
    # Its reciprocal difference is exact algebraically; the regular part
    # needs only a first-order correction at this perturbation size.
    small = x < 16
    safe = jnp.where(small, x, 1)
    # After two recurrences the alternating trigamma expansion has error
    # below 1/(30*(x+2)**9). This derivative multiplies a sub-ULP low word.
    inv2 = 1 / (safe + 2)
    regular = 1 / (safe + 1)**2 + inv2 * (
        1 + inv2/2 + inv2**2/6 - inv2**4/30 + inv2**6/42)
    near = relative / (safe * (1 + relative)) + (relative * safe) * regular
    inv = 1 / jnp.where(small, 16, x)
    far = relative * (1 + inv/2 + inv**2/6 - inv**4/30 + inv**6/42)
    return jnp.where(small, near, far)


def _dirichlet_adjustment(alpha, low, active, base):
    support = jnp.where(active, alpha, 0)
    peak = jnp.max(support, axis=-1, keepdims=True)
    relative = jnp.sum(jnp.where(active, low, 0) / peak, axis=-1, keepdims=True)
    relative /= jnp.sum(support / peak, axis=-1, keepdims=True)
    correction = (_digamma_adjustment(alpha, low / alpha)
                  - _digamma_adjustment(jnp.sum(support, axis=-1, keepdims=True), relative))
    # For a near-unit event even the entire log likelihood is below epsilon.
    # The high-word complement already resolves it to relative source precision.
    return jnp.where(jnp.abs(base) < jnp.finfo(alpha.dtype).eps, 0, correction)


def _covariance_trace(root, inverse_root, covariance, active):
    """Return tr(Psi^-1 C) without cancellation for nearly orthogonal directions."""
    d = root.shape[-1]
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


def _scale_response(perturbation, score_scale, active):
    """Return the inverse-scale change and log determinant in whitened coordinates.

    For symmetric D with spectral radius at most r < 1, replacing
    (I+D)^-1 D by D has spectral-norm error at most r²/(1-r).
    Replacing logdet(I+D) by tr(D) has error at most d*r²/(2*(1-r)).
    score_scale includes d/2 and bounds the absolute quadratic multiplier.
    """
    d = perturbation.shape[-1]
    radius = jnp.max(jnp.sum(jnp.abs(perturbation), axis=-1), axis=-1)
    # Multiply by score_scale first: r² can underflow for a distant reading.
    small = ((radius < 0.5) & ((score_scale * radius) * radius
                               <= jnp.finfo(perturbation.dtype).eps * (1 - radius) / 4))

    def exact():
        identity = jnp.eye(d, dtype=perturbation.dtype)
        change = jnp.linalg.solve(identity + perturbation, perturbation)
        log_volume = jnp.sum(jnp.log1p(jnp.linalg.eigvalsh(perturbation)), axis=-1)
        return change, log_volume

    return jax.lax.cond(
        jnp.all(~active | small),
        lambda: (perturbation, jnp.trace(perturbation, axis1=-2, axis2=-1)),
        exact,
    )


def _gaussian_log(bank, batch):
    """Return the NIW expected log likelihood and its low-word correction."""
    d = bank.mu.shape[1]
    corr, scale = _correlation(bank.psi)
    root = scale[:, :, None] * jnp.linalg.cholesky(corr)
    logdet = 2 * jnp.sum(jnp.log(jnp.diagonal(root, axis1=-2, axis2=-1)), axis=-1)
    carry = getattr(bank, "carry", None)
    degrees = bank.nu[:, None] - jnp.arange(d, dtype=bank.nu.dtype)
    orders = (degrees if carry is None else degrees + carry.nu[:, None]) / 2
    elogdet = jnp.sum(digamma(orders), axis=-1) + d * jnp.log(2.) - logdet

    mean = jnp.where(batch.gaussian_observed[:, None], batch.mean, 0)
    cov = jnp.where(batch.gaussian_observed[:, None, None], batch.covariance, 0)
    difference = mean.T[None, :, :] - bank.mu[:, :, None]
    whitened = solve_triangular(root, difference, lower=True)
    squared = jnp.sum(jnp.square(whitened), axis=1).T
    eye = jnp.broadcast_to(jnp.eye(d, dtype=root.dtype), root.shape)
    inverse = solve_triangular(root, eye, lower=True)
    trace = _covariance_trace(root, inverse, cov, bank.active)
    loglik = 0.5 * (elogdet - d * LOG2PI - d / bank.kappa - bank.nu * (squared + trace))
    observed = batch.gaussian_observed[:, None]
    base = jnp.where(observed, loglik, 0)
    if carry is None:
        return base, jnp.zeros_like(base)
    low_mean = solve_triangular(root, carry.mu[..., None], lower=True)
    delta = jnp.sum(low_mean * (low_mean - 2 * whitened), axis=1).T

    def scale_adjustment():
        # In whitened coordinates the effective scale is I+D. Solve for
        # (I+D)^-1 D directly, retaining precision corrections below one ULP
        # of I that can still change odds for distant observations.
        dot = jax.lax.Precision.HIGHEST
        def product(a, b):
            return jnp.matmul(a, b, precision=dot)
        def transpose(a):
            return a.swapaxes(-1, -2)
        perturbation = product(product(inverse, carry.psi), transpose(inverse))
        perturbation = 0.5 * perturbation + 0.5 * transpose(perturbation)
        centered = whitened - low_mean
        location_bound = jnp.max(jnp.sum(centered**2, axis=1), axis=-1)
        spread_bound = (d * jnp.max(jnp.abs(cov))) * jnp.sum(inverse**2, axis=(-2, -1))
        weight = jnp.abs(bank.nu) + jnp.abs(carry.nu)
        change, log_volume = _scale_response(
            perturbation, d / 2 + weight * (location_bound + spread_bound), bank.active)
        location = jnp.sum(centered * product(change, centered), axis=1).T
        precision_change = product(product(transpose(inverse), change), inverse)
        spread = jnp.einsum("kij,nji->nk", precision_change, cov, precision=dot)
        return location + spread, log_volume

    scale_delta, volume = jax.lax.cond(
        jnp.any(carry.psi != 0), scale_adjustment,
        lambda: (jnp.zeros_like(base), jnp.zeros_like(bank.nu)),
    )
    delta -= scale_delta
    correction = 0.5 * ((d / bank.kappa) * (carry.kappa / bank.kappa)
                        - volume - bank.nu * delta - carry.nu * (squared + trace + delta))
    return base, jnp.where(observed, correction, 0)


def score(bank, batch, *, include_mixing=False):
    """Return source-dtype absolute scores [N,K], assignments [N,K], and scalar validity.

    Low-word corrections enter the relative logits separately from large
    common likelihoods. The NIW uses the resolved primary scale and its
    whitened perturbation. Dirichlet corrections use the digamma recurrence
    at its pole and its derivative for the smooth sub-ULP displacement.
    """
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
        base, correction = _gaussian_log(bank, batch)
        scores = base + correction
        logits = relative(relative(base) + relative(correction))
        carry = getattr(bank, "carry", None)
        for i, (index, head) in enumerate(zip(batch.categories, bank.categories, strict=True)):
            index = category_indices(index)
            table = _dirichlet_log(head, True)
            term = jnp.where(index[:, None] >= 0, table.T[jnp.maximum(index, 0)], 0)
            scores += term
            # The absolute sum may round away a categorical log odds when
            # every component agrees on a very unlikely continuous field.
            logits = relative(logits + relative(term))
            if carry is not None:
                adjust = _dirichlet_adjustment(head, carry.categories[i], True, table)
                term = jnp.where(index[:, None] >= 0, adjust.T[jnp.maximum(index, 0)], 0)
                scores += term
                logits = relative(logits + relative(term))
        scores = jnp.where(bank.active, scores, -jnp.inf)
        mixing_log = jax.lax.cond(mixing, lambda: _dirichlet_log(bank.alpha, bank.active),
                                  lambda: jnp.zeros_like(bank.alpha))
        logits = logits + relative(mixing_log)
        if carry is not None:
            adjust = jax.lax.cond(mixing, lambda: _dirichlet_adjustment(
                bank.alpha, carry.alpha, bank.active, mixing_log), lambda: jnp.zeros_like(bank.alpha))
            logits += relative(adjust)
        weights = jax.nn.softmax(logits, axis=-1)
        finite = (jnp.all(~bank.active | jnp.isfinite(scores))
                  & jnp.all(~bank.active | jnp.isfinite(mixing_log))
                  & jnp.all(jnp.isfinite(weights)))
        return jnp.where(finite, scores, invalid[0]), jnp.where(finite, weights, 0), finite

    return jax.lax.cond(valid, evaluate, lambda: invalid)

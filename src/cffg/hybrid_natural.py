"""D12 hybrid natural coordinates and moment projections, without engine binding.

One label block represents ``exp(a + h @ (x-c) - (x-c) @ J @ (x-c) / 2)``.
J includes the optional derived ``precision_correction`` beside ``precision``.
The retained origin c is fixed during a solve, never a stationary unknown.
Absent origins mean zero: already-expanded inputs keep their stored polynomial
meaning and cannot recover coefficients rounded away by their caller. Only
a common shift of ``a`` is gauge. Messages may have signed precision; the
aggregate or tilted belief must be proper. Numerical failures return a bitmask
in both eager and transformed execution; ``raise_for_status`` is the host gate.
"""

from enum import IntFlag
from typing import NamedTuple

import jax
import jax.numpy as jnp

from cffg.hybrid_affine import affine, rounded_product
from cffg import hybrid_covariance as hc


class Status(IntFlag):
    OK = 0
    INVALID = 1
    EMPTY_SUPPORT = 2
    IMPROPER = 4
    SUPPORT_CHANGED = 8
    UNDERFLOW = 16
    NONFINITE = 32


def failure(condition, flag):
    return jnp.where(condition, jnp.int32(flag), jnp.int32(0))


def raise_for_status(status, context="hybrid region"):
    """Reject a failed numerical result at a host boundary, never inside JIT."""
    code = int(jnp.bitwise_or.reduce(jnp.asarray(status).reshape(-1)))
    if code:
        names = ", ".join(flag.name.lower() for flag in Status if code & flag)
        raise ValueError(f"{context}: {names}")


class Natural(NamedTuple):
    constant: jax.Array
    linear: jax.Array
    precision: jax.Array
    status: jax.Array = 0
    center: jax.Array | None = None
    precision_correction: jax.Array | None = None

    @property
    def correction(self):
        """Derived low word of assembled precision; never a message unknown."""
        return (jnp.zeros_like(self.precision) if self.precision_correction is None
                else self.precision_correction)

    @property
    def origin(self):
        return jnp.zeros_like(self.linear) if self.center is None else self.center

    def __add__(self, other):
        check_shapes(self)
        check_shapes(other)
        if any(a.shape != b.shape or a.dtype != b.dtype
               for a, b in zip(self.arrays(), other.arrays())):
            raise ValueError("natural coordinate shapes or dtypes disagree")
        center = other.center if self.center is None else self.center
        left, right = (rebase(value, center) for value in (self, other))
        constant = left.constant + right.constant
        linear = left.linear + right.linear
        precision, correction = hc.combine(left.precision, right.precision,
                                             left.correction, right.correction)
        low_status = correction_status(left.precision, left.precision_correction, name="precision")
        low_status |= correction_status(right.precision, right.precision_correction, name="precision")
        overflow = jnp.isfinite(left.constant) & jnp.isfinite(right.constant) & ~jnp.isfinite(constant)
        return Natural(jnp.where(overflow, jnp.nan, constant), linear, precision,
                       left.status | right.status | low_status | failure(jnp.any(overflow), Status.NONFINITE),
                       center, correction)

    def arrays(self):
        """Three message coefficient arrays; derived precision storage is separate."""
        return self.constant, self.linear, self.precision


class Belief(NamedTuple):
    probability: jax.Array
    displacement: jax.Array
    covariance: jax.Array
    log_probability: jax.Array
    log_normalizer: jax.Array
    status: jax.Array
    center: jax.Array
    covariance_correction: jax.Array | None = None

    @property
    def correction(self):
        """Low word: represented covariance is covariance + correction."""
        return (jnp.zeros_like(self.covariance) if self.covariance_correction is None
                else self.covariance_correction)

    @property
    def full_covariance(self):
        """Physical readout, rounded once to the belief dtype."""
        return (self.covariance if self.covariance_correction is None else
                self.covariance + self.covariance_correction)

    @property
    def mean(self):
        """World-coordinate readout; small displacements may round at large origins."""
        active = ~jnp.isneginf(self.log_probability)
        return jnp.where(active[..., None], self.center + self.displacement, 0)

    def relative_mean(self, center):
        """Change origins without a world-mean round trip."""
        return self.displacement + (self.center - center)

    @property
    def second(self):
        return _second_moment(self.full_covariance, self.mean)

    def entropy(self):
        """Labeled entropy, including the Gaussian constants for every label."""
        low_status = correction_status(self.covariance, self.covariance_correction)
        active = self.probability > 0
        safe = jnp.where(active[..., None, None], self.covariance,
                         jnp.eye(self.displacement.shape[-1], dtype=self.displacement.dtype))
        logdet = (jnp.linalg.slogdet(safe)[1] if self.covariance_correction is None else
                  hc.logdet(safe, jnp.where(active[..., None, None], self.correction, 0)))
        gaussian = (self.displacement.shape[-1] * (1 + jnp.log(2 * jnp.pi)) + logdet) / 2
        value = jnp.sum(jnp.where(active, self.probability *
                                 (gaussian - jnp.where(active, self.log_probability, 0)), 0))
        status = jnp.asarray(self.status, jnp.int32) | low_status
        return jnp.where(status == 0, value, jnp.nan)

    def expectation(self, natural):
        check_shapes(natural)
        if natural.linear.shape != self.displacement.shape:
            raise ValueError("belief and potential shapes disagree")
        low_status = correction_status(self.covariance, self.covariance_correction)
        active = self.probability > 0
        mean = self.relative_mean(natural.origin)
        value = jnp.where(active, natural.constant, 0)
        value += jnp.einsum("...d,...d->...", natural.linear, mean)
        trace = jnp.einsum("...ab,...ab->...", natural.precision, self.covariance)
        if self.covariance_correction is not None:
            trace += jnp.einsum("...ab,...ab->...", natural.precision, self.covariance_correction)
        trace += jnp.einsum("...ab,...ab->...", natural.correction, self.full_covariance)
        quadratic = affine(natural.precision, mean[..., None, :],
                           affine(natural.correction, mean[..., None, :]))
        value -= (trace + jnp.einsum("...a,...a->...", mean, quadratic)) / 2
        value = jnp.sum(jnp.where(active, self.probability * value, 0))
        status = (jnp.asarray(self.status, jnp.int32) | coordinate_status(natural)
                  | low_status)
        return jnp.where(status == 0, value, jnp.nan)


def _second_moment(covariance, mean):
    """Share rounded intermediates between the readout and overflow guards."""
    outer = rounded_product(mean[..., :, None], mean[..., None, :])
    return covariance + outer


def correction_status(covariance, correction, *, name="covariance"):
    """Validate a retained low word, including directly supplied belief fields."""
    if correction is None:
        return jnp.int32(0)
    if correction.shape != covariance.shape or correction.dtype != covariance.dtype:
        raise ValueError(f"{name} correction shapes or dtypes disagree")
    valid = jnp.all(correction == correction.swapaxes(-1, -2))
    # A low word is bounded by one rounding unit of its high word.
    valid &= jnp.all(abs(correction) <= jnp.finfo(covariance.dtype).eps * abs(covariance))
    return failure(~valid, Status.INVALID) | failure(~hc.supported(correction), Status.NONFINITE)


def check_shapes(natural):
    a, h, matrix = natural.arrays()
    if a.dtype not in (jnp.dtype("float32"), jnp.dtype("float64")):
        raise ValueError("coordinates must use float32 or float64")
    if h.ndim != a.ndim + 1 or h.shape[:-1] != a.shape or h.shape[-1] < 1:
        raise ValueError("natural coordinate shapes disagree")
    if matrix.shape != h.shape + (h.shape[-1],):
        raise ValueError("natural coordinate shapes disagree")
    if h.dtype != a.dtype or matrix.dtype != a.dtype:
        raise ValueError("natural coordinate dtypes disagree")
    if natural.center is not None:
        if natural.center.shape != h.shape:
            raise ValueError("center shapes disagree")
        if natural.center.dtype != a.dtype:
            raise ValueError("center dtypes disagree")


def coordinate_status(natural):
    """Finite or literal log-zero constants; finite symmetric Gaussian blocks."""
    check_shapes(natural)
    a, h, matrix = natural.arrays()
    finite = jnp.all(jnp.isfinite(a) | jnp.isneginf(a))
    finite &= jnp.all(jnp.isfinite(h)) & jnp.all(jnp.isfinite(matrix))
    finite &= jnp.all(jnp.isfinite(natural.origin))
    symmetric = jnp.all(matrix == matrix.swapaxes(-1, -2))
    return (jnp.asarray(natural.status, jnp.int32) | failure(~finite, Status.NONFINITE)
            | failure(~symmetric, Status.INVALID)
            | correction_status(matrix, natural.precision_correction, name="precision")
            | failure(~jnp.any(jnp.isfinite(a)), Status.EMPTY_SUPPORT))


def rebase(natural, center):
    """Express the same stored polynomial at a new fixed origin.

    With delta=c_new-c_old, h_new=h-J delta and
    a_new=a+h delta-delta'J delta/2. Large changes of origin can round
    coefficients; initialize near retained sites and freeze origins in a solve.
    """
    check_shapes(natural)
    # Fixed-width status avoids mixed weak-int batching in backend bitwise OR.
    natural = natural._replace(status=jnp.asarray(natural.status, jnp.int32))
    result = natural._replace(center=center)
    check_shapes(result)
    if center is None and natural.center is None:
        return natural
    delta = result.origin - natural.origin
    linear = affine(-natural.precision, delta[..., None, :], natural.linear,
                    affine(-natural.correction, delta[..., None, :]))
    midpoint = affine(-natural.precision / 2, delta[..., None, :], natural.linear,
                      affine(-natural.correction / 2, delta[..., None, :]))
    absent = jnp.isneginf(natural.constant)
    constant = affine(delta, midpoint, jnp.where(absent, 0, natural.constant))
    constant = jnp.where(absent, -jnp.inf, constant)
    overflow = jnp.isfinite(natural.constant) & ~jnp.isfinite(constant)
    result = Natural(jnp.where(overflow, jnp.nan, constant), linear, natural.precision,
                     natural.status | failure(jnp.any(overflow), Status.NONFINITE), center,
                     natural.precision_correction)
    return result._replace(status=coordinate_status(result))


def gaussian_blocks(matrix, active):
    """Inverse and log determinant, rejecting unresolved active PD blocks.

    D12's ``32 d eps * spectral_scale`` resolution protects a strictly
    positive integral. Identity substitutes only protect failed/inactive
    arithmetic; no substitute is returned as a successful belief.
    """
    dim = matrix.shape[-1]
    identity = jnp.eye(dim, dtype=matrix.dtype)
    finite = jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
    safe = jnp.where((active & finite)[..., None, None], matrix, identity)
    eigenvalues = jnp.linalg.eigvalsh(safe)
    threshold = 32 * dim * jnp.finfo(matrix.dtype).eps * jnp.max(abs(eigenvalues), axis=-1)
    proper = finite & (eigenvalues[..., 0] > threshold)
    safe = jnp.where((active & proper)[..., None, None], safe, identity)
    chol = jnp.linalg.cholesky(safe)
    inverse = jax.scipy.linalg.cho_solve((chol, True), jnp.broadcast_to(identity, matrix.shape))
    inverse = (inverse + inverse.swapaxes(-1, -2)) / 2
    logdet = 2 * jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=-1)
    status = failure(jnp.any(active & ~proper), Status.IMPROPER)
    finite_result = jnp.all(jnp.isfinite(inverse), axis=(-2, -1)) & jnp.isfinite(logdet)
    status |= failure(jnp.any(active & ~finite_result), Status.NONFINITE)
    return inverse, logdet, status


def represented_status(matrix, correction, active, high_status):
    """Unresolved spectral intervals fail independently of the high-word test."""
    if correction is None:
        return jnp.int32(0)
    active = active & (high_status == 0)
    finite = jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
    safe = jnp.where((active & finite)[..., None, None], matrix,
                     jnp.eye(matrix.shape[-1], dtype=matrix.dtype))
    low = jnp.where(active[..., None, None], correction, 0)
    return failure(jnp.any(active & ~hc.resolved(safe, low)), Status.NONFINITE)


def normalize(natural):
    """Exact Gaussian integration on a fixed label face, D12.2.

    Positive-label underflow is a failure and retains the finite log mass;
    it is never reinterpreted as a structural zero. Inactive coordinates are
    zero. An invalid active belief is NaN with a nonzero status.
    """
    status = coordinate_status(natural)
    a, h, matrix = natural.arrays()
    active = ~jnp.isneginf(a)
    covariance, logdet, pd_status = gaussian_blocks(matrix, active)
    pd_status |= represented_status(matrix, natural.precision_correction, active, pd_status)
    eye = jnp.broadcast_to(jnp.eye(h.shape[-1], dtype=h.dtype), matrix.shape)
    safe = jnp.where((active & (pd_status == 0))[..., None, None], matrix, eye)
    precision_low = jnp.where((active & (pd_status == 0))[..., None, None], natural.correction, 0)
    covariance, correction = hc.inverse(safe, precision_low)
    logdet = hc.logdet(safe, precision_low)
    mean = hc.product(covariance, jnp.where(active[..., None], h, 0)[..., None],
                      hc.product(correction, jnp.where(active[..., None], h, 0)[..., None]))[..., 0]
    mass = (h.shape[-1] * jnp.log(2 * jnp.pi) - logdet
            + jnp.einsum("...d,...d->...", h, mean)) / 2
    # Remove the common gauge before it can erase label-dependent integrals.
    constant_shift = jnp.max(jnp.where(active, a, -jnp.inf))
    constant_shift = jnp.where(jnp.any(active), constant_shift, 0)
    scores = jnp.where(active, a - constant_shift + mass, -jnp.inf)
    score_shift = jnp.max(scores)
    score_shift = jnp.where(jnp.isfinite(score_shift), score_shift, 0)
    centered = scores - score_shift
    log_total = jax.scipy.special.logsumexp(centered)
    log_probability = jnp.where(active, centered - log_total, -jnp.inf)
    normalizer = constant_shift + (score_shift + log_total)
    probability = jnp.where(active, jnp.exp(log_probability), 0)
    status |= pd_status | correction_status(covariance, correction)
    status |= failure(~jnp.isfinite(normalizer), Status.NONFINITE)
    status |= failure(jnp.any(active & (probability == 0)), Status.UNDERFLOW)
    status |= failure(~jnp.all(jnp.isfinite(mean)) | ~jnp.all(jnp.isfinite(covariance)), Status.NONFINITE)
    world_mean = natural.origin + mean
    second = _second_moment(covariance + correction, world_mean)
    status |= failure(jnp.any(active[..., None, None] & ~jnp.isfinite(second)), Status.NONFINITE)
    return Belief(jnp.where(active, jnp.where(status == 0, probability, jnp.nan), 0),
                  jnp.where(active[..., None], jnp.where(status == 0, mean, jnp.nan), 0),
                  jnp.where(active[..., None, None], jnp.where(status == 0, covariance, jnp.nan), 0),
                  log_probability, jnp.where(status == 0, normalizer, jnp.nan), status, natural.origin,
                  jnp.where(active[..., None, None], jnp.where(status == 0, correction, jnp.nan), 0))


def from_moments(probability, mean, covariance):
    """Construct a checked normalized moment belief for a projection or caller."""
    return from_centered_moments(probability, mean, covariance, jnp.zeros_like(mean))


def from_centered_moments(probability, displacement, covariance, center, *, covariance_correction=None):
    """Construct moments relative to a retained origin; no world-mean subtraction.

    The optional low word represents an additive covariance remainder, bounded
    elementwise by eps*abs(covariance). Both words must be finite and symmetric;
    this is derived numerical storage, never an independent solve coordinate.
    """
    check_shapes(Natural(probability, displacement, covariance, center=center))
    correction = jnp.zeros_like(covariance) if covariance_correction is None else covariance_correction
    low_status = correction_status(covariance, covariance_correction)
    mean = displacement
    active = probability > 0
    _, _, pd_status = gaussian_blocks(covariance, active)
    pd_status |= represented_status(covariance, covariance_correction, active, pd_status)
    finite = (jnp.all(jnp.isfinite(probability)) & jnp.all(jnp.isfinite(mean))
              & jnp.all(jnp.isfinite(covariance)) & jnp.all(jnp.isfinite(center)))
    world_mean = center + mean
    represented_covariance = covariance if covariance_correction is None else covariance + correction
    second = _second_moment(represented_covariance, world_mean)
    finite &= jnp.all(jnp.isfinite(second)) & jnp.all(jnp.isfinite(correction))
    # Sum roundoff scales with the number of label additions and the dtype.
    tolerance = 32 * probability.size * jnp.finfo(probability.dtype).eps
    valid = jnp.all(probability >= 0) & (abs(probability.sum() - 1) <= tolerance)
    valid &= jnp.all(covariance == covariance.swapaxes(-1, -2))
    status = (pd_status | low_status | failure(~finite, Status.NONFINITE)
              | failure(~valid, Status.INVALID) | failure(~jnp.any(active), Status.EMPTY_SUPPORT))
    log_probability = jnp.where(active, jnp.log(jnp.where(active, probability, 1)), -jnp.inf)
    return Belief(jnp.where(status == 0, probability, jnp.nan),
                  jnp.where(active[..., None], jnp.where(status == 0, mean, jnp.nan), 0),
                  jnp.where(active[..., None, None], jnp.where(status == 0, covariance, jnp.nan), 0),
                  log_probability, jnp.zeros((), probability.dtype), status, center,
                  None if covariance_correction is None else
                  jnp.where(active[..., None, None], jnp.where(status == 0, correction, jnp.nan), 0))


def gaussian_site(log_mass, mean, covariance, *, center=None):
    """Retain N(x;mean,covariance) times exp(log_mass), including every scale.

    The default fixed origin is the supplied mean. An explicit nearby origin
    is useful when several sites share a separator; choose it once before a
    solve. Use this constructor before expanding any large quadratic constant.
    Exact log-zero labels are allowed; vacuous/signed sites use Natural.
    """
    center = mean if center is None else center
    check_shapes(Natural(log_mass, mean, covariance, center=center))
    active = jnp.isfinite(log_mass)
    precision, logdet, status = gaussian_blocks(covariance, active)
    displacement = mean - center
    linear = jnp.einsum("...ab,...b->...a", precision, displacement)
    constant = log_mass - (mean.shape[-1] * jnp.log(2 * jnp.pi) + logdet
                           + jnp.einsum("...d,...d->...", displacement, linear)) / 2
    valid = (jnp.all(jnp.isfinite(mean)) & jnp.all(jnp.isfinite(covariance))
             & jnp.all(jnp.isfinite(log_mass) | jnp.isneginf(log_mass)))
    status |= failure(~valid, Status.NONFINITE)
    status |= failure(jnp.any(active & ~jnp.isfinite(constant)) | ~jnp.all(jnp.isfinite(linear)), Status.NONFINITE)
    status |= failure(~jnp.all(covariance == covariance.swapaxes(-1, -2)), Status.INVALID)
    result = Natural(jnp.where(active, jnp.where(status == 0, constant, jnp.nan), log_mass),
                     jnp.where(active[..., None], linear, 0),
                     jnp.where(active[..., None, None], precision, 0), status, center)
    return result._replace(status=coordinate_status(result))


def canonical(belief):
    """Normalized hybrid density, retaining label-dependent scales, D12.6."""
    p, mean, covariance = belief.probability, belief.displacement, belief.covariance
    low_status = correction_status(covariance, belief.covariance_correction)
    active = p > 0
    _, _, pd_status = gaussian_blocks(covariance, active)
    pd_status |= represented_status(covariance, belief.covariance_correction, active, pd_status)
    eye = jnp.broadcast_to(jnp.eye(mean.shape[-1], dtype=mean.dtype), covariance.shape)
    safe = jnp.where(active[..., None, None], covariance, eye)
    correction = jnp.where(active[..., None, None], belief.correction, 0)
    precision, precision_low = hc.inverse(safe, correction)
    logdet = hc.logdet(safe, correction)
    linear = hc.product(precision, mean[..., None], hc.product(precision_low, mean[..., None]))[..., 0]
    constant = belief.log_probability - (logdet + mean.shape[-1] * jnp.log(2 * jnp.pi)
                                         + jnp.einsum("...d,...d->...", mean, linear)) / 2
    status = (jnp.asarray(belief.status, jnp.int32) | pd_status
              | low_status)
    status |= failure(jnp.any(active & ~jnp.isfinite(constant))
                      | ~jnp.all(jnp.isfinite(linear)), Status.NONFINITE)
    result = Natural(jnp.where(active, jnp.where(status == 0, constant, jnp.nan), -jnp.inf),
                     jnp.where(active[..., None], jnp.where(status == 0, linear, jnp.nan), 0),
                     jnp.where(active[..., None, None], jnp.where(status == 0, precision, jnp.nan), 0),
                     status, belief.center,
                     jnp.where(active[..., None, None], jnp.where(status == 0, precision_low, jnp.nan), 0))
    return result._replace(status=coordinate_status(result))


def quotient(belief, cavity):
    """D12.7 full natural quotient, with one common constant fixed to zero.

    A tilt can restrict a cavity face. It cannot revive a cavity's absent
    label; that requires rebuilding the active inventory. Signed precision
    is preserved and inactive Gaussian coordinates remain zero.

    For C=C_hi+C_lo, solve C J_out=I-C_hi J_c-C_lo J_c and
    C h_out=mean-C_hi h_c-C_lo h_c. Compensated products retain the
    small residual before the Cholesky solve; inverse-minus-cavity would
    lose it. The constant retains the complete labeled normalization.
    """
    moments = Natural(belief.probability, belief.displacement, belief.covariance, center=belief.center)
    check_shapes(moments)
    cavity = rebase(cavity, belief.center)
    if any(a.shape != b.shape or a.dtype != b.dtype
           for a, b in zip(moments.arrays(), cavity.arrays())):
        raise ValueError("quotient shapes or dtypes disagree")
    active = belief.probability > 0
    _, _, pd_status = gaussian_blocks(belief.covariance, active)
    pd_status |= represented_status(belief.covariance, belief.covariance_correction, active, pd_status)
    status = (jnp.asarray(belief.status, jnp.int32) | pd_status | coordinate_status(cavity)
              | correction_status(belief.covariance, belief.covariance_correction))
    status |= failure(jnp.any(active & ~jnp.isfinite(cavity.constant)), Status.SUPPORT_CHANGED)
    dim = belief.displacement.shape[-1]
    eye = jnp.broadcast_to(jnp.eye(dim, dtype=belief.covariance.dtype), belief.covariance.shape)
    base = jnp.where(active[..., None, None], belief.covariance, eye)
    correction = jnp.where(active[..., None, None], belief.correction, 0)
    precision_rhs = hc.product(-base, cavity.precision,
                               hc.product(-correction, cavity.precision),
                               hc.product(-base, cavity.correction),
                               hc.product(-correction, cavity.correction), eye)
    precision = hc.solve(base, correction, precision_rhs)
    precision = (precision + precision.swapaxes(-1, -2)) / 2
    linear_rhs = hc.residual(base, correction, cavity.linear[..., None], belief.displacement[..., None])
    linear = hc.solve(base, correction, linear_rhs)[..., 0]
    mean = jnp.where(active[..., None], belief.displacement, 0)
    constant = affine(jnp.concatenate((-mean / 2, -mean / 2), axis=-1),
                      jnp.concatenate((cavity.linear, linear), axis=-1),
                      jnp.where(active, belief.log_probability, 0),
                      -hc.logdet(base, correction) / 2,
                      jnp.asarray(-dim * jnp.log(2 * jnp.pi) / 2, mean.dtype),
                      -jnp.where(jnp.isfinite(cavity.constant), cavity.constant, 0))
    first = jnp.argmax(active.reshape(-1))
    constant = affine(jnp.ones_like(constant)[..., None], constant[..., None],
                      -constant.reshape(-1)[first])
    values = [constant, linear, precision]
    outputs = []
    for i, value in enumerate(values):
        mask = active.reshape(active.shape + (1,) * i)
        status |= failure(jnp.any(mask & ~jnp.isfinite(value)), Status.NONFINITE)
        outputs.append(jnp.where(mask, value, -jnp.inf if i == 0 else 0))
    result = Natural(*(jnp.where(status == 0, x, jnp.nan) for x in outputs), status, belief.center)
    return result._replace(status=coordinate_status(result))


def _mix(region, axis, center):
    """Project one label axis around a retained conditional covariance.

    Compensated centering keeps the conditional mean division separate from
    the differences consumed by integer-significand covariance products.
    Ordinary subtraction can be recomputed differently across compiled uses
    near a power of two; the weighted projection tests exercise this boundary.
    """
    input_status = jnp.asarray(region.status, jnp.int32) | correction_status(region.covariance, region.covariance_correction)
    probability = region.probability.sum(axis=axis)
    denominator = jnp.where(probability > 0, probability, 1)
    relative = region.relative_mean(jnp.expand_dims(center, axis))
    weights = jnp.moveaxis(region.probability, axis, -1)
    mean = affine(weights[..., None, :], jnp.moveaxis(relative, axis, -1)) / denominator[..., None]
    centered = affine(jnp.ones_like(relative)[..., None], relative[..., None],
                      -jnp.expand_dims(mean, axis))
    first = jnp.argmax(region.probability > 0, axis=axis)
    indices = jnp.expand_dims(first, axis)[..., None, None]
    base = jnp.take_along_axis(region.covariance, indices, axis=axis).squeeze(axis)
    difference = region.covariance - jnp.expand_dims(base, axis)
    variation = affine(centered[..., :, None, None], centered[..., None, :, None],
                       difference, region.correction)
    increment = affine(weights[..., None, None, :], jnp.moveaxis(variation, axis, -1))
    increment /= denominator[..., None, None]
    covariance, correction = (hc.symmetric(word) for word in hc.combine(base, increment))
    result = from_centered_moments(probability, mean, covariance, center,
                                   covariance_correction=correction)
    status = result.status | input_status
    active = ~jnp.isneginf(result.log_probability)
    fields = {}
    for name, rank in (("probability", 0), ("displacement", 1), ("covariance", 2), ("covariance_correction", 2)):
        mask = active.reshape(active.shape + (1,) * rank)
        fields[name] = jnp.where(mask, jnp.where(status == 0, getattr(result, name), jnp.nan), 0)
    return result._replace(**fields, status=status,
                           log_normalizer=jnp.where(status == 0, result.log_normalizer, jnp.nan))


def collapse_label(region, axis):
    """Mix one label axis, retaining covariance roundoff until the quotient."""
    if axis not in (0, 1) or region.probability.ndim != 2:
        raise ValueError("collapse requires a pair of label axes")
    return _mix(region, axis, jnp.take(region.center, 0, axis=axis))


def marginal(region, side):
    """Projection onto source (0) or destination (1) hybrid moments."""
    if side not in (0, 1) or region.displacement.shape[-1] % 2:
        raise ValueError("grouped marginal needs two equal continuous blocks")
    dim = region.displacement.shape[-1] // 2
    selection = slice(side * dim, (side + 1) * dim)
    reduced = region._replace(displacement=region.displacement[..., selection],
                              center=region.center[..., selection],
                              covariance=region.covariance[..., selection, selection],
                              covariance_correction=None if region.covariance_correction is None else
                              region.covariance_correction[..., selection, selection])
    return collapse_label(reduced, 1 - side)


def lift(left, right):
    """Embed endpoint cavities in (i,j,[x;y]) without merging their labels."""
    check_shapes(left)
    check_shapes(right)
    if left.linear.shape != right.linear.shape or left.constant.ndim != 1:
        raise ValueError("endpoint shapes disagree")
    if left.constant.dtype != right.constant.dtype:
        raise ValueError("endpoint dtypes disagree")
    card, dim = left.linear.shape
    linear = jnp.zeros((card, card, 2 * dim), dtype=left.linear.dtype)
    matrix = jnp.zeros((card, card, 2 * dim, 2 * dim), dtype=left.linear.dtype)
    linear = linear.at[..., :dim].set(left.linear[:, None]).at[..., dim:].set(right.linear[None])
    matrix = matrix.at[..., :dim, :dim].set(left.precision[:, None])
    matrix = matrix.at[..., dim:, dim:].set(right.precision[None])
    correction = None
    if left.precision_correction is not None or right.precision_correction is not None:
        correction = jnp.zeros_like(matrix).at[..., :dim, :dim].set(left.correction[:, None])
        correction = correction.at[..., dim:, dim:].set(right.correction[None])
    center = jnp.concatenate((jnp.broadcast_to(left.origin[:, None], (card, card, dim)),
                              jnp.broadcast_to(right.origin[None], (card, card, dim))), axis=-1)
    constant = left.constant[:, None] + right.constant[None]
    overflow = jnp.isfinite(left.constant[:, None]) & jnp.isfinite(right.constant[None]) & ~jnp.isfinite(constant)
    return Natural(jnp.where(overflow, jnp.nan, constant), linear, matrix,
                   coordinate_status(left) | coordinate_status(right)
                   | failure(jnp.any(overflow), Status.NONFINITE), center, correction)


def gaussian_adapter(hybrid_cavity, gaussian_cavity):
    """D12.5 equality region: project the full tilt before dividing the cavity.

    The reduced hybrid belief owns minus its labeled entropy and zero factor
    energy. No entropy of a continuous Dirac density is evaluated.
    """
    check_shapes(gaussian_cavity)
    if gaussian_cavity.constant.shape != () or hybrid_cavity.constant.ndim != 1:
        raise ValueError("adapter requires hybrid labels and one ordinary Gaussian")
    if gaussian_cavity.linear.shape != hybrid_cavity.linear.shape[1:]:
        raise ValueError("Gaussian adapter dimensions disagree")
    embedded = Natural(*(jnp.broadcast_to(x, y.shape) for x, y in
                         zip(gaussian_cavity.arrays(), hybrid_cavity.arrays())), gaussian_cavity.status,
                       None if gaussian_cavity.center is None else
                       jnp.broadcast_to(gaussian_cavity.center, hybrid_cavity.linear.shape),
                       None if gaussian_cavity.precision_correction is None else
                       jnp.broadcast_to(gaussian_cavity.precision_correction, hybrid_cavity.precision.shape))
    belief = normalize(hybrid_cavity + embedded)
    center = belief.center[0] if gaussian_cavity.center is None else gaussian_cavity.center
    projected = _mix(belief, 0, center)
    return belief, quotient(belief, hybrid_cavity), quotient(projected, gaussian_cavity)


def categorical_adapter(hybrid_cavity, log_cavity):
    """D12.5 equality region to a categorical copy, preserving association."""
    if (hybrid_cavity.constant.ndim != 1 or log_cavity.shape != hybrid_cavity.constant.shape
            or log_cavity.dtype != hybrid_cavity.constant.dtype):
        raise ValueError("categorical adapter shapes or dtypes disagree")
    embedded = Natural(log_cavity, jnp.zeros_like(hybrid_cavity.linear),
                       jnp.zeros_like(hybrid_cavity.precision))
    belief = normalize(hybrid_cavity + embedded)
    active = belief.probability > 0
    outgoing = jnp.where(active, belief.log_probability - jnp.where(active, log_cavity, 0), -jnp.inf)
    outgoing -= jax.scipy.special.logsumexp(outgoing)
    outgoing = jnp.where(belief.status == 0, outgoing, jnp.nan)
    return belief, quotient(belief, hybrid_cavity), outgoing

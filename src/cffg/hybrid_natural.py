"""D12 hybrid natural coordinates and moment projections, without engine binding.

One label block represents ``exp(a + h @ x - x @ J @ x / 2)``. Only
a common shift of ``a`` is gauge. Messages may have signed precision; the
aggregate or tilted belief must be proper. Numerical failures return a bitmask
in both eager and transformed execution; ``raise_for_status`` is the host gate.
"""

from enum import IntFlag
from typing import NamedTuple

import jax
import jax.numpy as jnp


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

    def __add__(self, other):
        check_shapes(self)
        check_shapes(other)
        if any(a.shape != b.shape or a.dtype != b.dtype
               for a, b in zip(self.arrays(), other.arrays())):
            raise ValueError("natural coordinate shapes or dtypes disagree")
        return Natural(*(a + b for a, b in zip(self.arrays(), other.arrays())),
                       self.status | other.status)

    def arrays(self):
        return self.constant, self.linear, self.precision


class Belief(NamedTuple):
    probability: jax.Array
    mean: jax.Array
    covariance: jax.Array
    log_probability: jax.Array
    log_normalizer: jax.Array
    status: jax.Array

    @property
    def second(self):
        return self.covariance + self.mean[..., :, None] * self.mean[..., None, :]

    def entropy(self):
        """Labeled entropy, including the Gaussian constants for every label."""
        active = self.probability > 0
        safe = jnp.where(active[..., None, None], self.covariance,
                         jnp.eye(self.mean.shape[-1], dtype=self.mean.dtype))
        gaussian = (self.mean.shape[-1] * (1 + jnp.log(2 * jnp.pi))
                    + jnp.linalg.slogdet(safe)[1]) / 2
        value = jnp.sum(jnp.where(active, self.probability *
                                 (gaussian - jnp.where(active, self.log_probability, 0)), 0))
        return jnp.where(self.status == 0, value, jnp.nan)

    def expectation(self, natural):
        check_shapes(natural)
        if natural.linear.shape != self.mean.shape:
            raise ValueError("belief and potential shapes disagree")
        active = self.probability > 0
        value = jnp.where(active, natural.constant, 0)
        value += jnp.einsum("...d,...d->...", natural.linear, self.mean)
        value -= jnp.einsum("...ab,...ab->...", natural.precision, self.second) / 2
        value = jnp.sum(jnp.where(active, self.probability * value, 0))
        return jnp.where((self.status | coordinate_status(natural)) == 0, value, jnp.nan)


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


def coordinate_status(natural):
    """Finite or literal log-zero constants; finite symmetric Gaussian blocks."""
    check_shapes(natural)
    a, h, matrix = natural.arrays()
    finite = jnp.all(jnp.isfinite(a) | jnp.isneginf(a))
    finite &= jnp.all(jnp.isfinite(h)) & jnp.all(jnp.isfinite(matrix))
    symmetric = jnp.all(matrix == matrix.swapaxes(-1, -2))
    return (natural.status | failure(~finite, Status.NONFINITE)
            | failure(~symmetric, Status.INVALID)
            | failure(~jnp.any(jnp.isfinite(a)), Status.EMPTY_SUPPORT))


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


def normalize(natural):
    """Exact Gaussian integration on a fixed label face, D12.2.

    Positive-label underflow is a failure and retains the finite log mass;
    it is never reinterpreted as a structural zero. Inactive coordinates are
    zero. An invalid active belief is NaN with a nonzero status.
    """
    status = coordinate_status(natural)
    a, h, matrix = natural.arrays()
    active = jnp.isfinite(a)
    covariance, logdet, pd_status = gaussian_blocks(matrix, active)
    mean = jnp.einsum("...ab,...b->...a", covariance, jnp.where(active[..., None], h, 0))
    mass = (h.shape[-1] * jnp.log(2 * jnp.pi) - logdet
            + jnp.einsum("...d,...d->...", h, mean)) / 2
    scores = jnp.where(active, a + mass, -jnp.inf)
    normalizer = jax.scipy.special.logsumexp(scores)
    log_probability = jnp.where(active, scores - normalizer, -jnp.inf)
    probability = jnp.where(active, jnp.exp(log_probability), 0)
    status |= pd_status | failure(~jnp.isfinite(normalizer), Status.NONFINITE)
    status |= failure(jnp.any(active & (probability == 0)), Status.UNDERFLOW)
    status |= failure(~jnp.all(jnp.isfinite(mean)) | ~jnp.all(jnp.isfinite(covariance)), Status.NONFINITE)
    second = covariance + mean[..., :, None] * mean[..., None, :]
    status |= failure(jnp.any(active[..., None, None] & ~jnp.isfinite(second)), Status.NONFINITE)
    return Belief(jnp.where(active, jnp.where(status == 0, probability, jnp.nan), 0),
                  jnp.where(active[..., None], jnp.where(status == 0, mean, jnp.nan), 0),
                  jnp.where(active[..., None, None], jnp.where(status == 0, covariance, jnp.nan), 0),
                  log_probability, jnp.where(status == 0, normalizer, jnp.nan), status)


def from_moments(probability, mean, covariance):
    """Construct a checked normalized moment belief for a projection or caller."""
    check_shapes(Natural(probability, mean, covariance))
    active = probability > 0
    _, _, pd_status = gaussian_blocks(covariance, active)
    finite = (jnp.all(jnp.isfinite(probability)) & jnp.all(jnp.isfinite(mean))
              & jnp.all(jnp.isfinite(covariance)))
    second = covariance + mean[..., :, None] * mean[..., None, :]
    finite &= jnp.all(jnp.isfinite(second))
    # Sum roundoff scales with the number of label additions and the dtype.
    tolerance = 32 * probability.size * jnp.finfo(probability.dtype).eps
    valid = jnp.all(probability >= 0) & (abs(probability.sum() - 1) <= tolerance)
    valid &= jnp.all(covariance == covariance.swapaxes(-1, -2))
    status = (pd_status | failure(~finite, Status.NONFINITE)
              | failure(~valid, Status.INVALID) | failure(~jnp.any(active), Status.EMPTY_SUPPORT))
    log_probability = jnp.where(active, jnp.log(jnp.where(active, probability, 1)), -jnp.inf)
    return Belief(jnp.where(status == 0, probability, jnp.nan),
                  jnp.where(active[..., None], jnp.where(status == 0, mean, jnp.nan), 0),
                  jnp.where(active[..., None, None], jnp.where(status == 0, covariance, jnp.nan), 0),
                  log_probability, jnp.zeros((), probability.dtype), status)


def canonical(belief):
    """Normalized hybrid density, retaining label-dependent scales, D12.6."""
    p, mean, covariance = belief.probability, belief.mean, belief.covariance
    active = p > 0
    precision, logdet, pd_status = gaussian_blocks(covariance, active)
    linear = jnp.einsum("...ab,...b->...a", precision, mean)
    constant = belief.log_probability - (logdet + mean.shape[-1] * jnp.log(2 * jnp.pi)
                                         + jnp.einsum("...d,...d->...", mean, linear)) / 2
    status = belief.status | pd_status
    status |= failure(jnp.any(active & ~jnp.isfinite(constant))
                      | ~jnp.all(jnp.isfinite(linear)), Status.NONFINITE)
    return Natural(jnp.where(active, jnp.where(status == 0, constant, jnp.nan), -jnp.inf),
                   jnp.where(active[..., None], jnp.where(status == 0, linear, jnp.nan), 0),
                   jnp.where(active[..., None, None], jnp.where(status == 0, precision, jnp.nan), 0), status)


def quotient(belief, cavity):
    """D12.7 full natural quotient, with one common constant fixed to zero.

    A tilt can restrict a cavity face. It cannot revive a cavity's absent
    label; that requires rebuilding the active inventory. Signed precision
    is preserved and inactive Gaussian coordinates remain zero.
    """
    projected = canonical(belief)
    check_shapes(cavity)
    if any(a.shape != b.shape or a.dtype != b.dtype
           for a, b in zip(projected.arrays(), cavity.arrays())):
        raise ValueError("quotient shapes or dtypes disagree")
    active = belief.probability > 0
    status = projected.status | coordinate_status(cavity)
    status |= failure(jnp.any(active & ~jnp.isfinite(cavity.constant)), Status.SUPPORT_CHANGED)
    values = [a - jnp.where(jnp.isfinite(b), b, 0)
              for a, b in zip(projected.arrays(), cavity.arrays())]
    first = jnp.argmax(active.reshape(-1))
    values[0] -= values[0].reshape(-1)[first]
    outputs = []
    for i, value in enumerate(values):
        mask = active.reshape(active.shape + (1,) * i)
        status |= failure(jnp.any(mask & ~jnp.isfinite(value)), Status.NONFINITE)
        outputs.append(jnp.where(mask, value, -jnp.inf if i == 0 else 0))
    return Natural(*(jnp.where(status == 0, x, jnp.nan) for x in outputs), status)


def collapse_label(region, axis):
    """Mix one label axis, retaining between-mean conditional covariance."""
    if axis not in (0, 1) or region.probability.ndim != 2:
        raise ValueError("collapse requires a pair of label axes")
    probability = region.probability.sum(axis=axis)
    denominator = jnp.where(probability > 0, probability, 1)
    conditional = region.probability / jnp.expand_dims(denominator, axis)
    mean = (conditional[..., None] * region.mean).sum(axis=axis)
    centered = region.mean - jnp.expand_dims(mean, axis)
    second = region.covariance + centered[..., :, None] * centered[..., None, :]
    covariance = (conditional[..., None, None] * second).sum(axis=axis)
    result = from_moments(probability, mean, covariance)
    return result._replace(status=result.status | region.status)


def marginal(region, side):
    """Projection onto source (0) or destination (1) hybrid moments."""
    if side not in (0, 1) or region.mean.shape[-1] % 2:
        raise ValueError("grouped marginal needs two equal continuous blocks")
    dim = region.mean.shape[-1] // 2
    selection = slice(side * dim, (side + 1) * dim)
    reduced = region._replace(mean=region.mean[..., selection],
                              covariance=region.covariance[..., selection, selection])
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
    return Natural(left.constant[:, None] + right.constant[None], linear, matrix,
                   coordinate_status(left) | coordinate_status(right))


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
                         zip(gaussian_cavity.arrays(), hybrid_cavity.arrays())), gaussian_cavity.status)
    belief = normalize(hybrid_cavity + embedded)
    p, mean, covariance = belief.probability, belief.mean, belief.covariance
    ordinary_mean = jnp.einsum("k,kd->d", p, mean)
    centered = mean - ordinary_mean
    ordinary_covariance = jnp.einsum("k,kab->ab", p, covariance)
    ordinary_covariance += jnp.einsum("k,ka,kb->ab", p, centered, centered)
    projected = from_moments(jnp.ones((), p.dtype), ordinary_mean, ordinary_covariance)
    projected = projected._replace(status=projected.status | belief.status)
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

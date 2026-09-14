"""Local JAX kernels for the D12 ``hybrid-regime-ec-v1`` approximation.

Grouped destination-selected affine dynamics implements the alternating
R/H/D scopes by D12.3's exact elimination identity. It preserves declared
moments, messages and energy, not the full temporal mixture. All auxiliary
coordinates here are ``xi=c²``; no derivative in c certifies stationarity.
These kernels have no compiler, iteration policy, or persisted state binding.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from cffg.hybrid_natural import (
    Belief, Natural, Status, coordinate_status, failure, gaussian_blocks,
    lift, marginal, normalize, quotient,
)


def _recurrence_shapes(weights, offsets, xi):
    if weights.ndim != 3:
        raise ValueError("weights require [K,K-1,d]")
    card, sticks, dim = weights.shape
    if card < 2 or sticks != card - 1 or dim < 1:
        raise ValueError("weights require K>=2, K-1 sticks and d>=1")
    if offsets.shape != (card, sticks) or xi.shape != offsets.shape:
        raise ValueError("recurrence coordinate shapes disagree")
    if weights.dtype not in (jnp.dtype("float32"), jnp.dtype("float64")):
        raise ValueError("coordinates must use float32 or float64")
    if offsets.dtype != weights.dtype or xi.dtype != weights.dtype:
        raise ValueError("recurrence coordinate dtypes disagree")
    return card, dim


def pg_mean_xi(xi):
    """PG(1,sqrt(xi)) mean, with the analytic xi derivative -1/48 at zero."""
    small = xi <= jnp.sqrt(jnp.finfo(xi.dtype).eps)
    safe = jnp.sqrt(jnp.where(small, 1, xi))
    series_xi = jnp.where(small, xi, 0)
    series = 0.25 - series_xi / 48 + series_xi * series_xi / 480
    return jnp.where(small, series, jnp.tanh(safe / 2) / (2 * safe))


def recurrence_potential(weights, offsets, xi):
    """D11.5 polynomial using O(K² d²) prefix storage, no K³ stick tensor."""
    card, dim = _recurrence_shapes(weights, offsets, xi)
    finite = jnp.all(jnp.isfinite(weights)) & jnp.all(jnp.isfinite(offsets)) & jnp.all(jnp.isfinite(xi))
    status = failure(~finite, Status.NONFINITE) | failure(jnp.any(xi < 0), Status.INVALID)
    omega = pg_mean_xi(xi)
    small = xi <= jnp.sqrt(jnp.finfo(xi.dtype).eps)
    safe = jnp.sqrt(jnp.where(small, 1, xi))
    series_xi = jnp.where(small, xi, 0)
    log_cosh = jnp.where(small, jnp.log(jnp.asarray(2, xi.dtype)) + series_xi / 8
                        - series_xi * series_xi / 192 + series_xi ** 3 / 2880,
                        jnp.logaddexp(safe / 2, -safe / 2))
    rejected_a = -log_cosh + omega * (xi - offsets * offsets) / 2 - offsets / 2
    rejected_h = -(0.5 + omega * offsets)[..., None] * weights
    rejected_j = omega[..., None, None] * weights[..., :, None] * weights[..., None, :]

    def prefix_with_final(value):
        prefix = jnp.cumsum(value, axis=1)
        return jnp.concatenate((prefix, prefix[:, -1:]), axis=1)

    constant = prefix_with_final(rejected_a).at[:, :card - 1].add(offsets)
    linear = prefix_with_final(rejected_h).at[:, :card - 1].add(weights)
    precision = prefix_with_final(rejected_j)
    precision = (precision + precision.swapaxes(-1, -2)) / 2
    result = Natural(constant, linear, precision, status)
    return result._replace(status=coordinate_status(result))


def transition(weights, offsets, xi, matrix, bias, noise, *, channel="destination"):
    """D12.2 recurrence times normalized N(y; A_j x+b_j,Q_j).

    ``channel`` explicitly names the indexing convention. Source/pair-selected
    channels lack D12.3's elimination identity and are rejected, even when
    their raw shapes could match. No Gaussian normalization constant is dropped.
    """
    if channel != "destination":
        raise ValueError("D12 requires the destination-selected affine channel")
    card, dim = _recurrence_shapes(weights, offsets, xi)
    if matrix.shape != (card, dim, dim) or bias.shape != (card, dim) or noise.shape != matrix.shape:
        raise ValueError("destination channel shapes disagree")
    if any(value.dtype != weights.dtype for value in (matrix, bias, noise)):
        raise ValueError("destination channel dtypes disagree")
    recurrence = recurrence_potential(weights, offsets, xi)
    finite = jnp.all(jnp.isfinite(matrix)) & jnp.all(jnp.isfinite(bias)) & jnp.all(jnp.isfinite(noise))
    symmetric = jnp.all(noise == noise.swapaxes(-1, -2))
    inverse, logdet, pd_status = gaussian_blocks(noise, jnp.ones(card, dtype=bool))
    status = recurrence.status | pd_status | failure(~finite, Status.NONFINITE) | failure(~symmetric, Status.INVALID)
    wa = jnp.einsum("jab,jbc->jac", inverse, matrix)
    wb = jnp.einsum("jab,jb->ja", inverse, bias)
    atwa = jnp.einsum("jba,jbc->jac", matrix, wa)
    atwb = jnp.einsum("jba,jb->ja", matrix, wb)
    constant = recurrence.constant - (dim * jnp.log(2 * jnp.pi) + logdet
                                      + jnp.einsum("ja,ja->j", bias, wb))[None] / 2
    linear = jnp.zeros((card, card, 2 * dim), weights.dtype)
    linear = linear.at[..., :dim].set(recurrence.linear - atwb[None]).at[..., dim:].set(wb[None])
    precision = jnp.zeros((card, card, 2 * dim, 2 * dim), weights.dtype)
    precision = precision.at[..., :dim, :dim].set(recurrence.precision + atwa[None])
    precision = precision.at[..., :dim, dim:].set(-wa.swapaxes(-1, -2)[None])
    precision = precision.at[..., dim:, :dim].set(-wa[None])
    precision = precision.at[..., dim:, dim:].set(inverse[None])
    precision = (precision + precision.swapaxes(-1, -2)) / 2
    result = Natural(constant, linear, precision, status)
    return result._replace(status=coordinate_status(result))


def source_region(region):
    """Recurrence marginal (i,j,x); retains pair probabilities and source moments."""
    if region.mean.shape[-1] % 2:
        raise ValueError("grouped region needs equal source/destination dimensions")
    dim = region.mean.shape[-1] // 2
    return region._replace(mean=region.mean[..., :dim], covariance=region.covariance[..., :dim, :dim])


class Auxiliary(NamedTuple):
    target: jax.Array
    residual: jax.Array
    exposure: jax.Array
    linear: jax.Array
    precision: jax.Array
    status: jax.Array


def auxiliary_statistics(region, weights, offsets, xi):
    """D12.8 xi residual and D11.1 learning from the same pair belief.

    Reverse prefix moments sum destinations at risk without a K³ tensor.
    Zero exposure fixes xi to zero and produces exact zero learning. A stale
    nonzero inactive xi has a nonzero residual until rebuilt. The residual is
    conditional on exposure, so a rare active source cannot hide a bad xi.
    """
    card, dim = _recurrence_shapes(weights, offsets, xi)
    source = source_region(region)
    if source.mean.shape != (card, card, dim) or source.mean.dtype != weights.dtype:
        raise ValueError("auxiliary belief shapes or dtypes disagree")
    p, mean, second = source.probability, source.mean, source.second
    augmented = jnp.concatenate((mean, jnp.ones((*p.shape, 1), p.dtype)), axis=-1)
    moments = jnp.zeros((*p.shape, dim + 1, dim + 1), p.dtype)
    moments = moments.at[..., :dim, :dim].set(second)
    moments = moments.at[..., :dim, dim].set(mean).at[..., dim, :dim].set(mean)
    moments = moments.at[..., dim, dim].set(1)

    def at_risk(value):
        return jnp.cumsum(value[:, ::-1], axis=1)[:, ::-1][:, :-1]

    first = p[..., None] * augmented
    exposure = at_risk(p)
    accumulated = at_risk(p[..., None, None] * moments)
    active = exposure > 0
    conditional = accumulated / jnp.where(active, exposure, 1)[..., None, None]
    theta = jnp.concatenate((weights, offsets[..., None]), axis=-1)
    raw_target = jnp.einsum("ila,ilab,ilb->il", theta, conditional, theta)
    scale = jnp.einsum("ila,ilab,ilb->il", abs(theta), abs(conditional), abs(theta))
    # Only dtype-sized contraction roundoff may make a squared logit negative.
    tolerance = 32 * (dim + 1) * jnp.finfo(p.dtype).eps * jnp.maximum(scale, 1)
    valid = jnp.all(raw_target >= -tolerance) & jnp.all(xi >= 0)
    target = jnp.where(active, jnp.maximum(raw_target, 0), 0)
    residual = xi - target
    linear = first[:, :-1] - at_risk(first) / 2
    precision = pg_mean_xi(xi)[..., None, None] * accumulated
    finite = jnp.all(jnp.isfinite(weights)) & jnp.all(jnp.isfinite(offsets)) & jnp.all(jnp.isfinite(xi))
    finite &= jnp.all(jnp.isfinite(residual)) & jnp.all(jnp.isfinite(linear)) & jnp.all(jnp.isfinite(precision))
    status = region.status | failure(~valid, Status.INVALID) | failure(~finite, Status.NONFINITE)
    return Auxiliary(*(jnp.where(status == 0, x, jnp.nan) for x in
                       (target, residual, exposure, linear, precision)), status)


def moment_residual(projected, separator):
    """Scaled label and conditional-moment residual, including tiny active labels.

    Conditional means and covariances are compared without multiplying by
    probability. Coordinate scaling is relative to max(1, both magnitudes).
    Support mismatch requires rebuilding the active inventory.
    """
    if projected.mean.shape != separator.mean.shape:
        raise ValueError("incidence moment shapes disagree")
    active = projected.probability > 0
    status = projected.status | separator.status
    status |= failure(jnp.any(active != (separator.probability > 0)), Status.SUPPORT_CHANGED)
    value = jnp.max(abs(projected.probability - separator.probability))
    for rank, a, b in ((1, projected.mean, separator.mean),
                       (2, projected.covariance, separator.covariance)):
        scale = jnp.maximum(1, jnp.maximum(abs(a), abs(b)))
        mask = active.reshape(active.shape + (1,) * rank)
        value = jnp.maximum(value, jnp.max(jnp.where(mask, abs(a - b) / scale, 0)))
    status |= failure(~jnp.isfinite(value), Status.NONFINITE)
    return jnp.where(status == 0, value, jnp.nan), status


def natural_residual(message, target):
    """Relative full quotient residual on a fixed supported, gauged face."""
    if message.linear.shape != target.linear.shape:
        raise ValueError("message residual shapes disagree")
    active = jnp.isfinite(message.constant)
    status = coordinate_status(message) | coordinate_status(target)
    status |= failure(jnp.any(active != jnp.isfinite(target.constant)), Status.SUPPORT_CHANGED)
    value = jnp.zeros((), message.constant.dtype)
    for rank, (a, b) in enumerate(zip(message.arrays(), target.arrays())):
        mask = active.reshape(active.shape + (1,) * rank)
        a, b = jnp.where(mask, a, 0), jnp.where(mask, b, 0)
        scale = jnp.maximum(1, jnp.maximum(abs(a), abs(b)))
        value = jnp.maximum(value, jnp.max(abs(a - b) / scale))
    status |= failure(~jnp.isfinite(value), Status.NONFINITE)
    return jnp.where(status == 0, value, jnp.nan), status


class Evaluation(NamedTuple):
    regions: Belief
    separators: Belief
    targets: tuple[Natural, Natural]
    free_energy: jax.Array
    dual_energy: jax.Array
    moment_residual: jax.Array
    quotient_residual: jax.Array
    status: jax.Array


def _combine_status(value):
    return jnp.bitwise_or.reduce(jnp.asarray(value, jnp.int32).reshape(-1))


def evaluate(factors, sites, messages):
    """Pure batched fixed-xi chain evaluation; no iterative policy or Jacobian.

    Inputs are stacked Naturals: factors [T-1,K,K,2d], sites [T,K,d], and
    the two directed message banks [T-1,K,d]. Status has the same leading
    stack length. D12.4 energy is a feasible objective only at consistency;
    D12.9 dual and incidence residuals remain separately named diagnostics.
    Work is O(T K² d³), storage O(T K² d²); only local covariances exist.
    """
    left, right = messages
    count = factors.constant.shape[0]
    if count < 1 or sites.constant.shape[0] != count + 1:
        raise ValueError("chain lengths disagree")
    if left.constant.shape != sites.constant[:-1].shape or right.constant.shape != left.constant.shape:
        raise ValueError("message bank shapes disagree")

    def shifted(site, outgoing, at_left):
        edge = 0 if at_left else -1
        arrays = []
        for s, m in zip(site.arrays(), outgoing.arrays()):
            padding = jnp.zeros_like(s[edge:edge + 1] if at_left else s[-1:])
            embedded = jnp.concatenate((padding, m) if at_left else (m, padding), axis=0)
            arrays.append(s + embedded)
        padding = jnp.zeros((1,), jnp.int32)
        status = jnp.concatenate((padding, outgoing.status) if at_left else (outgoing.status, padding))
        return Natural(*arrays, site.status | status)

    from_left = shifted(sites, right, True)
    from_right = shifted(sites, left, False)
    totals = shifted(from_left, left, False)
    separators = jax.vmap(normalize)(totals)
    left_cavity = jax.tree.map(lambda x: x[:-1], from_left)
    right_cavity = jax.tree.map(lambda x: x[1:], from_right)
    tilted = factors + jax.vmap(lift)(left_cavity, right_cavity)
    regions = jax.vmap(normalize)(tilted)
    projections = tuple(jax.vmap(lambda b: marginal(b, side))(regions) for side in (0, 1))
    targets = tuple(jax.vmap(quotient)(b, c) for b, c in zip(projections, (left_cavity, right_cavity)))
    regional_energy = -jax.vmap(Belief.entropy)(regions) - jax.vmap(Belief.expectation)(regions, factors)
    separator_entropy = jax.vmap(Belief.entropy)(separators)
    site_energy = jax.vmap(Belief.expectation)(separators, sites)
    free_energy = regional_energy.sum() + separator_entropy[1:-1].sum() - site_energy.sum()
    dual_energy = -regions.log_normalizer.sum() + separators.log_normalizer[1:-1].sum()
    moment_values, natural_values, statuses = [], [], [regions.status, separators.status]
    for side, projection, message, target in zip((0, 1), projections, messages, targets):
        incident = jax.tree.map(lambda x: x[side:side + count], separators)
        value, status = jax.vmap(moment_residual)(projection, incident)
        moment_values.append(value)
        statuses.append(status)
        value, status = jax.vmap(natural_residual)(message, target)
        natural_values.append(value)
        statuses.append(status)
    moment_value = jnp.max(jnp.stack(moment_values))
    natural_value = jnp.max(jnp.stack(natural_values))
    status = _combine_status(jnp.concatenate(statuses))
    finite = jnp.all(jnp.isfinite(jnp.stack((free_energy, dual_energy, moment_value, natural_value))))
    status |= failure(~finite, Status.NONFINITE)
    return Evaluation(regions, separators, targets,
                      *(jnp.where(status == 0, x, jnp.nan) for x in
                        (free_energy, dual_energy, moment_value, natural_value)), status)

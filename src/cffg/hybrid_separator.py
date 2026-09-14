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

from cffg.hybrid_affine import affine, centered_affine
from cffg.hybrid_natural import (
    Belief, Natural, Status, coordinate_status, failure, gaussian_blocks,
    lift, marginal, normalize, quotient, rebase,
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


def _jj_base_xi(xi):
    """JJ constant -log(2 cosh(c/2))+omega(c)c²/2 with c²=xi.

    Combining the series before differentiation preserves its zero xi slope
    at the origin; canceling two separate O(1) slopes would hide small variance.
    """
    small = xi <= jnp.sqrt(jnp.finfo(xi.dtype).eps)
    safe_xi = jnp.where(small, 1, xi)
    c = jnp.sqrt(safe_xi)
    series_xi = jnp.where(small, xi, 0)
    series = -jnp.log(jnp.asarray(2, xi.dtype)) - series_xi ** 2 / 192 + series_xi ** 3 / 1440
    direct = -jnp.logaddexp(c / 2, -c / 2) + pg_mean_xi(safe_xi) * safe_xi / 2
    return jnp.where(small, series, direct)


def recurrence_potential(weights, offsets, xi, *, center=None):
    """D11.5 polynomial using O(K² d²) prefix storage, no K³ stick tensor."""
    card, dim = _recurrence_shapes(weights, offsets, xi)
    origin = None
    if center is not None:
        if center.shape != (card, dim) or center.dtype != weights.dtype:
            raise ValueError("recurrence center shapes or dtypes disagree")
        offsets = affine(weights, center[:, None], offsets)
        origin = jnp.broadcast_to(center[:, None], (card, card, dim))
    finite = jnp.all(jnp.isfinite(weights)) & jnp.all(jnp.isfinite(offsets)) & jnp.all(jnp.isfinite(xi))
    status = failure(~finite, Status.NONFINITE) | failure(jnp.any(xi < 0), Status.INVALID)
    omega = pg_mean_xi(xi)
    rejected_a = _jj_base_xi(xi) - omega * offsets * offsets / 2 - offsets / 2
    rejected_h = -(0.5 + omega * offsets)[..., None] * weights
    rejected_j = omega[..., None, None] * weights[..., :, None] * weights[..., None, :]

    def prefix_with_final(value):
        prefix = jnp.cumsum(value, axis=1)
        return jnp.concatenate((prefix, prefix[:, -1:]), axis=1)

    constant = prefix_with_final(rejected_a).at[:, :card - 1].add(offsets)
    linear = prefix_with_final(rejected_h).at[:, :card - 1].add(weights)
    precision = prefix_with_final(rejected_j)
    precision = (precision + precision.swapaxes(-1, -2)) / 2
    status |= failure(~jnp.all(jnp.isfinite(constant)), Status.NONFINITE)
    result = Natural(jnp.where(jnp.isfinite(constant), constant, jnp.nan), linear, precision, status, origin)
    return result._replace(status=coordinate_status(result))


def transition(weights, offsets, xi, matrix, bias, noise, *, channel="destination",
               left_center=None, right_center=None):
    """D12.2 recurrence times normalized N(y; A_j x+b_j,Q_j).

    ``channel`` explicitly names the indexing convention. Source/pair-selected
    channels lack D12.3's elimination identity and are rejected, even when
    their raw shapes could match. No Gaussian normalization constant is dropped.
    Fixed endpoint origins translate offsets to o+w*c_left and channel bias
    to A_j*c_left_i+b_j-c_right_j before expansion. The translated bias is
    pair-dependent only because its coordinates differ, not its physical law.
    """
    if channel != "destination":
        raise ValueError("D12 requires the destination-selected affine channel")
    card, dim = _recurrence_shapes(weights, offsets, xi)
    if matrix.shape != (card, dim, dim) or bias.shape != (card, dim) or noise.shape != matrix.shape:
        raise ValueError("destination channel shapes disagree")
    if any(value.dtype != weights.dtype for value in (matrix, bias, noise)):
        raise ValueError("destination channel dtypes disagree")
    left_center = jnp.zeros((card, dim), weights.dtype) if left_center is None else left_center
    right_center = jnp.zeros_like(left_center) if right_center is None else right_center
    for center in (left_center, right_center):
        if center.shape != (card, dim) or center.dtype != weights.dtype:
            raise ValueError("transition center shapes or dtypes disagree")
    centered_bias = affine(matrix[None], left_center[:, None, None], bias[None], -right_center[None])
    recurrence = recurrence_potential(weights, offsets, xi, center=left_center)
    finite = jnp.all(jnp.isfinite(matrix)) & jnp.all(jnp.isfinite(bias)) & jnp.all(jnp.isfinite(noise))
    symmetric = jnp.all(noise == noise.swapaxes(-1, -2))
    inverse, logdet, pd_status = gaussian_blocks(noise, jnp.ones(card, dtype=bool))
    status = recurrence.status | pd_status | failure(~finite, Status.NONFINITE) | failure(~symmetric, Status.INVALID)
    status |= failure(~jnp.all(jnp.isfinite(left_center)) | ~jnp.all(jnp.isfinite(right_center)), Status.NONFINITE)
    wa = jnp.einsum("jab,jbc->jac", inverse, matrix)
    wb = jnp.einsum("jab,ijb->ija", inverse, centered_bias)
    atwa = jnp.einsum("jba,jbc->jac", matrix, wa)
    atwb = jnp.einsum("jba,ijb->ija", matrix, wb)
    constant = recurrence.constant - (dim * jnp.log(2 * jnp.pi) + logdet[None]
                                      + jnp.einsum("ija,ija->ij", centered_bias, wb)) / 2
    linear = jnp.zeros((card, card, 2 * dim), weights.dtype)
    linear = linear.at[..., :dim].set(recurrence.linear - atwb).at[..., dim:].set(wb)
    precision = jnp.zeros((card, card, 2 * dim, 2 * dim), weights.dtype)
    precision = precision.at[..., :dim, :dim].set(recurrence.precision + atwa[None])
    precision = precision.at[..., :dim, dim:].set(-wa.swapaxes(-1, -2)[None])
    precision = precision.at[..., dim:, :dim].set(-wa[None])
    precision = precision.at[..., dim:, dim:].set(inverse[None])
    precision = (precision + precision.swapaxes(-1, -2)) / 2
    center = jnp.concatenate((jnp.broadcast_to(left_center[:, None], (card, card, dim)),
                              jnp.broadcast_to(right_center[None], (card, card, dim))), axis=-1)
    status |= failure(~jnp.all(jnp.isfinite(constant)), Status.NONFINITE)
    result = Natural(jnp.where(jnp.isfinite(constant), constant, jnp.nan), linear, precision, status, center)
    return result._replace(status=coordinate_status(result))


class TransitionParameters(NamedTuple):
    """Original D12 parameters retained for stable energy and xi derivatives."""

    weights: jax.Array
    offsets: jax.Array
    xi: jax.Array
    matrix: jax.Array
    bias: jax.Array
    noise: jax.Array


def source_region(region):
    """Recurrence marginal (i,j,x); retains pair probabilities and source moments."""
    if region.displacement.shape[-1] % 2:
        raise ValueError("grouped region needs equal source/destination dimensions")
    dim = region.displacement.shape[-1] // 2
    return region._replace(displacement=region.displacement[..., :dim], center=region.center[..., :dim],
                           covariance=region.covariance[..., :dim, :dim])


class Auxiliary(NamedTuple):
    """Learning statistics use the retained feature [x-center_i;1].

    Its weights are [w;o+w*center_i]. Physical readouts apply the exact affine
    feature transform, with the usual rounding at large world coordinates.
    """
    target: jax.Array
    residual: jax.Array
    exposure: jax.Array
    linear: jax.Array
    precision: jax.Array
    status: jax.Array
    center: jax.Array

    @property
    def physical_linear(self):
        return self.linear.at[..., :-1].add(self.center[:, None] * self.linear[..., -1:])

    @property
    def physical_precision(self):
        dim = self.center.shape[-1]
        transform = jnp.broadcast_to(jnp.eye(dim + 1, dtype=self.center.dtype),
                                     (self.center.shape[0], dim + 1, dim + 1))
        transform = transform.at[..., :dim, dim].set(self.center)
        return jnp.einsum("iac,ilcd,ibd->ilab", transform, self.precision, transform)


def _conditional_at_risk(probability, mean, covariance):
    """Stable suffix mixture moments, with O(K² d²) work and storage.

    Merge centered covariances instead of subtracting two large raw second
    moments. Each source/stick suffix includes within-pair covariance and
    between-pair means, even when its affine logit mean cancels an offset.
    """
    initial = (jnp.zeros_like(probability[:, 0]), jnp.zeros_like(mean[:, 0]),
               jnp.zeros_like(covariance[:, 0]))

    def merge(previous, component):
        exposure, old_mean, old_covariance = previous
        p, m, s = component
        total = exposure + p
        denominator = jnp.where(total > 0, total, 1)
        old_fraction, new_fraction = exposure / denominator, p / denominator
        delta = m - old_mean
        mixed_mean = old_mean + new_fraction[..., None] * delta
        mixed_mean = jnp.where((exposure > 0)[..., None], mixed_mean, m)
        mixed_covariance = old_fraction[..., None, None] * old_covariance
        mixed_covariance += new_fraction[..., None, None] * s
        mixed_covariance += ((old_fraction * new_fraction)[..., None, None]
                             * delta[..., :, None] * delta[..., None, :])
        result = (total, mixed_mean, mixed_covariance)
        return result, result

    _, suffixes = jax.lax.scan(merge, initial,
                              tuple(x.swapaxes(0, 1) for x in (probability, mean, covariance)), reverse=True)
    return tuple(x.swapaxes(0, 1)[:, :-1] for x in suffixes)


def auxiliary_statistics(region, weights, offsets, xi):
    """D12.8 xi residual and D11.1 learning from the same pair belief.

    Reverse prefix moments sum destinations at risk without a K³ tensor.
    Zero exposure fixes xi to zero and produces exact zero learning. A stale
    nonzero inactive xi has a nonzero residual until rebuilt. The residual is
    conditional on exposure, so a rare active source cannot hide a bad xi.
    """
    card, dim = _recurrence_shapes(weights, offsets, xi)
    source = source_region(region)
    if source.displacement.shape != (card, card, dim) or source.displacement.dtype != weights.dtype:
        raise ValueError("auxiliary belief shapes or dtypes disagree")
    center = source.center[:, 0]
    p, mean = source.probability, source.relative_mean(center[:, None])
    second = source.covariance + mean[..., :, None] * mean[..., None, :]
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
    _, conditional_mean, conditional_covariance = _conditional_at_risk(p, mean, source.covariance)
    logit_mean = centered_affine(weights, conditional_mean, center[:, None], offsets)
    logit_variance = jnp.einsum("ila,ilab,ilb->il", weights, conditional_covariance, weights)
    scale = jnp.einsum("ila,ilab,ilb->il", abs(weights), abs(conditional_covariance), abs(weights))
    # Only dtype-sized covariance-contraction roundoff may make variance negative.
    tolerance = 32 * (dim + 1) * jnp.finfo(p.dtype).eps * jnp.maximum(scale, 1)
    valid = jnp.all(logit_variance >= -tolerance) & jnp.all(xi >= 0)
    target = jnp.where(active, jnp.maximum(logit_variance, 0) + logit_mean ** 2, 0)
    residual = xi - target
    linear = first[:, :-1] - at_risk(first) / 2
    precision = pg_mean_xi(xi)[..., None, None] * accumulated
    finite = jnp.all(jnp.isfinite(weights)) & jnp.all(jnp.isfinite(offsets)) & jnp.all(jnp.isfinite(xi))
    finite &= jnp.all(jnp.isfinite(residual)) & jnp.all(jnp.isfinite(linear)) & jnp.all(jnp.isfinite(precision))
    status = region.status | failure(~valid, Status.INVALID) | failure(~finite, Status.NONFINITE)
    return Auxiliary(*(jnp.where(status == 0, x, jnp.nan) for x in
                       (target, residual, exposure, linear, precision)), status, center)


def transition_expectation(region, parameters):
    """Expected log recurrence/channel from the same belief, without expansion.

    Recurrence uses Var[psi]+E[psi]², preserving variance when a large mean
    cancels an offset. The affine channel whitens its residual mean and its
    transformed covariance square root. Both are algebraically the original
    D12 factor energy, including the complete Gaussian normalizer.

    An already-expanded Natural cannot recover small terms lost in its raw
    coefficients/second moments. The chain evaluator retains these original
    parameters and uses this path as its owned regional expectation.
    """
    weights, offsets, xi, matrix, bias, noise = parameters
    dim = weights.shape[-1]
    factor = transition(*parameters, left_center=region.center[:, 0, :dim],
                        right_center=region.center[0, :, dim:])
    auxiliary = auxiliary_statistics(region, weights, offsets, xi)
    source = source_region(region)
    mean = source.relative_mean(auxiliary.center[:, None])
    _, risk_mean, _ = _conditional_at_risk(source.probability, mean, source.covariance)
    success_mean = centered_affine(weights, mean[:, :-1], auxiliary.center[:, None], offsets)
    risk_logit = centered_affine(weights, risk_mean, auxiliary.center[:, None], offsets)
    linear = source.probability[:, :-1] * success_mean - auxiliary.exposure * risk_logit / 2
    recurrence = jnp.sum(linear + auxiliary.exposure *
                         (_jj_base_xi(xi) - pg_mean_xi(xi) * auxiliary.target / 2))

    card, _, dim = weights.shape
    active = region.probability > 0
    mean_residual = centered_affine(-matrix[None], source.displacement[..., None, :],
                                    source.center[..., None, :], region.center[..., dim:],
                                    region.displacement[..., dim:], -bias[None])
    covariance = jnp.where(active[..., None, None], region.covariance, jnp.eye(2 * dim, dtype=weights.dtype))
    covariance_root = jnp.linalg.cholesky(covariance)
    transform = jnp.concatenate((-matrix, jnp.broadcast_to(jnp.eye(dim, dtype=weights.dtype), matrix.shape)), axis=-1)
    residual_root = jnp.einsum("jab,ijbc->ijac", transform, covariance_root)
    noise_root = jnp.linalg.cholesky(noise)
    logdet = 2 * jnp.log(jnp.diagonal(noise_root, axis1=-2, axis2=-1)).sum(axis=-1)
    noise_root = jnp.broadcast_to(noise_root[None], (card, card, dim, dim))
    whitened_mean = jax.scipy.linalg.solve_triangular(noise_root, mean_residual[..., None], lower=True)
    whitened_root = jax.scipy.linalg.solve_triangular(noise_root, residual_root, lower=True)
    quadratic = (whitened_mean ** 2).sum(axis=(-2, -1)) + (whitened_root ** 2).sum(axis=(-2, -1))
    dynamics = -jnp.sum(jnp.where(active, region.probability *
                                (dim * jnp.log(2 * jnp.pi) + logdet[None] + quadratic) / 2, 0))
    value = recurrence + dynamics
    status = factor.status | auxiliary.status | failure(~jnp.isfinite(value), Status.NONFINITE)
    return jnp.where(status == 0, value, jnp.nan), status


def moment_residual(projected, separator):
    """Scaled label and conditional-moment residual, including tiny active labels.

    Conditional means and covariances are compared without multiplying by
    probability. Coordinate scaling is relative to max(1, both magnitudes).
    Support mismatch requires rebuilding the active inventory.
    """
    if projected.displacement.shape != separator.displacement.shape:
        raise ValueError("incidence moment shapes disagree")
    active = projected.probability > 0
    status = projected.status | separator.status
    status |= failure(jnp.any(active != (separator.probability > 0)), Status.SUPPORT_CHANGED)
    value = jnp.max(abs(projected.probability - separator.probability))
    for rank, a, b in ((1, projected.relative_mean(separator.center), separator.displacement),
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
    center = target.center if message.center is None else message.center
    message, target = rebase(message, center), rebase(target, center)
    active = jnp.isfinite(message.constant)
    status = coordinate_status(message) | coordinate_status(target)
    status |= failure(jnp.any(active != jnp.isfinite(target.constant)), Status.SUPPORT_CHANGED)
    value = jnp.zeros((), message.constant.dtype)
    for rank, (a, b) in enumerate(zip(message.arrays(), target.arrays())):
        mask = active.reshape(active.shape + (1,) * rank)
        a, b = jnp.where(mask, a, 0), jnp.where(mask, b, 0)
        if rank == 0:
            first = jnp.argmax(active.reshape(-1))
            a, b = (jnp.where(mask, x - x.reshape(-1)[first], 0) for x in (a, b))
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


def evaluate(parameters, sites, messages):
    """Pure batched fixed-xi chain evaluation; no iterative policy or Jacobian.

    Inputs are stacked TransitionParameters, sites [T,K,d], and two directed
    Natural message banks [T-1,K,d]. Natural status has the same leading
    stack length. Keeping original parameters avoids cancellation in the owned
    transition expectation. D12.4 energy is feasible only at consistency;
    D12.9 dual and incidence residuals remain separately named diagnostics.
    Work is O(T K² d³), storage O(T K² d²); only local covariances exist.
    """
    if not isinstance(parameters, TransitionParameters):
        raise TypeError("D12 evaluation requires original TransitionParameters for stable energy")
    left, right = messages
    count = parameters.weights.shape[0]
    if count < 1 or sites.constant.shape[0] != count + 1:
        raise ValueError("chain lengths disagree")
    if left.constant.shape != sites.constant[:-1].shape or right.constant.shape != left.constant.shape:
        raise ValueError("message bank shapes disagree")
    factors = jax.vmap(lambda p, l, r: transition(*p, left_center=l, right_center=r))(
        parameters, sites.origin[:-1], sites.origin[1:])

    def shifted(site, outgoing, at_left):
        outgoing = rebase(outgoing, site.origin[1:] if at_left else site.origin[:-1])
        edge = 0 if at_left else -1
        arrays = []
        for s, m in zip(site.arrays(), outgoing.arrays()):
            padding = jnp.zeros_like(s[edge:edge + 1] if at_left else s[-1:])
            embedded = jnp.concatenate((padding, m) if at_left else (m, padding), axis=0)
            arrays.append(embedded)
        padding = jnp.zeros((1,), jnp.int32)
        status = jnp.concatenate((padding, outgoing.status) if at_left else (outgoing.status, padding))
        return site + Natural(*arrays, status, site.center)

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
    expectations, energy_status = jax.vmap(transition_expectation)(regions, parameters)
    regional_energy = -jax.vmap(Belief.entropy)(regions) - expectations
    separator_entropy = jax.vmap(Belief.entropy)(separators)
    site_energy = jax.vmap(Belief.expectation)(separators, sites)
    free_energy = regional_energy.sum() + separator_entropy[1:-1].sum() - site_energy.sum()
    dual_energy = -regions.log_normalizer.sum() + separators.log_normalizer[1:-1].sum()
    moment_values, natural_values, statuses = [], [], [regions.status, separators.status, energy_status]
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

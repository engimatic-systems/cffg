"""Unbound complete D12 coordinate and signed stationarity residual.

The host inventory fixes support and origins. Only active gauged message
coefficients and xi are unknowns; projected covariance corrections are derived.
This module does not select a model, schedule runtime work, or persist state.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from cffg.hybrid_affine import affine
from cffg.hybrid_natural import Natural, Status, coordinate_status, failure, lift, marginal, normalize, quotient
from cffg import hybrid_separator as hs


@dataclass(frozen=True, eq=False)
class Coordinate:
    """Host-built fixed active inventory; array maps are inspectable metadata.

    Banks are ordered side, transition, label. Constants omit one common
    gauge per incidence; precision stores the upper triangle once. Inactive
    constants reconstruct as -inf and other inactive coefficients as zero.
    Rebuild this inventory when support or the chosen origins change.
    """

    shape: tuple[int, int, int]
    support: jax.Array
    origin: jax.Array
    active: jax.Array
    constants: jax.Array
    first: jax.Array
    auxiliary: jax.Array

    @classmethod
    def from_sites(cls, sites):
        if sites.linear.ndim != 3:
            raise ValueError("sites require [T,K,d] linear coefficients")
        t, k, d = sites.linear.shape
        if t < 2 or k < 2 or d < 1:
            raise ValueError("a hybrid chain requires T>=2, K>=2, d>=1")
        if sites.constant.shape != (t, k) or sites.precision.shape != (t, k, d, d):
            raise ValueError("site coordinate shapes disagree")
        if sites.linear.dtype not in (jnp.dtype('float32'), jnp.dtype('float64')):
            raise ValueError("coordinates require float32 or float64")
        if any(x.dtype != sites.linear.dtype for x in sites.arrays()):
            raise ValueError("site coordinate dtypes disagree")
        if sites.origin.shape != sites.linear.shape or sites.origin.dtype != sites.linear.dtype:
            raise ValueError("site origin shapes or dtypes disagree")
        support = ~np.isneginf(np.asarray(sites.constant))
        incidence = np.stack((support[:-1], support[1:]))
        first = incidence.argmax(axis=-1)
        free = incidence.copy()
        np.put_along_axis(free, first[..., None], False, axis=-1)
        at_risk = np.flip(np.logical_or.accumulate(np.flip(support[1:], axis=1), axis=1), axis=1)[:, :-1]
        auxiliary = support[:-1, :, None] & at_risk[:, None, :]
        indices = lambda x: jnp.asarray(np.flatnonzero(x), dtype=jnp.int32)
        return cls((t, k, d), jnp.asarray(support), jnp.asarray(sites.origin),
                   indices(incidence), indices(free), jnp.asarray(first, jnp.int32), indices(auxiliary))

    @property
    def message_size(self):
        d = self.shape[-1]
        return self.constants.size + self.active.size * (d + d * (d + 1) // 2)

    @property
    def size(self):
        return self.message_size + self.auxiliary.size

    @property
    def residual_size(self):
        d = self.shape[-1]
        return self.size + self.active.size * (1 + d + d * (d + 1) // 2)

    @property
    def incidence_shape(self):
        t, k, _ = self.shape
        return (2, t - 1, k)

    def pack(self, messages, xi):
        """Gather valid message banks, preserving rejection before gauge removal.

        Support and origins must match the inventory and status must be zero.
        Nonzero precision corrections belong to assembled potentials, not the
        outgoing message coordinate; quotient messages have no correction word.
        Invalid numeric input produces NaN coordinates in host/JIT/vmap calls,
        so downstream evaluation cannot certify it. An empty coordinate already
        has invalid empty support. No metadata becomes an extra unknown.
        """
        t, k, d = self.shape
        shapes = ((t - 1, k), (t - 1, k, d), (t - 1, k, d, d))
        if len(messages) != 2 or any(value.shape != shape for message in messages
                                    for value, shape in zip(message.arrays(), shapes)):
            raise ValueError("message shapes disagree with inventory")
        if jnp.shape(xi) != (t - 1, k, k - 1):
            raise ValueError("auxiliary shape disagrees with inventory")
        if any(value.dtype != self.origin.dtype for message in messages for value in message.arrays()):
            raise ValueError("message dtypes disagree with inventory")
        if xi.dtype != self.origin.dtype:
            raise ValueError("auxiliary dtype disagrees with inventory")
        for message in messages:
            status = jnp.asarray(message.status)
            if status.shape not in ((), (t - 1,)) or not jnp.issubdtype(status.dtype, jnp.integer):
                raise ValueError("message status requires integer scalar or incidence flags")
        a, h, p = (jnp.stack([m.arrays()[rank] for m in messages]) for rank in range(3))
        anchor = jnp.take_along_axis(a, self.first[..., None], axis=-1)
        supported = jnp.stack((self.support[:-1], self.support[1:]))
        valid = jnp.all(jnp.isneginf(a) == ~supported)
        for side, message in enumerate(messages):
            valid &= jnp.all(coordinate_status(message) == 0)
            valid &= jnp.all(jnp.asarray(message.status) == 0)
            valid &= jnp.all(message.origin == self.origin[side:side + t - 1])
            valid &= jnp.all(message.correction == 0)
        a = jnp.where(supported, a - jnp.where(jnp.isfinite(anchor), anchor, 0), 0)
        row, col = jnp.triu_indices(d)
        vector = jnp.concatenate((a.reshape(-1)[self.constants],
                                  h.reshape(-1, d)[self.active].reshape(-1),
                                  p.reshape(-1, d, d)[self.active][:, row, col].reshape(-1),
                                  xi.reshape(-1)[self.auxiliary]))
        return jnp.where(valid, vector, jnp.nan)

    def unpack(self, vector):
        if vector.shape != (self.size,) or vector.dtype != self.origin.dtype:
            raise ValueError("packed coordinate shape or dtype disagrees with inventory")
        t, k, d = self.shape
        count = 2 * (t - 1) * k
        nc, na, ns = self.constants.size, self.active.size, d * (d + 1) // 2
        a = jnp.full((count,), -jnp.inf, vector.dtype).at[self.active].set(0)
        a = a.at[self.constants].set(vector[:nc]).reshape(self.incidence_shape)
        h = jnp.zeros((count, d), vector.dtype).at[self.active].set(vector[nc:nc + na * d].reshape(na, d))
        p = jnp.zeros((count, d, d), vector.dtype)
        triangle = vector[nc + na * d:self.message_size].reshape(na, ns)
        row, col = jnp.triu_indices(d)
        p = p.at[self.active[:, None], row[None], col[None]].set(triangle)
        p = p.at[self.active[:, None], col[None], row[None]].set(triangle)
        h, p = h.reshape((*self.incidence_shape, d)), p.reshape((*self.incidence_shape, d, d))
        status = jnp.zeros((t - 1,), jnp.int32)
        messages = tuple(Natural(a[side], h[side], p[side], status,
                                 self.origin[side:side + t - 1]) for side in range(2))
        xi = jnp.zeros(((t - 1) * k * (k - 1),), vector.dtype)
        return messages, xi.at[self.auxiliary].set(vector[self.message_size:]).reshape(t - 1, k, k - 1)

    def metadata(self):
        return dict(support=self.support, origin=self.origin, active=self.active,
                    constants=self.constants, first=self.first, auxiliary=self.auxiliary)


class Certificates(NamedTuple):
    natural: jax.Array
    label_mass: jax.Array
    conditional_mean: jax.Array
    conditional_covariance: jax.Array
    auxiliary: jax.Array
    primal_dual: jax.Array

    def maximum(self):
        return jnp.max(jnp.stack(self))


class Evaluation(NamedTuple):
    local: hs.Evaluation
    auxiliary: hs.Auxiliary
    residual: jax.Array
    scales: jax.Array
    certificates: Certificates
    status: jax.Array


def covariance_difference(left, right):
    """Subtract represented covariances before rounding away their low words."""
    return affine(jnp.ones_like(left.covariance)[..., None], left.covariance[..., None],
                  -right.covariance, left.correction, -right.correction)


def _maximum(value):
    return jnp.max(abs(value), initial=0)


def _scale(a, b):
    return jnp.maximum(1, jnp.maximum(abs(a), abs(b)))


def validate_context(coordinate, parameters, sites):
    """Static shape/dtype boundary; numeric validity remains array-valued."""
    t, k, d = coordinate.shape
    shapes = ((t - 1, k, k - 1, d), (t - 1, k, k - 1), (t - 1, k, k - 1),
              (t - 1, k, d, d), (t - 1, k, d), (t - 1, k, d, d))
    if not isinstance(parameters, hs.TransitionParameters):
        raise TypeError("original TransitionParameters are required")
    if any(x.shape != shape for x, shape in zip(parameters, shapes)):
        raise ValueError("transition parameter shapes disagree")
    if any(x.dtype != coordinate.origin.dtype for x in (*parameters, *sites.arrays(), sites.origin)):
        raise ValueError("context dtypes disagree with inventory")
    if (sites.constant.shape != (t, k) or sites.linear.shape != (t, k, d)
            or sites.precision.shape != (t, k, d, d) or sites.origin.shape != (t, k, d)):
        raise ValueError("context shapes disagree with inventory")
    if jnp.shape(sites.status) != (t,):
        raise ValueError("sites require one int32 status per time")


def context_status(coordinate, sites):
    same_support = jnp.all((~jnp.isneginf(sites.constant)) == coordinate.support)
    return (failure(~same_support, Status.SUPPORT_CHANGED)
            | failure(jnp.any(sites.origin != coordinate.origin), Status.INVALID))


def evaluate(coordinate, parameters, sites, vector):
    """Evaluate the complete rectangular residual at one message/xi coordinate.

    Scales are returned separately: freeze them across a linear subproblem and
    its trials. Certificates always use the current dynamic scales. Conditional
    moment errors are unweighted, even when an active label has tiny mass.
    """
    validate_context(coordinate, parameters, sites)
    sites = sites._replace(status=jnp.asarray(sites.status, jnp.int32))
    messages, xi = coordinate.unpack(vector)
    parameters = parameters._replace(xi=xi)
    local = hs.evaluate(parameters, sites, messages)
    auxiliary = jax.vmap(hs.auxiliary_statistics)(local.regions, parameters.weights, parameters.offsets, xi)
    target = coordinate.pack(local.targets, auxiliary.target)
    natural = target[:coordinate.message_size] - vector[:coordinate.message_size]
    natural_scale = _scale(target[:coordinate.message_size], vector[:coordinate.message_size])
    blocks, scales = [natural], [natural_scale]
    projections = tuple(jax.vmap(lambda r: marginal(r, side))(local.regions) for side in (0, 1))
    t, _, d = coordinate.shape
    row, col = jnp.triu_indices(d)
    mass, means, covariance, mean_scales, covariance_scales = [], [], [], [], []
    for side, projected in enumerate(projections):
        incident = jax.tree.map(lambda x: x[side:side + t - 1], local.separators)
        mass.append(projected.probability - incident.probability)
        pm = projected.relative_mean(incident.center)
        means.append(pm - incident.displacement)
        mean_scales.append(_scale(pm, incident.displacement))
        covariance.append(covariance_difference(projected, incident))
        covariance_scales.append(_scale(projected.full_covariance, incident.full_covariance))
    select = lambda values, tail: jnp.stack(values).reshape((-1, *tail))[coordinate.active].reshape(-1)
    mass = select(mass, ())
    means, mean_scales = select(means, (d,)), select(mean_scales, (d,))
    select_cov = lambda values: jnp.stack(values).reshape(-1, d, d)[coordinate.active][:, row, col].reshape(-1)
    covariance, covariance_scales = select_cov(covariance), select_cov(covariance_scales)
    aux = auxiliary.residual.reshape(-1)[coordinate.auxiliary]
    aux_scale = 1 + abs(vector[coordinate.message_size:])
    blocks += [mass, means, covariance, aux]
    scales += [jnp.ones_like(mass), mean_scales, covariance_scales, aux_scale]
    residual, scales = jnp.concatenate(blocks), jnp.concatenate(scales)
    certificates = Certificates(_maximum(natural / natural_scale), _maximum(mass),
                                _maximum(means / mean_scales), _maximum(covariance / covariance_scales),
                                _maximum(aux / aux_scale),
                                abs(local.free_energy - local.dual_energy) / (1 + abs(local.free_energy)))
    status = jnp.asarray(local.status, jnp.int32) | jnp.bitwise_or.reduce(auxiliary.status)
    status |= context_status(coordinate, sites)
    status |= failure(~jnp.all(jnp.isfinite(residual)) | ~jnp.all(jnp.isfinite(scales)), Status.NONFINITE)
    status |= failure(~jnp.isfinite(certificates.maximum()), Status.NONFINITE)
    return Evaluation(local, auxiliary, residual, scales, certificates, status)


def initialize(coordinate, parameters, sites, initial_xi=0., *, perturb=False):
    """One simultaneous outgoing quotient from unary cavities, then certify.

    Unary cavities may be vacuous or signed. Their joint local tilts must be
    proper; no positivity clamp is applied to a message. The returned numeric
    status diagnoses invalid initialization without throwing inside transforms.
    """
    validate_context(coordinate, parameters, sites)
    sites = sites._replace(status=jnp.asarray(sites.status, jnp.int32))
    vector = jnp.zeros((coordinate.size,), sites.linear.dtype)
    start = jnp.broadcast_to(jnp.asarray(initial_xi, vector.dtype), parameters.xi.shape)
    vector = vector.at[coordinate.message_size:].set(start.reshape(-1)[coordinate.auxiliary])
    messages, xi = coordinate.unpack(vector)
    parameters = parameters._replace(xi=xi)
    factors = jax.vmap(lambda p, l, r: hs.transition(*p, left_center=l, right_center=r))(
        parameters, coordinate.origin[:-1], coordinate.origin[1:])
    cavities = tuple(jax.tree.map(lambda x: x[side:side + coordinate.shape[0] - 1], sites)
                     for side in range(2))
    regions = jax.vmap(normalize)(factors + jax.vmap(lift)(*cavities))
    projections = tuple(jax.vmap(lambda r: marginal(r, side))(regions) for side in range(2))
    messages = tuple(jax.vmap(quotient)(p, c) for p, c in zip(projections, cavities))
    vector = coordinate.pack(messages, xi)
    magnitude = jnp.asarray(perturb, vector.dtype) * .01
    vector = vector.at[:coordinate.message_size].add(
        magnitude * jnp.sin(jnp.arange(coordinate.message_size, dtype=vector.dtype) + 1))
    value = evaluate(coordinate, parameters, sites, vector)
    status = value.status | jnp.bitwise_or.reduce(regions.status)
    status |= failure(jnp.any(start.reshape(-1)[coordinate.auxiliary] < 0), Status.INVALID)
    status |= failure(~jnp.all(jnp.isfinite(start.reshape(-1)[coordinate.auxiliary])), Status.NONFINITE)
    return vector, value._replace(status=status)


def linearize(coordinate, parameters, sites, vector, scales, coordinate_scales):
    """One accepted residual linearization and its transpose, with frozen scales.

    The N-dimensional tangent is a scaled coordinate step; the image is an
    M-dimensional residual. The transpose reuses the same nonlinear captures.
    Only call at a proper accepted point, never through trial rejection masks.
    """
    def residual(step):
        return evaluate(coordinate, parameters, sites, vector + coordinate_scales * step).residual / scales
    residual, forward = jax.linearize(residual, jnp.zeros_like(vector))
    transpose = jax.linear_transpose(forward, jnp.zeros_like(vector))
    return residual, forward, lambda value: transpose(value)[0]

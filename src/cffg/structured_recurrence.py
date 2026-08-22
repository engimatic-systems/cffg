"""Structured categorical authority for recurrent stick-breaking factors.

These pure functions implement D9 in ``docs/DERIVATIONS.md`` for one
time-local recurrence factor.  They are intentionally not dispatched by the
engine yet: the existing recurrence runtime retains its D8 mean-field rules
until the compiler and schedule integration lands.

The source regime indexes pair rows and the destination regime indexes pair
columns.  Incoming categorical cavities are log potentials.  Numerical
failures return ``proper=False`` with NaN-valued messages or statistics, while
energy functions return positive infinity.  This keeps the functions usable
under ``jit`` and ``vmap`` without letting invalid results look successful.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from cffg import gauss


class StructuredPairPotential(NamedTuple):
    """Bounded transition table and its eliminated PG coordinate."""

    log_potential: jax.Array
    omega: jax.Array
    proper: jax.Array


class StructuredPairBelief(NamedTuple):
    """Two factor messages and their normalized pair-region belief."""

    to_out: jax.Array
    to_in: jax.Array
    probability: jax.Array
    source_probability: jax.Array
    destination_probability: jax.Array
    proper: jax.Array


class StructuredGaussianMessage(NamedTuple):
    """Pair-derived Gaussian message and stick responsibility masses."""

    eta: jax.Array
    precision: jax.Array
    exposure: jax.Array
    signed_success: jax.Array
    proper: jax.Array


def _floating_dtype(*values) -> jnp.dtype:
    return jnp.result_type(*values, jnp.float32)


def _pg_mean(xi: jax.Array) -> jax.Array:
    """E[PG(1, xi)] with the analytic limit selected at dtype scale."""
    threshold = jnp.sqrt(jnp.finfo(xi.dtype).eps)
    safe_xi = jnp.where(xi > threshold, xi, jnp.ones_like(xi))
    direct = jnp.tanh(0.5 * xi) / (2.0 * safe_xi)
    square = jnp.square(xi)
    series = 0.25 - square / 48.0 + jnp.square(square) / 480.0
    return jnp.where(xi > threshold, direct, series)


def structured_pair_potential(
    drive_mean,
    drive_covariance,
    weights,
    offsets,
) -> StructuredPairPotential:
    """Return the D9 JJ expected-log table for one recurrence factor.

    Shapes are ``mean [D]``, ``covariance [D,D]``,
    ``weights [K,K-1,D]``, and ``offsets [K,K-1]``.  A zero covariance is
    valid and makes the JJ table equal the exact fixed-drive transition log
    probability.  Improper moments or non-finite inputs are loud through the
    returned ``proper`` flag and NaN outputs.
    """
    mean = jnp.asarray(drive_mean)
    covariance = jnp.asarray(drive_covariance)
    weights = jnp.asarray(weights)
    offsets = jnp.asarray(offsets)
    if mean.ndim != 1 or covariance.shape != (mean.shape[0], mean.shape[0]):
        raise ValueError(
            "drive moments require mean [D] and covariance [D,D]"
        )
    if weights.ndim != 3:
        raise ValueError(
            "recurrence parameters require weights [K,K-1,D] and "
            "offsets [K,K-1]"
        )
    card, sticks, dimension = weights.shape
    if (
        card < 2
        or sticks != card - 1
        or dimension != mean.shape[0]
        or offsets.shape != (card, sticks)
    ):
        raise ValueError(
            "recurrence parameters require weights [K,K-1,D] and "
            "offsets [K,K-1]"
        )

    dtype = _floating_dtype(mean, covariance, weights, offsets)
    mean = mean.astype(dtype)
    covariance = covariance.astype(dtype)
    weights = weights.astype(dtype)
    offsets = offsets.astype(dtype)
    parameter_ok = (
        jnp.all(jnp.isfinite(mean))
        & jnp.all(jnp.isfinite(weights))
        & jnp.all(jnp.isfinite(offsets))
    )
    covariance_finite = jnp.all(jnp.isfinite(covariance))
    safe_covariance = jnp.where(
        covariance_finite,
        covariance,
        jnp.eye(dimension, dtype=dtype),
    )
    covariance_ok = covariance_finite & gauss._psd_ok(safe_covariance)
    symmetric_covariance = 0.5 * (
        safe_covariance + safe_covariance.swapaxes(-1, -2)
    )

    logit_mean = jnp.einsum("ild,d->il", weights, mean) + offsets
    logit_variance = jnp.einsum(
        "ild,de,ile->il", weights, symmetric_covariance, weights
    )
    raw_second = jnp.square(logit_mean) + logit_variance
    variance_scale = jnp.einsum(
        "ild,de,ile->il",
        jnp.abs(weights),
        jnp.abs(symmetric_covariance),
        jnp.abs(weights),
    )
    second_scale = jnp.maximum(
        jnp.maximum(jnp.abs(raw_second), variance_scale),
        jnp.asarray(jnp.finfo(dtype).tiny, dtype),
    )
    # Invariant protected: only contraction roundoff may make E[psi^2]
    # negative; materially invalid Gaussian moments remain improper.
    second_tolerance = (
        32.0 * (dimension + 1) * jnp.finfo(dtype).eps * second_scale
    )
    second_ok = jnp.all(raw_second >= -second_tolerance)
    xi = jnp.sqrt(jnp.maximum(raw_second, 0.0))
    omega = _pg_mean(xi)

    common = jax.nn.log_sigmoid(xi) - 0.5 * xi
    success = common + 0.5 * logit_mean
    failure = common - 0.5 * logit_mean
    failure_prefix = jnp.concatenate(
        [
            jnp.zeros((card, 1), dtype=dtype),
            jnp.cumsum(failure, axis=-1),
        ],
        axis=-1,
    )
    log_potential = jnp.concatenate(
        [
            failure_prefix[:, :-1] + success,
            failure_prefix[:, -1:],
        ],
        axis=-1,
    )
    finite_output = (
        jnp.all(jnp.isfinite(omega))
        & jnp.all(jnp.isfinite(log_potential))
    )
    proper = parameter_ok & covariance_ok & second_ok & finite_output

    def valid_or_nan(value):
        return jnp.where(proper, value, jnp.full_like(value, jnp.nan))

    return StructuredPairPotential(
        valid_or_nan(log_potential),
        valid_or_nan(omega),
        proper,
    )


def _log_input_ok(value: jax.Array) -> jax.Array:
    return jnp.all(jnp.isfinite(value) | jnp.isneginf(value))


def _normalized_log(value: jax.Array) -> tuple[jax.Array, jax.Array]:
    normalizer = jax.scipy.special.logsumexp(value)
    proper = jnp.isfinite(normalizer)
    normalized = value - jnp.where(proper, normalizer, 0.0)
    return normalized, proper


def structured_pair_belief(
    source_cavity,
    destination_cavity,
    log_potential,
) -> StructuredPairBelief:
    """Apply D9 sum-product messages and form one pair-region belief.

    ``source_cavity`` is the incoming log message on the recurrence ``in``
    role.  ``destination_cavity`` is the incoming log message on its ``out``
    role.  Negative infinity denotes literal zero support.
    """
    source = jnp.asarray(source_cavity)
    destination = jnp.asarray(destination_cavity)
    potential = jnp.asarray(log_potential)
    if (
        source.ndim != 1
        or destination.shape != source.shape
        or source.shape[0] < 2
        or potential.shape != (source.shape[0], source.shape[0])
    ):
        raise ValueError(
            "source/destination cavities require [K] and pair potential "
            "[K,K]"
        )

    dtype = _floating_dtype(source, destination, potential)
    source = source.astype(dtype)
    destination = destination.astype(dtype)
    potential = potential.astype(dtype)
    inputs_ok = (
        _log_input_ok(source)
        & _log_input_ok(destination)
        & _log_input_ok(potential)
    )
    safe_source = jnp.where(
        jnp.isfinite(source) | jnp.isneginf(source), source, -jnp.inf
    )
    safe_destination = jnp.where(
        jnp.isfinite(destination) | jnp.isneginf(destination),
        destination,
        -jnp.inf,
    )
    safe_potential = jnp.where(
        jnp.isfinite(potential) | jnp.isneginf(potential),
        potential,
        -jnp.inf,
    )

    raw_to_out = jax.scipy.special.logsumexp(
        safe_source[:, None] + safe_potential, axis=0
    )
    raw_to_in = jax.scipy.special.logsumexp(
        safe_potential + safe_destination[None, :], axis=1
    )
    to_out, out_ok = _normalized_log(raw_to_out)
    to_in, in_ok = _normalized_log(raw_to_in)
    pair_log, pair_ok = _normalized_log(
        (
            safe_source[:, None]
            + safe_potential
            + safe_destination[None, :]
        ).reshape(-1)
    )
    pair = jnp.exp(pair_log).reshape(potential.shape)
    proper = inputs_ok & out_ok & in_ok & pair_ok
    source_probability = pair.sum(axis=1)
    destination_probability = pair.sum(axis=0)

    def valid_or_nan(value):
        return jnp.where(proper, value, jnp.full_like(value, jnp.nan))

    return StructuredPairBelief(
        valid_or_nan(to_out),
        valid_or_nan(to_in),
        valid_or_nan(pair),
        valid_or_nan(source_probability),
        valid_or_nan(destination_probability),
        proper,
    )


def _pair_probability_ok(probability: jax.Array) -> jax.Array:
    card = probability.shape[0]
    tolerance = (
        1024.0 * card * jnp.finfo(probability.dtype).eps
    )
    return (
        jnp.all(jnp.isfinite(probability))
        & jnp.all(probability >= 0)
        & (jnp.abs(probability.sum() - 1.0) <= tolerance)
    )


def structured_gaussian_message(
    pair_probability,
    omega,
    weights,
    offsets,
) -> StructuredGaussianMessage:
    """Build D9's Gaussian message directly from a joint pair belief."""
    pair = jnp.asarray(pair_probability)
    omega = jnp.asarray(omega)
    weights = jnp.asarray(weights)
    offsets = jnp.asarray(offsets)
    if pair.ndim != 2 or pair.shape[0] != pair.shape[1] or pair.shape[0] < 2:
        raise ValueError("pair probability requires [K,K]")
    card = pair.shape[0]
    if (
        weights.ndim != 3
        or weights.shape[:2] != (card, card - 1)
        or omega.shape != (card, card - 1)
        or offsets.shape != (card, card - 1)
    ):
        raise ValueError(
            "Gaussian message requires pair [K,K], omega/offsets "
            "[K,K-1], and weights [K,K-1,D]"
        )

    dtype = _floating_dtype(pair, omega, weights, offsets)
    pair = pair.astype(dtype)
    omega = omega.astype(dtype)
    weights = weights.astype(dtype)
    offsets = offsets.astype(dtype)
    exposure = jnp.cumsum(pair[:, ::-1], axis=1)[:, ::-1][:, :-1]
    signed_success = pair[:, :-1] - 0.5 * exposure
    precision_weight = exposure * omega
    linear_weight = signed_success - precision_weight * offsets
    eta = jnp.einsum("il,ild->d", linear_weight, weights)
    precision = jnp.einsum(
        "il,ild,ile->de", precision_weight, weights, weights
    )
    precision = 0.5 * (precision + precision.T)
    inputs_ok = (
        _pair_probability_ok(pair)
        & jnp.all(jnp.isfinite(omega))
        & jnp.all(omega >= 0)
        & jnp.all(jnp.isfinite(weights))
        & jnp.all(jnp.isfinite(offsets))
    )
    finite_output = (
        jnp.all(jnp.isfinite(exposure))
        & jnp.all(jnp.isfinite(signed_success))
        & jnp.all(jnp.isfinite(eta))
        & jnp.all(jnp.isfinite(precision))
        & gauss._psd_ok(precision)
    )
    proper = inputs_ok & finite_output

    def valid_or_nan(value):
        return jnp.where(proper, value, jnp.full_like(value, jnp.nan))

    return StructuredGaussianMessage(
        valid_or_nan(eta),
        valid_or_nan(precision),
        valid_or_nan(exposure),
        valid_or_nan(signed_success),
        proper,
    )


def structured_pair_energy(pair_probability, log_potential) -> jax.Array:
    """Return ``-H[rho] - E_rho[ell]`` for one pair region."""
    pair = jnp.asarray(pair_probability)
    potential = jnp.asarray(log_potential)
    if (
        pair.ndim != 2
        or pair.shape[0] != pair.shape[1]
        or pair.shape[0] < 2
        or potential.shape != pair.shape
    ):
        raise ValueError("pair energy requires probability/potential [K,K]")
    dtype = _floating_dtype(pair, potential)
    pair = pair.astype(dtype)
    potential = potential.astype(dtype)
    potential_ok = _log_input_ok(potential)
    supported = (pair == 0) | jnp.isfinite(potential)
    positive = pair > 0
    safe_log_pair = jnp.where(positive, jnp.log(pair), 0.0)
    safe_potential = jnp.where(positive, potential, 0.0)
    negative_entropy = jnp.sum(pair * safe_log_pair)
    expected_potential = jnp.sum(pair * safe_potential)
    value = negative_entropy - expected_potential
    proper = (
        _pair_probability_ok(pair)
        & potential_ok
        & jnp.all(supported)
        & jnp.isfinite(value)
    )
    return jnp.where(proper, value, jnp.inf)


def structured_separator_energy(probability) -> jax.Array:
    """Return the ``+H[gamma]`` interior-edge separator correction."""
    probability = jnp.asarray(probability)
    if probability.ndim != 1 or probability.shape[0] < 2:
        raise ValueError("separator probability requires [K]")
    dtype = _floating_dtype(probability)
    probability = probability.astype(dtype)
    tolerance = (
        1024.0 * probability.shape[0] * jnp.finfo(dtype).eps
    )
    proper = (
        jnp.all(jnp.isfinite(probability))
        & jnp.all(probability >= 0)
        & (jnp.abs(probability.sum() - 1.0) <= tolerance)
    )
    positive = probability > 0
    safe_log_probability = jnp.where(
        positive, jnp.log(probability), 0.0
    )
    entropy = -jnp.sum(probability * safe_log_probability)
    return jnp.where(proper & jnp.isfinite(entropy), entropy, jnp.inf)


def structured_node_energy(
    pair_probability,
    log_potential,
    drive_precision,
) -> jax.Array:
    """Return the D9 hybrid node energy including Gaussian ``-H[q]``."""
    precision = jnp.asarray(drive_precision)
    if (
        precision.ndim != 2
        or precision.shape[0] != precision.shape[1]
        or precision.shape[0] < 1
    ):
        raise ValueError("drive precision requires [D,D]")
    pair_value = structured_pair_energy(pair_probability, log_potential)
    gaussian_value = gauss._neg_entropy(precision)
    value = pair_value + gaussian_value
    return jnp.where(jnp.isfinite(value), value, jnp.inf)


__all__ = [
    "StructuredGaussianMessage",
    "StructuredPairBelief",
    "StructuredPairPotential",
    "structured_gaussian_message",
    "structured_node_energy",
    "structured_pair_belief",
    "structured_pair_energy",
    "structured_pair_potential",
    "structured_separator_energy",
]

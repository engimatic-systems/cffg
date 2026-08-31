"""Structured recurrence beliefs and pair-derived learning statistics."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from cffg import gauss
from cffg.structured_recurrence import (
    structured_gaussian_message,
    structured_pair_belief,
    structured_pair_potential,
)


class StructuredRecurrenceReadout(NamedTuple):
    """One normalized D9 pair and its shared Gaussian/learning masses."""

    pair_probability: jax.Array
    source_probability: jax.Array
    destination_probability: jax.Array
    pair_support: jax.Array
    source_support: jax.Array
    destination_support: jax.Array
    drive_mean: jax.Array
    drive_covariance: jax.Array
    drive_second: jax.Array
    omega: jax.Array
    exposure: jax.Array
    signed_success: jax.Array
    proper: jax.Array


class RecurrenceWeightStats(NamedTuple):
    """Pair-derived natural-statistic increments for stick coefficients."""

    omega: jax.Array
    exposure: jax.Array
    signed_success: jax.Array
    linear: jax.Array
    quadratic: jax.Array
    proper: jax.Array
    probability_ok: jax.Array
    drive_ok: jax.Array
    parameter_ok: jax.Array
    second_ok: jax.Array
    finite_output_ok: jax.Array
    probability_minimum: jax.Array
    probability_sum_error: jax.Array
    probability_tolerance: jax.Array
    drive_min_eigenvalue: jax.Array
    drive_symmetry_error: jax.Array
    drive_tolerance: jax.Array
    logit_second_minimum: jax.Array
    logit_second_tolerance: jax.Array


def structured_readout(
    destination_cavity,
    source_cavity,
    drive_eta,
    drive_precision,
    weights,
    offsets,
) -> StructuredRecurrenceReadout:
    """Batched public readout of the factor-local D9 recurrence belief.

    Pair rows are source regimes and columns are destination regimes. Inputs
    use negative infinity for literal categorical zeros. A readout requires a
    proper Gaussian drive; the vacuous state accepted as an inference startup
    transient is therefore improper here.
    """

    def one(destination, source, eta, precision, row_weights, row_offsets):
        dimension = eta.shape[0]
        source_support = jnp.isfinite(source)
        destination_support = jnp.isfinite(destination)
        pair_support = source_support[:, None] & destination_support[None, :]
        precision_ok = gauss._pd_ok(precision)
        safe_precision = jnp.where(
            precision_ok,
            precision,
            jnp.eye(dimension, dtype=precision.dtype),
        )
        covariance = gauss._pd_inverse(safe_precision)
        mean = gauss._pd_solve(safe_precision, eta)
        second = covariance + jnp.outer(mean, mean)
        potential = structured_pair_potential(
            mean, covariance, row_weights, row_offsets
        )
        pair = structured_pair_belief(
            source, destination, potential.log_potential
        )
        gaussian = structured_gaussian_message(
            pair.probability,
            potential.omega,
            row_weights,
            row_offsets,
        )
        proper = (
            precision_ok
            & potential.proper
            & pair.proper
            & gaussian.proper
            & jnp.all(jnp.isfinite(mean))
            & jnp.all(jnp.isfinite(covariance))
            & jnp.all(jnp.isfinite(second))
        )

        def valid_or_nan(value):
            return jnp.where(
                proper, value, jnp.full_like(value, jnp.nan)
            )

        return StructuredRecurrenceReadout(
            valid_or_nan(pair.probability),
            valid_or_nan(pair.source_probability),
            valid_or_nan(pair.destination_probability),
            pair_support,
            source_support,
            destination_support,
            valid_or_nan(mean),
            valid_or_nan(covariance),
            valid_or_nan(second),
            valid_or_nan(potential.omega),
            valid_or_nan(gaussian.exposure),
            valid_or_nan(gaussian.signed_success),
            proper,
        )

    return jax.vmap(one)(
        destination_cavity,
        source_cavity,
        drive_eta,
        drive_precision,
        weights,
        offsets,
    )


def weight_stats(
    pair_probability,
    drive_mean,
    drive_covariance,
    weights,
    offsets,
) -> RecurrenceWeightStats:
    """Build D9.12--D9.14 coefficient statistics from normalized pairs."""

    def one(pair, mean, covariance, row_weights, row_offsets):
        card = pair.shape[0]
        dimension = mean.shape[0]
        dtype = jnp.result_type(
            pair, mean, covariance, row_weights, row_offsets, jnp.float32
        )
        pair = pair.astype(dtype)
        mean = mean.astype(dtype)
        covariance = covariance.astype(dtype)
        row_weights = row_weights.astype(dtype)
        row_offsets = row_offsets.astype(dtype)

        probability_tolerance = (
            1024.0 * card * jnp.finfo(dtype).eps
        )
        probability_sum_error = jnp.abs(pair.sum() - 1.0)
        probability_minimum = jnp.min(pair)
        probability_ok = (
            jnp.all(jnp.isfinite(pair))
            & jnp.all(pair >= 0)
            & (probability_sum_error <= probability_tolerance)
        )

        covariance_finite = jnp.all(jnp.isfinite(covariance))
        safe_covariance = jnp.where(
            covariance_finite,
            covariance,
            jnp.eye(dimension, dtype=dtype),
        )
        symmetric_covariance = 0.5 * (
            safe_covariance + safe_covariance.swapaxes(-1, -2)
        )
        drive_min_eigenvalue = jnp.linalg.eigvalsh(symmetric_covariance)[0]
        drive_symmetry_error = jnp.max(
            jnp.abs(safe_covariance - safe_covariance.swapaxes(-1, -2))
        )
        drive_tolerance = (
            16.0
            * dimension
            * jnp.finfo(dtype).eps
            * gauss._matrix_scale(safe_covariance)
        )
        drive_ok = (
            jnp.all(jnp.isfinite(mean))
            & covariance_finite
            & gauss._psd_ok(safe_covariance)
        )
        parameter_ok = (
            jnp.all(jnp.isfinite(row_weights))
            & jnp.all(jnp.isfinite(row_offsets))
        )

        second = symmetric_covariance + jnp.outer(mean, mean)
        feature_mean = jnp.concatenate(
            [mean, jnp.ones(1, dtype=dtype)]
        )
        feature_second = jnp.zeros(
            (dimension + 1, dimension + 1), dtype=dtype
        )
        feature_second = feature_second.at[:-1, :-1].set(second)
        feature_second = feature_second.at[:-1, -1].set(mean)
        feature_second = feature_second.at[-1, :-1].set(mean)
        feature_second = feature_second.at[-1, -1].set(1.0)

        logit_mean = (
            jnp.einsum("ild,d->il", row_weights, mean) + row_offsets
        )
        logit_variance = jnp.einsum(
            "ild,de,ile->il",
            row_weights,
            symmetric_covariance,
            row_weights,
        )
        raw_logit_second = jnp.square(logit_mean) + logit_variance
        variance_scale = jnp.einsum(
            "ild,de,ile->il",
            jnp.abs(row_weights),
            jnp.abs(symmetric_covariance),
            jnp.abs(row_weights),
        )
        second_scale = jnp.maximum(
            jnp.maximum(jnp.abs(raw_logit_second), variance_scale),
            jnp.asarray(jnp.finfo(dtype).tiny, dtype),
        )
        logit_second_tolerance_rows = (
            32.0
            * (dimension + 1)
            * jnp.finfo(dtype).eps
            * second_scale
        )
        second_ok = jnp.all(
            raw_logit_second >= -logit_second_tolerance_rows
        )
        logit_second_index = jnp.argmin(raw_logit_second)
        logit_second_minimum = raw_logit_second.reshape(-1)[
            logit_second_index
        ]
        logit_second_tolerance = logit_second_tolerance_rows.reshape(-1)[
            logit_second_index
        ]

        potential = structured_pair_potential(
            mean,
            symmetric_covariance,
            row_weights,
            row_offsets,
        )
        gaussian = structured_gaussian_message(
            pair,
            potential.omega,
            row_weights,
            row_offsets,
        )
        linear = gaussian.signed_success[..., None] * feature_mean
        quadratic = (
            gaussian.exposure[..., None, None]
            * potential.omega[..., None, None]
            * feature_second
        )
        finite_output_ok = (
            jnp.all(jnp.isfinite(potential.omega))
            & jnp.all(jnp.isfinite(gaussian.exposure))
            & jnp.all(jnp.isfinite(gaussian.signed_success))
            & jnp.all(jnp.isfinite(linear))
            & jnp.all(jnp.isfinite(quadratic))
        )
        proper = (
            probability_ok
            & drive_ok
            & parameter_ok
            & second_ok
            & potential.proper
            & gaussian.proper
            & finite_output_ok
        )

        def valid_or_nan(value):
            return jnp.where(
                proper, value, jnp.full_like(value, jnp.nan)
            )

        return RecurrenceWeightStats(
            valid_or_nan(potential.omega),
            valid_or_nan(gaussian.exposure),
            valid_or_nan(gaussian.signed_success),
            valid_or_nan(linear),
            valid_or_nan(quadratic),
            proper,
            probability_ok,
            drive_ok,
            parameter_ok,
            second_ok,
            finite_output_ok,
            probability_minimum,
            probability_sum_error,
            probability_tolerance,
            drive_min_eigenvalue,
            drive_symmetry_error,
            drive_tolerance,
            logit_second_minimum,
            logit_second_tolerance,
        )

    return jax.vmap(one)(
        pair_probability,
        drive_mean,
        drive_covariance,
        weights,
        offsets,
    )


__all__ = [
    "RecurrenceWeightStats",
    "StructuredRecurrenceReadout",
    "structured_readout",
    "weight_stats",
]

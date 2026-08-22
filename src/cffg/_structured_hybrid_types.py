"""Shared values for the pure structured-hybrid authority."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


SWEEP_FINITE = 0
SWEEP_DIVERGED = 2
LOG2PI = jnp.log(2.0 * jnp.pi)


class StructuredHybridModel(NamedTuple):
    """Tied-boundary rSLDS potentials and installed point parameters."""

    log_initial: jax.Array
    log_unary: jax.Array
    gaussian_eta: jax.Array
    gaussian_precision: jax.Array
    gaussian_log_constant: jax.Array
    dynamics_matrix: jax.Array
    dynamics_offset: jax.Array
    dynamics_covariance: jax.Array
    recurrence_weights: jax.Array
    recurrence_offsets: jax.Array


class CategoricalTrajectory(NamedTuple):
    """Normalized chain belief with source-row/destination-column pairs."""

    log_singleton: jax.Array
    singleton: jax.Array
    pair: jax.Array
    log_normalizer: jax.Array
    entropy: jax.Array
    proper: jax.Array


class GaussianTrajectory(NamedTuple):
    """Block-Gaussian belief and adjacent-time moments."""

    mean: jax.Array
    covariance: jax.Array
    lag_covariance: jax.Array
    second_moment: jax.Array
    lag_second_moment: jax.Array
    precision_diagonal: jax.Array
    precision_lower: jax.Array
    information: jax.Array
    entropy: jax.Array
    proper: jax.Array


class RecurrenceAuxiliary(NamedTuple):
    """Expected-log transition tables and the expanded-bound auxiliary."""

    log_potential: jax.Array
    omega: jax.Array
    xi: jax.Array
    proper: jax.Array


class DynamicsStatistics(NamedTuple):
    """D10.5 component-MNIW increments without prior installation."""

    count: jax.Array
    syy: jax.Array
    syx: jax.Array
    sxx: jax.Array
    proper: jax.Array


class RecurrenceStatistics(NamedTuple):
    """Pair-derived PG increments without prior installation."""

    omega: jax.Array
    exposure: jax.Array
    signed_success: jax.Array
    linear: jax.Array
    quadratic: jax.Array
    proper: jax.Array


class StructuredHybridBelief(NamedTuple):
    """The two structured factors in the D10 variational family."""

    categorical: CategoricalTrajectory
    gaussian: GaussianTrajectory


class StructuredHybridSweep(NamedTuple):
    """One auxiliary, categorical, Gaussian, auxiliary sweep."""

    belief: StructuredHybridBelief
    recurrence: RecurrenceAuxiliary
    dynamics_log_potential: jax.Array
    dynamics_statistics: DynamicsStatistics
    recurrence_statistics: RecurrenceStatistics
    previous_objective: jax.Array
    categorical_objective: jax.Array
    fixed_auxiliary_objective: jax.Array
    objective: jax.Array
    residual: jax.Array
    status: jax.Array


def floating_dtype(*values) -> jnp.dtype:
    return jnp.result_type(*values, jnp.float32)


def valid_log(value: jax.Array) -> jax.Array:
    return jnp.all(jnp.isfinite(value) | jnp.isneginf(value))


def entropy(probability: jax.Array) -> jax.Array:
    positive = probability > 0
    return -jnp.sum(
        jnp.where(positive, probability * jnp.log(probability), 0.0)
    )


def probability_rows_ok(probability: jax.Array) -> jax.Array:
    width = probability.shape[-1]
    tolerance = 1024.0 * width * jnp.finfo(probability.dtype).eps
    return (
        jnp.all(jnp.isfinite(probability))
        & jnp.all(probability >= 0)
        & jnp.all(jnp.abs(probability.sum(axis=-1) - 1.0) <= tolerance)
    )


def model_arrays(model: StructuredHybridModel) -> StructuredHybridModel:
    values = tuple(jnp.asarray(value) for value in model)
    dtype = floating_dtype(*values)
    return StructuredHybridModel(*(value.astype(dtype) for value in values))


def model_shape(model: StructuredHybridModel) -> tuple[int, int, int]:
    time, card = model.log_unary.shape if model.log_unary.ndim == 2 else (-1, -1)
    dimension = model.gaussian_eta.shape[1] if model.gaussian_eta.ndim == 2 else -1
    if (
        time < 1
        or card < 1
        or dimension < 1
        or model.log_initial.shape != (card,)
        or model.gaussian_eta.shape != (time, dimension)
        or model.gaussian_precision.shape != (time, dimension, dimension)
        or model.gaussian_log_constant.shape != (time,)
        or model.dynamics_matrix.shape != (card, dimension, dimension)
        or model.dynamics_offset.shape != (card, dimension)
        or model.dynamics_covariance.shape != (card, dimension, dimension)
        or model.recurrence_weights.shape != (card, card - 1, dimension)
        or model.recurrence_offsets.shape != (card, card - 1)
    ):
        raise ValueError(
            "structured hybrid model shapes must be [K], [T,K], [T,D], "
            "[T,D,D], [T], [K,D,D], [K,D], [K,D,D], "
            "[K,K-1,D], and [K,K-1]"
        )
    return time, card, dimension

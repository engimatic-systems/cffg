"""Objective and learning-statistic authorities for D10."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from cffg import gauss
from cffg._structured_hybrid_chain import recurrence_auxiliary
from cffg._structured_hybrid_types import (
    CategoricalTrajectory,
    DynamicsStatistics,
    GaussianTrajectory,
    RecurrenceStatistics,
    floating_dtype,
    probability_rows_ok,
    valid_log,
)


def _expected_log(probability, log_potential) -> tuple[jax.Array, jax.Array]:
    supported = (probability == 0) | jnp.isfinite(log_potential)
    safe = jnp.where(probability > 0, log_potential, 0.0)
    value = jnp.sum(probability * safe)
    proper = valid_log(log_potential) & jnp.all(supported) & jnp.isfinite(value)
    return value, proper


def structured_hybrid_objective(
    categorical: CategoricalTrajectory,
    gaussian: GaussianTrajectory,
    log_initial,
    log_unary,
    gaussian_eta,
    gaussian_precision,
    gaussian_log_constant,
    dynamics_log_potential,
    recurrence_log_potential,
) -> jax.Array:
    """Evaluate the expanded or collapsed D10.6 objective.

    Supply a fixed-``xi`` recurrence table during the Gaussian coordinate and
    a reoptimized table only at a complete sweep boundary.
    """
    initial = jnp.asarray(log_initial)
    unary = jnp.asarray(log_unary)
    external_eta = jnp.asarray(gaussian_eta)
    external_precision = jnp.asarray(gaussian_precision)
    external_constant = jnp.asarray(gaussian_log_constant)
    dynamics = jnp.asarray(dynamics_log_potential)
    recurrence = jnp.asarray(recurrence_log_potential)
    time, card = categorical.singleton.shape
    dimension = gaussian.mean.shape[1]
    if (
        initial.shape != (card,)
        or unary.shape != (time, card)
        or external_eta.shape != (time, dimension)
        or external_precision.shape != (time, dimension, dimension)
        or external_constant.shape != (time,)
        or dynamics.shape != (time - 1, card)
        or recurrence.shape != (time - 1, card, card)
    ):
        raise ValueError("structured hybrid objective shapes disagree")
    dtype = floating_dtype(
        categorical.singleton,
        gaussian.mean,
        initial,
        unary,
        external_eta,
        external_precision,
        external_constant,
        dynamics,
        recurrence,
    )
    (
        initial,
        unary,
        external_eta,
        external_precision,
        external_constant,
        dynamics,
        recurrence,
    ) = tuple(
        value.astype(dtype)
        for value in (
            initial,
            unary,
            external_eta,
            external_precision,
            external_constant,
            dynamics,
            recurrence,
        )
    )
    initial_value, initial_ok = _expected_log(
        categorical.singleton[0], initial
    )
    unary_value, unary_ok = _expected_log(categorical.singleton, unary)
    dynamics_value, dynamics_ok = _expected_log(
        categorical.singleton[1:], dynamics
    )
    recurrence_value, recurrence_ok = _expected_log(
        categorical.pair, recurrence
    )
    external_value = jnp.sum(
        -0.5
        * jnp.einsum(
            "tij,tji->t", external_precision, gaussian.second_moment
        )
        + jnp.einsum("ti,ti->t", external_eta, gaussian.mean)
        + external_constant
    )
    external_ok = (
        jnp.all(jnp.isfinite(external_eta))
        & jnp.all(jnp.isfinite(external_precision))
        & jnp.all(jnp.isfinite(external_constant))
        & jnp.all(jax.vmap(gauss._psd_ok)(external_precision))
    )
    value = (
        -categorical.entropy
        - gaussian.entropy
        - initial_value
        - unary_value
        - external_value
        - dynamics_value
        - recurrence_value
    )
    proper = (
        categorical.proper
        & gaussian.proper
        & initial_ok
        & unary_ok
        & dynamics_ok
        & recurrence_ok
        & external_ok
        & jnp.isfinite(external_value)
        & jnp.isfinite(value)
    )
    return jnp.where(proper, value, jnp.inf)


def _feature_moments(gaussian: GaussianTrajectory):
    time, dimension = gaussian.mean.shape
    dtype = gaussian.mean.dtype
    feature_mean = jnp.concatenate(
        [gaussian.mean[:-1], jnp.ones((time - 1, 1), dtype=dtype)], axis=1
    )
    feature_second = jnp.zeros(
        (time - 1, dimension + 1, dimension + 1), dtype=dtype
    )
    feature_second = feature_second.at[:, :-1, :-1].set(
        gaussian.second_moment[:-1]
    )
    feature_second = feature_second.at[:, :-1, -1].set(
        gaussian.mean[:-1]
    )
    feature_second = feature_second.at[:, -1, :-1].set(
        gaussian.mean[:-1]
    )
    feature_second = feature_second.at[:, -1, -1].set(1.0)
    return feature_mean, feature_second


def dynamics_statistics(
    singleton,
    gaussian: GaussianTrajectory,
) -> DynamicsStatistics:
    """Return singleton-weighted D10.5 dynamics statistics."""
    singleton = jnp.asarray(singleton)
    time, _ = gaussian.mean.shape
    if singleton.ndim != 2 or singleton.shape[0] != time:
        raise ValueError("dynamics statistics require singleton [T,K]")
    dtype = floating_dtype(singleton, gaussian.mean)
    singleton = singleton.astype(dtype)
    feature_mean, feature_second = _feature_moments(gaussian)
    cross = jnp.concatenate(
        [gaussian.lag_second_moment, gaussian.mean[1:, :, None]], axis=2
    )
    responsibility = singleton[1:]
    count = responsibility.sum(axis=0)
    syy = jnp.einsum(
        "tk,tij->kij", responsibility, gaussian.second_moment[1:]
    )
    syx = jnp.einsum("tk,tij->kij", responsibility, cross)
    sxx = jnp.einsum("tk,tij->kij", responsibility, feature_second)
    proper = (
        gaussian.proper
        & probability_rows_ok(singleton)
        & jnp.all(jnp.isfinite(feature_mean))
        & jnp.all(jnp.isfinite(count))
        & jnp.all(jnp.isfinite(syy))
        & jnp.all(jnp.isfinite(syx))
        & jnp.all(jnp.isfinite(sxx))
    )

    def valid_or_nan(value):
        return jnp.where(proper, value, jnp.full_like(value, jnp.nan))

    return DynamicsStatistics(
        valid_or_nan(count),
        valid_or_nan(syy),
        valid_or_nan(syx),
        valid_or_nan(sxx),
        proper,
    )


def recurrence_statistics(
    pair,
    gaussian: GaussianTrajectory,
    weights,
    offsets,
) -> RecurrenceStatistics:
    """Return joint-pair D9.14 statistics from the same D10 belief."""
    pair = jnp.asarray(pair)
    weights = jnp.asarray(weights)
    offsets = jnp.asarray(offsets)
    time, dimension = gaussian.mean.shape
    if pair.ndim != 3 or pair.shape[0] != time - 1:
        raise ValueError("recurrence statistics require pair [T-1,K,K]")
    card = pair.shape[1]
    if (
        pair.shape[2] != card
        or weights.shape != (card, card - 1, dimension)
        or offsets.shape != (card, card - 1)
    ):
        raise ValueError("recurrence statistic shapes disagree")
    dtype = floating_dtype(pair, gaussian.mean, weights, offsets)
    pair = pair.astype(dtype)
    weights = weights.astype(dtype)
    offsets = offsets.astype(dtype)
    auxiliaries = jax.vmap(
        lambda mean, covariance: recurrence_auxiliary(
            mean, covariance, weights, offsets
        )
    )(gaussian.mean[:-1], gaussian.covariance[:-1])
    exposure = jnp.cumsum(pair[:, :, ::-1], axis=2)[:, :, ::-1][
        :, :, :-1
    ]
    signed_success = pair[:, :, :-1] - 0.5 * exposure
    feature_mean, feature_second = _feature_moments(gaussian)
    linear = jnp.einsum("til,ta->ila", signed_success, feature_mean)
    quadratic = jnp.einsum(
        "til,til,tab->ilab",
        exposure,
        auxiliaries.omega,
        feature_second,
    )
    pair_ok = (
        jnp.all(jnp.isfinite(pair))
        & jnp.all(pair >= 0)
        & jnp.all(
            jnp.abs(pair.sum(axis=(1, 2)) - 1.0)
            <= 1024.0 * card * jnp.finfo(dtype).eps
        )
    )
    proper = (
        gaussian.proper
        & pair_ok
        & jnp.all(auxiliaries.proper)
        & jnp.all(jnp.isfinite(exposure))
        & jnp.all(jnp.isfinite(signed_success))
        & jnp.all(jnp.isfinite(linear))
        & jnp.all(jnp.isfinite(quadratic))
    )

    def valid_or_nan(value):
        return jnp.where(proper, value, jnp.full_like(value, jnp.nan))

    return RecurrenceStatistics(
        valid_or_nan(auxiliaries.omega),
        valid_or_nan(exposure),
        valid_or_nan(signed_success),
        valid_or_nan(linear),
        valid_or_nan(quadratic),
        proper,
    )

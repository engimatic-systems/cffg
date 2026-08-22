"""Categorical and recurrence coordinates for structured-hybrid inference."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from cffg import gauss
from cffg._structured_hybrid_types import (
    CategoricalTrajectory,
    RecurrenceAuxiliary,
    entropy,
    floating_dtype,
    probability_rows_ok,
    valid_log,
)
from cffg.structured_recurrence import structured_pair_potential


def _normalize_log(value: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    normalizer = jax.scipy.special.logsumexp(value)
    proper = jnp.isfinite(normalizer)
    normalized = value - jnp.where(proper, normalizer, 0.0)
    return normalized, normalizer, proper


def categorical_coordinate(
    log_initial,
    log_unary,
    dynamics_log_potential,
    recurrence_log_potential,
) -> CategoricalTrajectory:
    """Return the exact D10.3 categorical-chain coordinate in log space."""
    initial = jnp.asarray(log_initial)
    unary = jnp.asarray(log_unary)
    dynamics = jnp.asarray(dynamics_log_potential)
    transition = jnp.asarray(recurrence_log_potential)
    if unary.ndim != 2:
        raise ValueError("categorical unary evidence requires [T,K]")
    time, card = unary.shape
    if (
        time < 1
        or card < 1
        or initial.shape != (card,)
        or dynamics.shape != (time - 1, card)
        or transition.shape != (time - 1, card, card)
    ):
        raise ValueError(
            "categorical coordinate requires [K], [T,K], [T-1,K], "
            "and [T-1,K,K]"
        )
    dtype = floating_dtype(initial, unary, dynamics, transition)
    initial = initial.astype(dtype)
    unary = unary.astype(dtype)
    dynamics = dynamics.astype(dtype)
    transition = transition.astype(dtype)
    combined = unary.at[1:].add(dynamics)
    inputs_ok = valid_log(initial) & valid_log(combined) & valid_log(transition)
    alpha0, scale0, alpha0_ok = _normalize_log(initial + combined[0])

    def forward(log_alpha, values):
        pair_potential, next_unary = values
        predicted = jax.scipy.special.logsumexp(
            log_alpha[:, None] + pair_potential, axis=0
        )
        next_alpha, scale, proper = _normalize_log(predicted + next_unary)
        return next_alpha, (next_alpha, scale, proper)

    _, (alpha_tail, scales, alpha_ok) = jax.lax.scan(
        forward, alpha0, (transition, combined[1:])
    )
    log_alpha = jnp.concatenate([alpha0[None, :], alpha_tail], axis=0)

    def backward(log_beta, values):
        pair_potential, next_unary = values
        raw = jax.scipy.special.logsumexp(
            pair_potential + next_unary[None, :] + log_beta[None, :], axis=1
        )
        next_beta, _, proper = _normalize_log(raw)
        return next_beta, (next_beta, proper)

    _, (beta_head, beta_ok) = jax.lax.scan(
        backward,
        jnp.zeros((card,), dtype=dtype),
        (transition, combined[1:]),
        reverse=True,
    )
    log_beta = jnp.concatenate(
        [beta_head, jnp.zeros((1, card), dtype=dtype)], axis=0
    )

    def normalize_singleton(value):
        normalized, _, proper = _normalize_log(value)
        return normalized, proper

    log_singleton, singleton_ok = jax.vmap(normalize_singleton)(
        log_alpha + log_beta
    )
    singleton = jnp.exp(log_singleton)
    pair_log = (
        log_alpha[:-1, :, None]
        + transition
        + combined[1:, None, :]
        + log_beta[1:, None, :]
    )

    def normalize_pair(value):
        normalized, _, proper = _normalize_log(value.reshape(-1))
        return normalized.reshape(value.shape), proper

    normalized_pair, pair_ok = jax.vmap(normalize_pair)(pair_log)
    pair = jnp.exp(normalized_pair)
    tolerance = 2048.0 * card * jnp.finfo(dtype).eps
    pair_consistent = (
        jnp.all(jnp.abs(pair.sum(axis=2) - singleton[:-1]) <= tolerance)
        & jnp.all(jnp.abs(pair.sum(axis=1) - singleton[1:]) <= tolerance)
    )
    if time == 1:
        chain_entropy = entropy(singleton[0])
    else:
        chain_entropy = jnp.sum(jax.vmap(entropy)(pair))
        if time > 2:
            chain_entropy = chain_entropy - jnp.sum(
                jax.vmap(entropy)(singleton[1:-1])
            )
    log_normalizer = scale0 + jnp.sum(scales)
    proper = (
        inputs_ok
        & alpha0_ok
        & jnp.all(alpha_ok)
        & jnp.all(beta_ok)
        & jnp.all(singleton_ok)
        & jnp.all(pair_ok)
        & pair_consistent
        & probability_rows_ok(singleton)
        & jnp.isfinite(chain_entropy)
        & jnp.isfinite(log_normalizer)
    )

    def valid_or_nan(value):
        return jnp.where(proper, value, jnp.full_like(value, jnp.nan))

    return CategoricalTrajectory(
        valid_or_nan(log_singleton),
        valid_or_nan(singleton),
        valid_or_nan(pair),
        jnp.where(proper, log_normalizer, jnp.inf),
        jnp.where(proper, chain_entropy, jnp.inf),
        proper,
    )


def _pg_mean(xi: jax.Array) -> jax.Array:
    threshold = jnp.sqrt(jnp.finfo(xi.dtype).eps)
    safe_xi = jnp.where(xi > threshold, xi, jnp.ones_like(xi))
    direct = jnp.tanh(0.5 * xi) / (2.0 * safe_xi)
    square = jnp.square(xi)
    series = 0.25 - square / 48.0 + jnp.square(square) / 480.0
    return jnp.where(xi > threshold, direct, series)


def recurrence_auxiliary(
    drive_mean,
    drive_covariance,
    weights,
    offsets,
    *,
    fixed_xi=None,
) -> RecurrenceAuxiliary:
    """Return optimized or fixed-auxiliary expected-log transition tables."""
    mean = jnp.asarray(drive_mean)
    covariance = jnp.asarray(drive_covariance)
    weights = jnp.asarray(weights)
    offsets = jnp.asarray(offsets)
    if mean.ndim != 1 or covariance.shape != (mean.shape[0], mean.shape[0]):
        raise ValueError("drive moments require mean [D] and covariance [D,D]")
    dimension = mean.shape[0]
    if weights.ndim != 3:
        raise ValueError("recurrence weights require [K,K-1,D]")
    card, sticks, weight_dimension = weights.shape
    if (
        card < 1
        or sticks != card - 1
        or weight_dimension != dimension
        or offsets.shape != (card, sticks)
    ):
        raise ValueError(
            "recurrence parameters require [K,K-1,D] and [K,K-1]"
        )
    dtype = floating_dtype(mean, covariance, weights, offsets)
    mean = mean.astype(dtype)
    covariance = covariance.astype(dtype)
    weights = weights.astype(dtype)
    offsets = offsets.astype(dtype)
    moment_ok = (
        jnp.all(jnp.isfinite(mean))
        & jnp.all(jnp.isfinite(covariance))
        & gauss._psd_ok(covariance)
    )
    parameter_ok = (
        jnp.all(jnp.isfinite(weights)) & jnp.all(jnp.isfinite(offsets))
    )
    if card == 1:
        log_potential = jnp.zeros((1, 1), dtype=dtype)
        empty = jnp.zeros((1, 0), dtype=dtype)
        proper = moment_ok & parameter_ok
        return RecurrenceAuxiliary(
            jnp.where(proper, log_potential, jnp.nan),
            jnp.where(proper, empty, jnp.nan),
            jnp.where(proper, empty, jnp.nan),
            proper,
        )

    symmetric_covariance = 0.5 * (covariance + covariance.T)
    logit_mean = jnp.einsum("ild,d->il", weights, mean) + offsets
    logit_variance = jnp.einsum(
        "ild,de,ile->il", weights, symmetric_covariance, weights
    )
    raw_second = jnp.square(logit_mean) + logit_variance
    scale = jnp.maximum(
        jnp.maximum(jnp.abs(raw_second), jnp.square(logit_mean)),
        jnp.asarray(jnp.finfo(dtype).tiny, dtype),
    )
    tolerance = 32.0 * (dimension + 1) * jnp.finfo(dtype).eps * scale
    second_ok = jnp.all(raw_second >= -tolerance)
    second = jnp.maximum(raw_second, 0.0)
    optimized_xi = jnp.sqrt(second)
    if fixed_xi is None:
        authority = structured_pair_potential(
            mean, covariance, weights, offsets
        )
        proper = moment_ok & parameter_ok & second_ok & authority.proper
        log_potential = authority.log_potential
        omega = authority.omega
        xi = optimized_xi
    else:
        xi = jnp.asarray(fixed_xi, dtype=dtype)
        if xi.shape != (card, sticks):
            raise ValueError("fixed recurrence xi requires [K,K-1]")
        xi_ok = jnp.all(jnp.isfinite(xi)) & jnp.all(xi >= 0)
        omega = _pg_mean(xi)
        common = (
            jax.nn.log_sigmoid(xi)
            - 0.5 * xi
            + 0.5 * omega * jnp.square(xi)
            - 0.5 * omega * second
        )
        success = common + 0.5 * logit_mean
        failure = common - 0.5 * logit_mean
        prefix = jnp.concatenate(
            [
                jnp.zeros((card, 1), dtype=dtype),
                jnp.cumsum(failure, axis=-1),
            ],
            axis=-1,
        )
        log_potential = jnp.concatenate(
            [prefix[:, :-1] + success, prefix[:, -1:]], axis=-1
        )
        proper = (
            moment_ok
            & parameter_ok
            & second_ok
            & xi_ok
            & jnp.all(jnp.isfinite(log_potential))
            & jnp.all(jnp.isfinite(omega))
        )

    def valid_or_nan(value):
        return jnp.where(proper, value, jnp.full_like(value, jnp.nan))

    return RecurrenceAuxiliary(
        valid_or_nan(log_potential),
        valid_or_nan(omega),
        valid_or_nan(xi),
        proper,
    )


def conditioned_categorical_coordinate(
    log_initial,
    log_unary,
    fixed_drives,
    weights,
    offsets,
) -> CategoricalTrajectory:
    """D9 reduction for conditioned drives; no singular Gaussian is formed."""
    drives = jnp.asarray(fixed_drives)
    if drives.ndim != 2:
        raise ValueError("conditioned drives require [T-1,D]")
    dtype = floating_dtype(drives, weights, offsets)
    covariance = jnp.zeros(
        (drives.shape[0], drives.shape[1], drives.shape[1]), dtype=dtype
    )
    auxiliaries = jax.vmap(
        lambda mean, cov: recurrence_auxiliary(mean, cov, weights, offsets)
    )(drives.astype(dtype), covariance)
    card = jnp.asarray(log_initial).shape[0]
    dynamics = jnp.zeros((drives.shape[0], card), dtype=dtype)
    result = categorical_coordinate(
        log_initial,
        log_unary,
        dynamics,
        auxiliaries.log_potential,
    )
    proper = result.proper & jnp.all(auxiliaries.proper)
    return CategoricalTrajectory(
        jnp.where(proper, result.log_singleton, jnp.nan),
        jnp.where(proper, result.singleton, jnp.nan),
        jnp.where(proper, result.pair, jnp.nan),
        jnp.where(proper, result.log_normalizer, jnp.inf),
        jnp.where(proper, result.entropy, jnp.inf),
        proper,
    )

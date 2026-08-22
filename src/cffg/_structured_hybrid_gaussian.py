"""Gaussian trajectory coordinate for structured-hybrid inference."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from cffg import gauss
from cffg._structured_hybrid_types import (
    GaussianTrajectory,
    LOG2PI,
    floating_dtype,
    probability_rows_ok,
)


def expected_dynamics_log_potential(
    mean,
    covariance,
    lag_covariance,
    matrices,
    offsets,
    process_covariance,
) -> tuple[jax.Array, jax.Array]:
    """Return D10.2 for each destination time and regime."""
    mean = jnp.asarray(mean)
    covariance = jnp.asarray(covariance)
    lag = jnp.asarray(lag_covariance)
    matrices = jnp.asarray(matrices)
    offsets = jnp.asarray(offsets)
    process_covariance = jnp.asarray(process_covariance)
    if mean.ndim != 2:
        raise ValueError("Gaussian means require [T,D]")
    time, dimension = mean.shape
    if (
        covariance.shape != (time, dimension, dimension)
        or lag.shape != (time - 1, dimension, dimension)
        or matrices.ndim != 3
        or matrices.shape[1:] != (dimension, dimension)
        or offsets.shape != (matrices.shape[0], dimension)
        or process_covariance.shape != matrices.shape
    ):
        raise ValueError("dynamics moments or parameter shapes disagree")
    dtype = floating_dtype(
        mean, covariance, lag, matrices, offsets, process_covariance
    )
    mean = mean.astype(dtype)
    covariance = covariance.astype(dtype)
    lag = lag.astype(dtype)
    matrices = matrices.astype(dtype)
    offsets = offsets.astype(dtype)
    process_covariance = process_covariance.astype(dtype)
    process_ok = jax.vmap(gauss._pd_ok)(process_covariance)
    safe_process = jnp.where(
        process_ok[:, None, None],
        process_covariance,
        jnp.broadcast_to(
            jnp.eye(dimension, dtype=dtype), process_covariance.shape
        ),
    )
    precision = jax.vmap(gauss._pd_inverse)(safe_process)
    logdet_precision = jax.vmap(gauss._pd_logdet)(precision)
    residual_mean = (
        mean[1:, None, :]
        - jnp.einsum("kij,tj->tki", matrices, mean[:-1])
        - offsets[None, :, :]
    )
    residual_covariance = (
        covariance[1:, None, :, :]
        + jnp.einsum(
            "kij,tjl,knl->tkin", matrices, covariance[:-1], matrices
        )
        - jnp.einsum("tij,klj->tkil", lag, matrices)
        - jnp.einsum("kij,tlj->tkil", matrices, lag)
    )
    residual_second = residual_covariance + jnp.einsum(
        "tki,tkj->tkij", residual_mean, residual_mean
    )
    quadratic = jnp.einsum("kij,tkji->tk", precision, residual_second)
    log_potential = (
        0.5 * logdet_precision[None, :]
        - 0.5 * dimension * LOG2PI
        - 0.5 * quadratic
    )
    joint_covariance = jnp.concatenate(
        [
            jnp.concatenate([covariance[1:], lag], axis=2),
            jnp.concatenate(
                [lag.swapaxes(-1, -2), covariance[:-1]], axis=2
            ),
        ],
        axis=1,
    )
    moments_ok = (
        jnp.all(jnp.isfinite(mean))
        & jnp.all(jnp.isfinite(covariance))
        & jnp.all(jnp.isfinite(lag))
        & jnp.all(jax.vmap(gauss._psd_ok)(covariance))
        & jnp.all(jax.vmap(gauss._psd_ok)(joint_covariance))
    )
    parameter_ok = (
        jnp.all(jnp.isfinite(matrices))
        & jnp.all(jnp.isfinite(offsets))
        & jnp.all(process_ok)
    )
    proper = moments_ok & parameter_ok & jnp.all(jnp.isfinite(log_potential))
    return jnp.where(proper, log_potential, jnp.nan), proper


def _assemble_gaussian_precision(
    external_eta,
    external_precision,
    singleton,
    matrices,
    offsets,
    process_covariance,
    recurrence_eta,
    recurrence_precision,
):
    _, dimension = external_eta.shape
    process_ok = jax.vmap(gauss._pd_ok)(process_covariance)
    safe_process = jnp.where(
        process_ok[:, None, None],
        process_covariance,
        jnp.broadcast_to(
            jnp.eye(dimension, dtype=external_eta.dtype),
            process_covariance.shape,
        ),
    )
    weights = jax.vmap(gauss._pd_inverse)(safe_process)
    wa = jnp.einsum("kij,kjl->kil", weights, matrices)
    atwa = jnp.einsum("kji,kjl->kil", matrices, wa)
    wb = jnp.einsum("kij,kj->ki", weights, offsets)
    atwb = jnp.einsum("kji,kj->ki", matrices, wb)
    destination = singleton[1:]
    out_precision = jnp.einsum("tk,kij->tij", destination, weights)
    in_precision = jnp.einsum("tk,kij->tij", destination, atwa)
    lower = -jnp.einsum("tk,kij->tij", destination, wa)
    out_eta = jnp.einsum("tk,ki->ti", destination, wb)
    in_eta = -jnp.einsum("tk,ki->ti", destination, atwb)
    diagonal = external_precision
    information = external_eta
    diagonal = diagonal.at[1:].add(out_precision)
    diagonal = diagonal.at[:-1].add(in_precision)
    information = information.at[1:].add(out_eta)
    information = information.at[:-1].add(in_eta)
    diagonal = diagonal.at[:-1].add(recurrence_precision)
    information = information.at[:-1].add(recurrence_eta)
    diagonal = 0.5 * (diagonal + diagonal.swapaxes(-1, -2))
    return information, diagonal, lower, process_ok


def gaussian_coordinate(
    external_eta,
    external_precision,
    singleton,
    matrices,
    offsets,
    process_covariance,
    recurrence_eta,
    recurrence_precision,
) -> GaussianTrajectory:
    """Return the exact D10.4 block-tridiagonal Gaussian coordinate."""
    external_eta = jnp.asarray(external_eta)
    external_precision = jnp.asarray(external_precision)
    singleton = jnp.asarray(singleton)
    matrices = jnp.asarray(matrices)
    offsets = jnp.asarray(offsets)
    process_covariance = jnp.asarray(process_covariance)
    recurrence_eta = jnp.asarray(recurrence_eta)
    recurrence_precision = jnp.asarray(recurrence_precision)
    if external_eta.ndim != 2 or singleton.ndim != 2:
        raise ValueError(
            "Gaussian eta and categorical singleton require [T,D]/[T,K]"
        )
    time, dimension = external_eta.shape
    card = singleton.shape[1]
    if (
        time < 1
        or dimension < 1
        or singleton.shape[0] != time
        or external_precision.shape != (time, dimension, dimension)
        or matrices.shape != (card, dimension, dimension)
        or offsets.shape != (card, dimension)
        or process_covariance.shape != (card, dimension, dimension)
        or recurrence_eta.shape != (time - 1, dimension)
        or recurrence_precision.shape != (time - 1, dimension, dimension)
    ):
        raise ValueError("Gaussian coordinate shapes disagree")
    dtype = floating_dtype(
        external_eta,
        external_precision,
        singleton,
        matrices,
        offsets,
        process_covariance,
        recurrence_eta,
        recurrence_precision,
    )
    (
        external_eta,
        external_precision,
        singleton,
        matrices,
        offsets,
        process_covariance,
        recurrence_eta,
        recurrence_precision,
    ) = tuple(
        value.astype(dtype)
        for value in (
            external_eta,
            external_precision,
            singleton,
            matrices,
            offsets,
            process_covariance,
            recurrence_eta,
            recurrence_precision,
        )
    )
    input_ok = (
        probability_rows_ok(singleton)
        & jnp.all(jnp.isfinite(external_eta))
        & jnp.all(jnp.isfinite(external_precision))
        & jnp.all(jax.vmap(gauss._psd_ok)(external_precision))
        & jnp.all(jnp.isfinite(matrices))
        & jnp.all(jnp.isfinite(offsets))
        & jnp.all(jnp.isfinite(process_covariance))
        & jnp.all(jnp.isfinite(recurrence_eta))
        & jnp.all(jnp.isfinite(recurrence_precision))
        & jnp.all(jax.vmap(gauss._psd_ok)(recurrence_precision))
    )
    information, diagonal, lower, process_ok = _assemble_gaussian_precision(
        external_eta,
        external_precision,
        singleton,
        matrices,
        offsets,
        process_covariance,
        recurrence_eta,
        recurrence_precision,
    )
    identity = jnp.eye(dimension, dtype=dtype)
    first_ok = gauss._pd_ok(diagonal[0])
    first_schur = jnp.where(first_ok, diagonal[0], identity)
    first_rhs = jnp.where(
        first_ok, information[0], jnp.zeros_like(information[0])
    )

    def factor(carry, values):
        previous_schur, previous_rhs, previous_ok = carry
        block, diagonal_block, rhs = values
        gain = gauss._pd_solve(previous_schur, block.T).T
        schur = diagonal_block - gain @ block.T
        schur = 0.5 * (schur + schur.T)
        current_ok = gauss._pd_ok(schur) & jnp.all(jnp.isfinite(rhs))
        safe_schur = jnp.where(current_ok, schur, identity)
        next_rhs = rhs - gain @ previous_rhs
        safe_rhs = jnp.where(current_ok, next_rhs, jnp.zeros_like(next_rhs))
        return (
            safe_schur,
            safe_rhs,
            previous_ok & current_ok,
        ), (safe_schur, safe_rhs, current_ok)

    (_, _, factor_ok), (schur_tail, rhs_tail, local_ok) = jax.lax.scan(
        factor,
        (first_schur, first_rhs, first_ok),
        (lower, diagonal[1:], information[1:]),
    )
    schur = jnp.concatenate([first_schur[None, :, :], schur_tail], axis=0)
    rhs = jnp.concatenate([first_rhs[None, :], rhs_tail], axis=0)
    solved = jax.vmap(gauss._pd_solve)(schur, rhs)

    def back_mean(next_mean, values):
        local_solved, local_schur, block = values
        mean = local_solved - gauss._pd_solve(
            local_schur, block.T @ next_mean
        )
        return mean, mean

    _, mean_head = jax.lax.scan(
        back_mean,
        solved[-1],
        (solved[:-1], schur[:-1], lower),
        reverse=True,
    )
    mean = jnp.concatenate([mean_head, solved[-1:]], axis=0)
    inverse_schur = jax.vmap(gauss._pd_inverse)(schur)

    def back_covariance(next_covariance, values):
        local_inverse, local_schur, block = values
        gain = gauss._pd_solve(local_schur, block.T)
        covariance = local_inverse + gain @ next_covariance @ gain.T
        covariance = 0.5 * (covariance + covariance.T)
        lag_covariance = -next_covariance @ gain.T
        return covariance, (covariance, lag_covariance)

    _, (covariance_head, lag_covariance) = jax.lax.scan(
        back_covariance,
        inverse_schur[-1],
        (inverse_schur[:-1], schur[:-1], lower),
        reverse=True,
    )
    covariance = jnp.concatenate(
        [covariance_head, inverse_schur[-1:]], axis=0
    )
    second = covariance + jnp.einsum("ti,tj->tij", mean, mean)
    lag_second = lag_covariance + jnp.einsum(
        "ti,tj->tij", mean[1:], mean[:-1]
    )
    logdet = jnp.sum(jax.vmap(gauss._pd_logdet)(schur))
    trajectory_entropy = 0.5 * (
        time * dimension * (1.0 + LOG2PI) - logdet
    )
    proper = (
        input_ok
        & jnp.all(process_ok)
        & first_ok
        & factor_ok
        & jnp.all(local_ok)
        & jnp.all(jnp.isfinite(mean))
        & jnp.all(jnp.isfinite(covariance))
        & jnp.all(jnp.isfinite(lag_covariance))
        & jnp.isfinite(trajectory_entropy)
    )

    def valid_or_nan(value):
        return jnp.where(proper, value, jnp.full_like(value, jnp.nan))

    return GaussianTrajectory(
        valid_or_nan(mean),
        valid_or_nan(covariance),
        valid_or_nan(lag_covariance),
        valid_or_nan(second),
        valid_or_nan(lag_second),
        valid_or_nan(diagonal),
        valid_or_nan(lower),
        valid_or_nan(information),
        jnp.where(proper, trajectory_entropy, jnp.inf),
        proper,
    )

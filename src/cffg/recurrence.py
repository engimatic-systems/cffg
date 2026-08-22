"""Variational messages for a recurrent stick-breaking transition.

For previous regime ``i``, destination ``j``, and stick ``l < K - 1``,

``psi[i, l](x) = weights[i, l] @ x + offsets[i, l]``.

The destination takes stick ``j`` after rejecting every earlier stick, while
the final destination rejects all sticks. The exact Pólya–Gamma augmentation
of each at-risk Bernoulli term is conditionally Gaussian in ``x``. Replacing
the auxiliary by its coordinate-optimal mean-field factor
``PG(b, xi)``, with ``xi² = E[psi²]``, is exactly the Jaakkola–Jordan bound.

The bound is lower than the exact expected log transition. Consequently the
node energy returned here is an upper bound on the exact variational node
energy at the same edge beliefs. The discrete and Gaussian edge entropies are
included once in the node term; the engine's ordinary interior-edge entropy
corrections then restore the global region counting numbers.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from cffg import gauss
from cffg.compiler import NEG
from cffg.structured_recurrence import (
    StructuredGaussianMessage,
    StructuredPairBelief,
    StructuredPairPotential,
    StructuredRecurrenceUpdate as StructuredRecurrenceUpdate,
    structured_gaussian_message,
    structured_node_energy,
    structured_pair_belief,
    structured_pair_energy,
    structured_pair_potential,
    structured_recurrence_energy as structured_recurrence_energy,
    structured_recurrence_update as structured_recurrence_update,
    structured_separator_energy,
)


class RecurrenceUpdate(NamedTuple):
    """Batched factor messages and local numerical status."""

    to_out: jax.Array
    to_in: jax.Array
    drive_eta: jax.Array
    drive_precision: jax.Array
    status: jax.Array


class RecurrenceBelief(NamedTuple):
    """Categorical and continuous moments of a recurrence-link belief."""

    out_probability: jax.Array
    in_probability: jax.Array
    drive_mean: jax.Array
    drive_covariance: jax.Array
    drive_second: jax.Array
    out_support: jax.Array
    in_support: jax.Array
    proper: jax.Array


class RecurrenceWeightStats(NamedTuple):
    """Natural-statistic increments for augmented stick coefficients."""

    omega: jax.Array
    exposure: jax.Array
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


def _edge_probability(messages: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Product of two categorical messages without relaxing exact support."""
    support = jnp.all(messages != jnp.asarray(NEG, messages.dtype), axis=-2)
    log_probability = jnp.where(support, messages.sum(axis=-2), -jnp.inf)
    valid = jnp.any(support) & jnp.all(jnp.isfinite(messages))
    normalizer = jnp.where(
        valid,
        jax.scipy.special.logsumexp(log_probability),
        jnp.asarray(0.0, messages.dtype),
    )
    log_probability = jnp.where(support, log_probability - normalizer, -jnp.inf)
    probability = jnp.where(support, jnp.exp(log_probability), 0.0)
    return probability, log_probability, valid


def belief(
    out_messages,
    in_messages,
    drive_eta,
    drive_precision,
) -> RecurrenceBelief:
    """Read the current three-edge belief without evaluating the PG bound."""

    def one(out_m, in_m, eta, precision):
        q_out, _, out_ok = _edge_probability(out_m)
        q_in, _, in_ok = _edge_probability(in_m)
        out_support = jnp.all(
            out_m != jnp.asarray(NEG, out_m.dtype), axis=-2
        )
        in_support = jnp.all(
            in_m != jnp.asarray(NEG, in_m.dtype), axis=-2
        )
        precision_ok = gauss._pd_ok(precision)
        safe_precision = jnp.where(
            precision_ok,
            precision,
            jnp.eye(precision.shape[-1], dtype=precision.dtype),
        )
        covariance = gauss._pd_inverse(safe_precision)
        mean = gauss._pd_solve(safe_precision, eta)
        second = covariance + jnp.outer(mean, mean)
        proper = (
            out_ok
            & in_ok
            & precision_ok
            & jnp.all(jnp.isfinite(mean))
            & jnp.all(jnp.isfinite(covariance))
            & jnp.all(jnp.isfinite(second))
        )
        return RecurrenceBelief(
            jnp.where(proper, q_out, jnp.full_like(q_out, jnp.nan)),
            jnp.where(proper, q_in, jnp.full_like(q_in, jnp.nan)),
            jnp.where(proper, mean, jnp.full_like(mean, jnp.nan)),
            jnp.where(proper, covariance, jnp.full_like(covariance, jnp.nan)),
            jnp.where(proper, second, jnp.full_like(second, jnp.nan)),
            out_support,
            in_support,
            proper,
        )

    return jax.vmap(one)(
        out_messages,
        in_messages,
        drive_eta,
        drive_precision,
    )


def weight_stats(
    out_probability,
    in_probability,
    drive_mean,
    drive_covariance,
    weights,
    offsets,
) -> RecurrenceWeightStats:
    """Build PG Gaussian statistics for ``[weights; offset]``.

    Inputs are batched factor-local mean-field beliefs. The direct drive
    covariance avoids reconstructing it from nearly equal second-moment terms.
    The PG coordinate is evaluated at the point parameters used by inference;
    each returned row is one between-trial natural-parameter increment.
    """

    def one(q_out, q_in, mean, covariance, row_weights, row_offsets):
        card = q_out.shape[-1]
        dimension = mean.shape[-1]
        probability_tolerance = (
            32.0 * card * jnp.finfo(q_out.dtype).eps
        )
        probability_ok = (
            jnp.all(jnp.isfinite(q_out))
            & jnp.all(jnp.isfinite(q_in))
            & jnp.all(q_out >= 0)
            & jnp.all(q_in >= 0)
            & (jnp.abs(q_out.sum() - 1.0) <= probability_tolerance)
            & (jnp.abs(q_in.sum() - 1.0) <= probability_tolerance)
        )
        probability_minimum = jnp.minimum(jnp.min(q_out), jnp.min(q_in))
        probability_sum_error = jnp.maximum(
            jnp.abs(q_out.sum() - 1.0),
            jnp.abs(q_in.sum() - 1.0),
        )
        symmetric_covariance = 0.5 * (
            covariance + covariance.swapaxes(-1, -2)
        )
        drive_min_eigenvalue = jnp.linalg.eigvalsh(symmetric_covariance)[0]
        drive_symmetry_error = jnp.max(
            jnp.abs(covariance - covariance.swapaxes(-1, -2))
        )
        drive_tolerance = (
            16.0
            * dimension
            * jnp.finfo(covariance.dtype).eps
            * gauss._matrix_scale(covariance)
        )
        drive_ok = (
            jnp.all(jnp.isfinite(mean))
            & jnp.all(jnp.isfinite(covariance))
            & gauss._psd_ok(covariance)
        )
        parameter_ok = (
            jnp.all(jnp.isfinite(row_weights))
            & jnp.all(jnp.isfinite(row_offsets))
        )

        # Invariant protected: after material asymmetry is rejected above,
        # sufficient statistics use one exactly symmetric covariance.
        covariance = symmetric_covariance
        second = covariance + jnp.outer(mean, mean)
        feature_mean = jnp.concatenate(
            [mean, jnp.ones(1, dtype=mean.dtype)]
        )
        feature_second = jnp.zeros(
            (dimension + 1, dimension + 1), dtype=second.dtype
        )
        feature_second = feature_second.at[:-1, :-1].set(second)
        feature_second = feature_second.at[:-1, -1].set(mean)
        feature_second = feature_second.at[-1, :-1].set(mean)
        feature_second = feature_second.at[-1, -1].set(1.0)
        logit_mean = (
            jnp.einsum("ild,d->il", row_weights, mean) + row_offsets
        )
        logit_variance = jnp.einsum(
            "ild,de,ile->il", row_weights, covariance, row_weights
        )
        raw_logit_second = jnp.square(logit_mean) + logit_variance
        variance_scale = jnp.einsum(
            "ild,de,ile->il",
            jnp.abs(row_weights),
            jnp.abs(covariance),
            jnp.abs(row_weights),
        )
        second_scale = jnp.maximum(
            jnp.maximum(
                jnp.maximum(jnp.abs(raw_logit_second), jnp.square(logit_mean)),
                variance_scale,
            ),
            jnp.asarray(jnp.finfo(raw_logit_second.dtype).tiny),
        )
        second_tolerance = (
            32.0
            * (dimension + 1)
            * jnp.finfo(raw_logit_second.dtype).eps
            * second_scale
        )
        second_ok = jnp.all(raw_logit_second >= -second_tolerance)
        logit_second_index = jnp.argmin(raw_logit_second)
        logit_second_minimum = raw_logit_second.reshape(-1)[logit_second_index]
        logit_second_tolerance = second_tolerance.reshape(-1)[
            logit_second_index
        ]
        xi = jnp.sqrt(jnp.maximum(raw_logit_second, 0.0))
        omega = _pg_mean(xi)

        at_risk = jnp.cumsum(q_out[::-1])[::-1][:-1]
        kappa = q_out[:-1] - 0.5 * at_risk
        exposure = q_in[:, None] * at_risk[None, :]
        linear_weight = q_in[:, None] * kappa[None, :]
        linear = linear_weight[..., None] * feature_mean[None, None, :]
        quadratic = (
            exposure[..., None, None]
            * omega[..., None, None]
            * feature_second[None, None, :, :]
        )
        finite_output_ok = (
            jnp.all(jnp.isfinite(omega))
            & jnp.all(jnp.isfinite(exposure))
            & jnp.all(jnp.isfinite(linear))
            & jnp.all(jnp.isfinite(quadratic))
        )
        proper = (
            probability_ok
            & drive_ok
            & parameter_ok
            & second_ok
            & finite_output_ok
        )

        def valid_or_nan(value):
            return jnp.where(proper, value, jnp.full_like(value, jnp.nan))

        return RecurrenceWeightStats(
            valid_or_nan(omega),
            valid_or_nan(exposure),
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
        out_probability,
        in_probability,
        drive_mean,
        drive_covariance,
        weights,
        offsets,
    )


def _pg_mean(xi: jax.Array) -> jax.Array:
    """E[PG(1, xi)] with a dtype-scaled analytic limit at zero."""
    threshold = jnp.sqrt(jnp.finfo(xi.dtype).eps)
    safe_xi = jnp.where(xi > threshold, xi, jnp.ones_like(xi))
    direct = jnp.tanh(0.5 * xi) / (2.0 * safe_xi)
    square = jnp.square(xi)
    series = 0.25 - square / 48.0 + jnp.square(square) / 480.0
    return jnp.where(xi > threshold, direct, series)


def _coordinates(
    q_out: jax.Array,
    q_in: jax.Array,
    eta: jax.Array,
    precision: jax.Array,
    weights: jax.Array,
    offsets: jax.Array,
):
    covariance = gauss._pd_inverse(precision)
    mean = gauss._pd_solve(precision, eta)
    logit_mean = jnp.einsum("ild,d->il", weights, mean) + offsets
    logit_variance = jnp.einsum(
        "ild,de,ile->il", weights, covariance, weights
    )
    raw_second = jnp.square(logit_mean) + logit_variance

    # Invariant protected: only eigensolver/contraction roundoff may turn the
    # nonnegative second moment slightly negative; material negativity remains
    # a loud numerical failure.
    scale = jnp.maximum(
        jnp.maximum(jnp.abs(raw_second), jnp.square(logit_mean)),
        jnp.asarray(1.0, raw_second.dtype),
    )
    tolerance = (
        32.0
        * weights.shape[-1]
        * jnp.finfo(raw_second.dtype).eps
        * scale
    )
    second_ok = jnp.all(raw_second >= -tolerance)
    second = jnp.maximum(raw_second, 0.0)
    xi = jnp.sqrt(second)
    omega = _pg_mean(xi)

    common = jax.nn.log_sigmoid(xi) - 0.5 * xi
    success_score = common + 0.5 * logit_mean
    failure_score = common - 0.5 * logit_mean
    failure_prefix = jnp.concatenate(
        [
            jnp.zeros((*failure_score.shape[:-1], 1), failure_score.dtype),
            jnp.cumsum(failure_score, axis=-1),
        ],
        axis=-1,
    )
    scores = jnp.concatenate(
        [
            failure_prefix[..., :-1] + success_score,
            failure_prefix[..., -1:],
        ],
        axis=-1,
    )

    to_out = jnp.einsum("i,ij->j", q_in, scores)
    to_in = jnp.einsum("j,ij->i", q_out, scores)

    at_risk = jnp.cumsum(q_out[::-1])[::-1][:-1]
    kappa = q_out[:-1] - 0.5 * at_risk
    pg_weight = q_in[:, None] * at_risk[None, :] * omega
    linear_weight = q_in[:, None] * kappa[None, :] - pg_weight * offsets
    drive_eta = jnp.einsum("il,ild->d", linear_weight, weights)
    drive_precision = jnp.einsum(
        "il,ild,ile->de", pg_weight, weights, weights
    )
    drive_precision = 0.5 * (drive_precision + drive_precision.T)
    expected_log_bound = jnp.einsum("i,j,ij->", q_in, q_out, scores)
    finite = (
        second_ok
        & jnp.all(jnp.isfinite(scores))
        & jnp.all(jnp.isfinite(drive_eta))
        & jnp.all(jnp.isfinite(drive_precision))
        & gauss._psd_ok(drive_precision)
    )
    return (
        jax.nn.log_softmax(to_out),
        jax.nn.log_softmax(to_in),
        drive_eta,
        drive_precision,
        expected_log_bound,
        finite,
    )


def update(
    out_messages,
    in_messages,
    drive_eta,
    drive_precision,
    weights,
    offsets,
) -> RecurrenceUpdate:
    """Batched coordinate updates for all three recurrence-link edges.

    The Gaussian inputs are the full current edge belief (the sum of both
    information-form messages), as required by the local JJ coordinate.
    Exactly vacuous input is a neutral transient so flooding can receive a
    facing Gaussian prior on the next sweep. Other improper inputs are loud.
    """

    def one(out_m, in_m, eta, precision, row_weights, row_offsets):
        q_out, _, out_ok = _edge_probability(out_m)
        q_in, _, in_ok = _edge_probability(in_m)
        vacuous = jnp.all(eta == 0) & jnp.all(precision == 0)
        proper = gauss._pd_ok(precision)

        def neutral(_):
            return RecurrenceUpdate(
                jax.nn.log_softmax(jnp.zeros_like(q_out)),
                jax.nn.log_softmax(jnp.zeros_like(q_in)),
                jnp.zeros_like(eta),
                jnp.zeros_like(precision),
                jnp.asarray(0),
            )

        def evaluate(_):
            safe_precision = jnp.where(
                proper,
                precision,
                jnp.eye(precision.shape[-1], dtype=precision.dtype),
            )
            values = _coordinates(
                q_out,
                q_in,
                eta,
                safe_precision,
                row_weights,
                row_offsets,
            )
            to_out, to_in, out_eta, out_precision, _, finite = values
            valid = out_ok & in_ok & proper & finite
            return RecurrenceUpdate(
                jnp.where(valid, to_out, jnp.zeros_like(to_out)),
                jnp.where(valid, to_in, jnp.zeros_like(to_in)),
                jnp.where(valid, out_eta, jnp.zeros_like(out_eta)),
                jnp.where(valid, out_precision, jnp.zeros_like(out_precision)),
                jnp.where(valid, 0, 2),
            )

        return jax.lax.cond(vacuous, neutral, evaluate, operand=None)

    return jax.vmap(one)(
        out_messages,
        in_messages,
        drive_eta,
        drive_precision,
        weights,
        offsets,
    )


def energy(
    out_messages,
    in_messages,
    drive_eta,
    drive_precision,
    weights,
    offsets,
) -> jax.Array:
    """Batched JJ variational node energy, an upper-bound contribution."""

    def one(out_m, in_m, eta, precision, row_weights, row_offsets):
        q_out, log_q_out, out_ok = _edge_probability(out_m)
        q_in, log_q_in, in_ok = _edge_probability(in_m)
        proper = gauss._pd_ok(precision)
        safe_precision = jnp.where(
            proper,
            precision,
            jnp.eye(precision.shape[-1], dtype=precision.dtype),
        )
        values = _coordinates(
            q_out,
            q_in,
            eta,
            safe_precision,
            row_weights,
            row_offsets,
        )
        expected_log_bound, finite = values[-2:]
        neg_h_out = jnp.sum(
            jnp.where(q_out > 0, q_out * log_q_out, 0.0)
        )
        neg_h_in = jnp.sum(
            jnp.where(q_in > 0, q_in * log_q_in, 0.0)
        )
        value = (
            neg_h_out
            + neg_h_in
            + gauss._neg_entropy(safe_precision)
            - expected_log_bound
        )
        valid = out_ok & in_ok & proper & finite & jnp.isfinite(value)
        return jnp.where(valid, value, jnp.inf)

    return jax.vmap(one)(
        out_messages,
        in_messages,
        drive_eta,
        drive_precision,
        weights,
        offsets,
    )


def expected_log_bound(
    q_out,
    q_in,
    drive_eta,
    drive_precision,
    weights,
    offsets,
) -> jax.Array:
    """Expected JJ log-transition bound for already normalized beliefs."""

    def one(out_q, in_q, eta, precision, row_weights, row_offsets):
        return _coordinates(
            out_q,
            in_q,
            eta,
            precision,
            row_weights,
            row_offsets,
        )[-2]

    return jax.vmap(one)(
        q_out,
        q_in,
        drive_eta,
        drive_precision,
        weights,
        offsets,
    )

"""Explicit local updates for factors coupling different message domains."""

import jax
import jax.numpy as jnp

from cffg import gauss, mixg, recurrence
from cffg.compiled import MixGroupPlan, PrecGroupPlan, RecurrencePlan
from cffg.execution.contracts import (
    ExecutionDelta,
    GaussianMessage,
    HybridMessages,
    PrecisionMessages,
    RecurrenceMessages,
)
from cffg.execution.parameters import MixtureParameters, RecurrenceParameters
from cffg.execution.support import log_support


def execute_mixture(
    plan: MixGroupPlan,
    selector: jax.Array,
    eta: jax.Array,
    precision: jax.Array,
    params: MixtureParameters,
) -> ExecutionDelta[HybridMessages]:
    do = plan.dims[0]
    out_eta, out_precision = eta[:, 0, :do], precision[:, 0, :do, :do]
    if plan.type == "gaussian_mixture_linear":
        di = plan.dims[1]
        in_eta, in_precision = eta[:, 1, :di], precision[:, 1, :di, :di]
        gaussian = (
            GaussianMessage(
                *mixg.mixlin_forward(
                    in_eta,
                    in_precision,
                    selector,
                    params.matrix,
                    params.offset,
                    params.precision,
                )
            ),
            GaussianMessage(
                *mixg.mixlin_backward(
                    out_eta,
                    out_precision,
                    selector,
                    params.matrix,
                    params.offset,
                    params.precision,
                )
            ),
        )
        to_selector = mixg.mixlin_to_selector(
            out_eta,
            out_precision,
            in_eta,
            in_precision,
            params.matrix,
            params.offset,
            params.precision,
        )
    else:
        gaussian = (
            GaussianMessage(*mixg.mix_forward(selector, params.mean, params.precision)),
        )
        to_selector = mixg.mix_to_selector(
            out_eta, out_precision, params.mean, params.precision
        )
    return ExecutionDelta(HybridMessages(to_selector, gaussian))


def execute_recurrence(
    plan: RecurrencePlan,
    destination_cavity: jax.Array,
    source_cavity: jax.Array,
    drive_eta_both: jax.Array,
    drive_precision_both: jax.Array,
    params: RecurrenceParameters,
) -> ExecutionDelta[RecurrenceMessages]:
    """Categorical cavities and the complete Gaussian drive belief for D9."""
    destination = jnp.where(
        log_support(destination_cavity), destination_cavity, -jnp.inf
    )
    source = jnp.where(log_support(source_cavity), source_cavity, -jnp.inf)
    result = recurrence.structured_recurrence_update(
        destination,
        source,
        drive_eta_both.sum(axis=1),
        drive_precision_both.sum(axis=1),
        params.weights,
        params.offsets,
    )
    return ExecutionDelta(
        RecurrenceMessages(
            result.to_out,
            result.to_in,
            GaussianMessage(result.drive_eta, result.drive_precision),
        ),
        status=result.status,
    )


def execute_precision(
    plan: PrecGroupPlan,
    tau_both: jax.Array,
    eta_both: jax.Array,
    precision_both: jax.Array,
    mean: jax.Array,
) -> ExecutionDelta[PrecisionMessages]:
    """Both VMP rules consume edge beliefs, not one-sided Gaussian cavities."""
    gaussian = GaussianMessage(*gauss.prec_to_x(tau_both.sum(axis=1), mean))
    to_precision = gauss.prec_to_tau(
        eta_both.sum(axis=1), precision_both.sum(axis=1), mean
    )
    return ExecutionDelta(PrecisionMessages(gaussian, to_precision))

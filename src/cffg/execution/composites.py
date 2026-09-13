"""Explicit Dirichlet and P-substitution local message owners."""

import jax
import jax.numpy as jnp

from cffg import efrules, hybrid_gfe, rules
from cffg.compiled import NEG, DirichletGroupPlan, GfePlan, GfeTmPlan, HybridGfePlan
from cffg.execution.contracts import (
    DirichletMessages,
    ExecutionDelta,
    GaussianMessage,
    GfeMessages,
    HybridMessages,
    TransitionMessages,
)
from cffg.execution.parameters import GfeParameters, HybridParameters


def execute_dirichlet(
    plan: DirichletGroupPlan, incoming_both: jax.Array, prior: jax.Array
) -> ExecutionDelta[DirichletMessages]:
    outgoing, counts = rules.dirichlet_vmp_update(
        plan.type, incoming_both, prior, plan.cards, max(plan.cards)
    )
    return ExecutionDelta(DirichletMessages(outgoing, prior, counts))


def execute_gfe(
    plan: GfePlan,
    incoming: jax.Array,
    params: GfeParameters,
    goal_counts: jax.Array | None = None,
) -> ExecutionDelta[GfeMessages]:
    """One composite's messages and fresh contribution to a shared goal.

    The caller pools contributions across all groups before damping. A local
    executor must not finalize a parameter shared with another composite.
    """
    if plan.goal_learned:
        if plan.mu3:
            support = jnp.any(params.mean > 0, axis=2)
        elif plan.learned:
            support = jnp.any(params.channel > 0, axis=2)
        else:
            support = jnp.any(params.channel != NEG, axis=2)
        posterior = efrules.DIRICHLET.posterior(params.goal_prior, goal_counts)
        # Excluded rows must not form zero times trigamma(tiny) in reverse mode.
        safe = jnp.where(support, posterior, 1.0)
        log_goal = jnp.where(support, rules.dirichlet_expected_log(safe), 0.0)
    else:
        log_goal = params.log_goal
    if plan.mu3:
        latent, prediction, residual, status = rules.gfe_update_stats(
            incoming,
            params.mean,
            params.column_entropy,
            log_goal,
            max(plan.cards),
            return_diagnostics=True,
        )
    else:
        latent, prediction, residual, status = rules.gfe_update(
            incoming,
            params.channel,
            log_goal,
            max(plan.cards),
            learned=plan.learned,
            return_diagnostics=True,
        )
    goal_prediction = (
        jnp.exp(prediction[..., : plan.cards[0]]) if plan.goal_learned else None
    )
    return ExecutionDelta(
        GfeMessages(latent, prediction, params.goal_prior, goal_prediction),
        residual,
        status,
    )


def execute_transition_gfe(
    plan: GfeTmPlan,
    destination: jax.Array,
    source: jax.Array,
    selector: jax.Array,
    channel: jax.Array,
) -> ExecutionDelta[TransitionMessages]:
    to_out, to_in, to_selector, residual, status = rules.gfe_tm_update(
        source,
        selector,
        destination,
        channel,
        max(plan.cards),
        learned=plan.learned,
        return_diagnostics=True,
    )
    return ExecutionDelta(
        TransitionMessages(to_out, to_in, to_selector), residual, status
    )


def execute_hybrid_gfe(
    plan: HybridGfePlan,
    selector: jax.Array,
    eta: jax.Array,
    precision: jax.Array,
    params: HybridParameters,
) -> ExecutionDelta[HybridMessages]:
    channel = params.channel
    if plan.type == "gaussian_mixture_linear":
        di = plan.dims[1]
        to_selector, out_eta, out_lam, in_eta, in_lam, residual, status = (
            hybrid_gfe.linear_update(
                selector,
                eta[:, 1, :di],
                precision[:, 1, :di, :di],
                channel.matrix,
                channel.offset,
                channel.precision,
                params.goal_mean,
                params.goal_precision,
            )
        )
        gaussian = (GaussianMessage(out_eta, out_lam), GaussianMessage(in_eta, in_lam))
    else:
        to_selector, out_eta, out_lam, residual, status = hybrid_gfe.prior_update(
            selector,
            channel.mean,
            channel.precision,
            params.goal_mean,
            params.goal_precision,
        )
        gaussian = (GaussianMessage(out_eta, out_lam),)
    return ExecutionDelta(HybridMessages(to_selector, gaussian), residual, status)

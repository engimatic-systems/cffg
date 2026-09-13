"""Fixed-particle numerical phases, separate from stochastic lifecycle policy."""

import jax

from cffg import rules
from cffg.compiled import Mu3Plan
from cffg.execution.contracts import ExecutionDelta, FunctionalEstimate


def execute_functional_stationary(
    plan: Mu3Plan,
    incoming: jax.Array,
    mean: jax.Array,
    column_entropy: jax.Array,
    log_goal: jax.Array,
) -> ExecutionDelta[jax.Array]:
    beliefs, residual, status = rules.gfe_stationary_stats(
        incoming, mean, column_entropy, log_goal
    )
    return ExecutionDelta(beliefs, residual, status)


def execute_functional_importance(
    plan: Mu3Plan,
    particles: jax.Array,
    beliefs: jax.Array,
    log_goal: jax.Array,
    previous_mean: jax.Array,
) -> ExecutionDelta[FunctionalEstimate]:
    """Existing importance estimate; the host retains its low-ESS policy."""
    mean, entropy, weights, log_z, expected_phi, kl, ess, status = (
        rules.mu3_importance_update(particles, beliefs, log_goal, previous_mean)
    )
    return ExecutionDelta(
        FunctionalEstimate(mean, entropy, weights, log_z, expected_phi, kl, ess),
        status=status,
    )

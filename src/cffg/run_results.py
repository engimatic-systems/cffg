"""Public result construction from internal iteration evidence."""

import numpy as np

from cffg.iteration_types import IterationOutcome, IterationReason
from cffg.runtime_types import OuterTerminationReason, RunResult, RunStatus

_STATUS_BY_REASON = {
    IterationReason.BUDGET: RunStatus.BUDGET_EXHAUSTED,
    IterationReason.FE_STATIONARY: RunStatus.CONVERGED,
    IterationReason.MAX_STATIONARY: RunStatus.CONVERGED,
    IterationReason.INITIAL_NONFINITE: RunStatus.DIVERGED,
    IterationReason.UPDATE_NONFINITE: RunStatus.DIVERGED,
    IterationReason.EMPTY_SUPPORT: RunStatus.INFEASIBLE,
    IterationReason.INITIAL_EMPTY_SUPPORT: RunStatus.INFEASIBLE,
    IterationReason.INNER_BUDGET: RunStatus.BUDGET_EXHAUSTED,
    IterationReason.INNER_DIVERGED: RunStatus.DIVERGED,
    IterationReason.FUNCTIONAL_BUDGET: RunStatus.BUDGET_EXHAUSTED,
    IterationReason.FUNCTIONAL_DIVERGED: RunStatus.DIVERGED,
    IterationReason.OBJECTIVE_NONFINITE: RunStatus.DIVERGED,
    IterationReason.UNDEFINED_LEDGER: RunStatus.DIVERGED,
}


def finalize_iteration(
    outcome: IterationOutcome,
    *,
    legacy_picard: bool = False,
    mu3_ess: tuple[float, ...] = (),
    mu3_log_normalizers: tuple[float, ...] = (),
    mu3_statuses: tuple[RunStatus, ...] = (),
) -> RunResult:
    """Validate terminal evidence and preserve the public result field schema."""
    if outcome.reason is IterationReason.RUNNING:
        raise ValueError("iteration outcome is not terminal")
    status = _STATUS_BY_REASON[outcome.reason]
    deltas = np.asarray(outcome.message_deltas)
    energies = np.asarray(outcome.free_energies)
    if status is RunStatus.CONVERGED:
        if (
            outcome.state_finite is not True
            or outcome.zero_support is not False
            or outcome.inner_code != 0
            or not np.isfinite(outcome.inner_residual)
            or not np.isfinite(outcome.residual)
            or not len(deltas)
            or not np.isfinite(deltas).all()
        ):
            raise ValueError("converged iteration outcome lacks finite update evidence")
        if outcome.reason is IterationReason.FE_STATIONARY and (
            len(energies) < 2 or not np.isfinite(energies).all()
        ):
            raise ValueError("converged iteration outcome lacks a finite FE comparison")
    return RunResult(
        state=outcome.state,
        status=status,
        iterations=outcome.iterations,
        message_deltas=tuple(deltas.tolist()),
        free_energies=tuple(energies.tolist()),
        residual=outcome.residual,
        inner_status=(
            RunStatus.CONVERGED
            if outcome.inner_code == 0
            else RunStatus.BUDGET_EXHAUSTED
            if outcome.inner_code == 1
            else RunStatus.DIVERGED
        ),
        inner_residual=outcome.inner_residual,
        mu3_ess=mu3_ess,
        mu3_log_normalizers=mu3_log_normalizers,
        mu3_statuses=mu3_statuses,
        termination_reason=(
            OuterTerminationReason.LEGACY_PICARD
            if legacy_picard and not outcome.initial_rejection
            else OuterTerminationReason.NOT_RUN
        ),
    )

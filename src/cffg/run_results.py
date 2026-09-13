"""Public result construction from strategy-specific termination evidence."""

import numpy as np

from cffg.exact_strategy import ExactOutcome, ExactReason
from cffg.iteration_types import IterationOutcome, IterationReason
from cffg.outer_outcome import OuterOutcome
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

_EXACT_STATUS_BY_REASON = {
    ExactReason.CERTIFIED: RunStatus.CONVERGED,
    ExactReason.BUDGET: RunStatus.BUDGET_EXHAUSTED,
    ExactReason.INITIAL_NONFINITE: RunStatus.DIVERGED,
    ExactReason.PRE_EMPTY_SUPPORT: RunStatus.INFEASIBLE,
    ExactReason.UPDATE_NONFINITE: RunStatus.DIVERGED,
    ExactReason.POST_EMPTY_SUPPORT: RunStatus.INFEASIBLE,
    ExactReason.OBJECTIVE_NONFINITE: RunStatus.DIVERGED,
}


def finalize_exact(outcome: ExactOutcome) -> RunResult:
    """Preserve exact logical counts and the structural tolerance exception."""
    status = _EXACT_STATUS_BY_REASON[outcome.reason]
    deltas = np.asarray(outcome.message_deltas)
    energies = np.asarray(outcome.free_energies)
    if status is RunStatus.CONVERGED and (
        not (outcome.structural or outcome.by_tolerance)
        or outcome.iterations != 1
        or len(deltas) != 1
        or not np.isfinite(deltas).all()
        or not np.isfinite(energies).all()
        or not np.isfinite(outcome.residual)
    ):
        raise ValueError("converged exact outcome lacks a finite certificate")
    initial_nonfinite = outcome.reason is ExactReason.INITIAL_NONFINITE
    return RunResult(
        state=outcome.state,
        status=status,
        iterations=outcome.iterations,
        message_deltas=tuple(deltas.tolist()),
        free_energies=tuple(energies.tolist()),
        residual=outcome.residual,
        inner_status=RunStatus.DIVERGED if initial_nonfinite else RunStatus.CONVERGED,
        inner_residual=float("inf") if initial_nonfinite else 0.0,
    )


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


def finalize_outer(
    outcome: OuterOutcome,
    *,
    mu3_ess: tuple[float, ...] = (),
    mu3_log_normalizers: tuple[float, ...] = (),
    mu3_statuses: tuple[RunStatus, ...] = (),
) -> RunResult:
    """Map outer evidence, preserving its budget and functional-update semantics."""
    reason = outcome.reason
    deltas = np.asarray(outcome.message_deltas)
    energies = np.asarray(outcome.free_energies)
    if outcome.functional_failure:
        status = (
            RunStatus.BUDGET_EXHAUSTED
            if outcome.inner_code == 1
            else RunStatus.DIVERGED
        )
    elif (
        reason is OuterTerminationReason.ROOT_CONVERGED
        and not outcome.functional_pending
    ):
        status = (
            RunStatus.CONVERGED
            if outcome.tolerance_requested
            else RunStatus.BUDGET_EXHAUSTED
        )
    elif reason is OuterTerminationReason.INFEASIBLE:
        status = RunStatus.INFEASIBLE
    elif (
        reason is OuterTerminationReason.NONFINITE_EVALUATION
        or (reason is OuterTerminationReason.INNER_FAILURE and outcome.inner_code == 2)
        or (
            reason is OuterTerminationReason.SUPPORT_RESTART_EXHAUSTED
            and not len(energies)
        )
    ):
        status = RunStatus.DIVERGED
    else:
        status = RunStatus.BUDGET_EXHAUSTED
    if reason in (OuterTerminationReason.NOT_RUN, OuterTerminationReason.LEGACY_PICARD):
        raise ValueError("outer outcome lacks an outer termination reason")
    if status is RunStatus.CONVERGED and (
        outcome.last_solve is None
        or not len(energies)
        or not np.isfinite(energies).all()
        or not np.isfinite(deltas).all()
        or not np.isfinite(outcome.residual)
        or outcome.inner_code != 0
        or not np.isfinite(outcome.inner_residual)
    ):
        raise ValueError("converged outer outcome lacks finite solver evidence")
    solved = outcome.last_solve
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
        outer_residual=solved.outer_residual if solved is not None else float("nan"),
        outer_merit=solved.merit if solved is not None else float("nan"),
        outer_gradient_norm=solved.gradient_norm
        if solved is not None
        else float("nan"),
        trust_radius=solved.trust_radius if solved is not None else float("nan"),
        accepted_steps=outcome.accepted_steps,
        rejected_steps=outcome.rejected_steps,
        termination_reason=reason,
        outer_map_evaluations=outcome.outer_map_evaluations,
        trial_evaluations=outcome.trial_evaluations,
        linear_iterations=outcome.linear_iterations,
        solver_compilations=outcome.solver_compilations,
    )

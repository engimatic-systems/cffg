"""Public runtime data and termination codes, independent of execution owners."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import equinox as eqx
import jax


class OuterTerminationReason(str, Enum):
    """Machine-readable detail beneath the public inference status."""

    NOT_RUN = "not_run"
    ROOT_CONVERGED = "root_converged"
    STATIONARY_NONROOT = "stationary_nonroot"
    INNER_FAILURE = "inner_failure"
    NONFINITE_EVALUATION = "nonfinite_evaluation"
    INFEASIBLE = "infeasible"
    SUPPORT_RESTART_EXHAUSTED = "support_restart_exhausted"
    STAGNATION = "stagnation"
    TRUST_REGION_COLLAPSED = "trust_region_collapsed"
    ITERATION_BUDGET = "iteration_budget"
    LINEAR_SOLVE_BUDGET = "linear_solve_budget"
    LEGACY_PICARD = "legacy_picard"


class State(eqx.Module):
    """Pure-data instance state. The Gaussian arrays are zero-size on
    discrete-only models — same pytree structure either way, so colonies mix
    freely. g_obs/g_mask are the *conditioned* Gaussian observations (see
    cffg.gauss: conditioning, not clamping) with mask 0 = missing."""

    messages: jax.Array  # [sum_slot K] log-space discrete segments
    params: dict[str, jax.Array]  # group name -> [M, ...] log-factors
    g_eta: jax.Array  # [sum_slot d] information-vector segments
    g_lam: jax.Array  # [sum_slot d*d] flattened precision blocks
    g_obs: jax.Array  # [sum_observation d] conditioned observation segments
    g_mask: jax.Array  # [n_obs_rows] 1.0 = bound, 0.0 = missing
    ef_eta: jax.Array  # [n_efslots, 2] EF natural parameters (cff-8s6a)
    ef_obs: jax.Array  # [n_ef_obs] conditioned counts/flips
    ef_mask: jax.Array  # [n_ef_obs] 1.0 = bound, 0.0 = missing


class RunStatus(str, Enum):
    """Mutually exclusive public termination outcomes for :meth:`Engine.run`."""

    CONVERGED = "converged"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DIVERGED = "diverged"
    INFEASIBLE = "infeasible"


class InferenceError(RuntimeError):
    """Raised when a caller requires a converged state and did not get one."""

    def __init__(self, result: RunResult):
        super().__init__(
            f"inference {result.status.value} after {result.iterations} iterations "
            f"(residual={result.residual!r})"
        )
        self.result = result


@dataclass(frozen=True)
class FailureDiagnostic:
    """Run-local explanation, not a replacement for the authoritative status.

    ``reason`` and ``phase`` are stable lowercase codes. ``factor`` is the
    serialized model-local name, or None when ownership is unknown.
    Initialization and priming use iteration zero; update attempts start at
    one. A functional failure can precede that attempt's message update.
    Diagnostics are not persisted in agent checkpoints.
    """

    reason: str
    factor: str | None
    phase: str
    iteration: int


@dataclass(frozen=True)
class RunResult:
    """Status-bearing result of an inference run.

    This intentionally is not tuple-compatible: migrating callers must name
    ``state`` and therefore cross the status-bearing API boundary explicitly.
    ``free_energies`` contains only sweeps on which the ledger is defined
    (improper continuous beliefs during propagation have no FE entry).
    """

    state: State
    status: RunStatus
    iterations: int
    message_deltas: tuple[float, ...]
    free_energies: tuple[float, ...]
    residual: float
    inner_status: RunStatus = RunStatus.CONVERGED
    inner_residual: float = 0.0
    mu3_ess: tuple[float, ...] = ()
    mu3_log_normalizers: tuple[float, ...] = ()
    mu3_statuses: tuple[RunStatus, ...] = ()
    outer_residual: float = float("nan")
    outer_merit: float = float("nan")
    outer_gradient_norm: float = float("nan")
    trust_radius: float = float("nan")
    accepted_steps: int = 0
    rejected_steps: int = 0
    termination_reason: OuterTerminationReason = OuterTerminationReason.NOT_RUN
    outer_map_evaluations: int = 0
    trial_evaluations: int = 0
    linear_iterations: int = 0
    solver_compilations: int = 0
    failure: FailureDiagnostic | None = None

    @property
    def delta_trace(self) -> tuple[float, ...]:
        return self.message_deltas

    @property
    def fe_trace(self) -> tuple[float, ...]:
        return self.free_energies

    def require_converged(self) -> State:
        """Return the state only when the run proved convergence."""
        if self.status is not RunStatus.CONVERGED:
            raise InferenceError(self)
        return self.state

    def require_converged_with_trace(self) -> tuple[State, tuple[float, ...]]:
        """Explicit migration helper for the old ``state, deltas`` API."""
        return self.require_converged(), self.message_deltas


@dataclass(frozen=True)
class Mu3ScoreResult:
    """Independent-particle-epoch policy score with explicit key threading."""

    state: State
    status: RunStatus
    epoch_scores: tuple[float, ...]
    policy_score: float
    next_key: jax.Array
    runs: tuple[RunResult, ...]


def _replace(state: State, **kw) -> State:
    return State(
        **{
            f: kw.get(f, getattr(state, f))
            for f in (
                "messages",
                "params",
                "g_eta",
                "g_lam",
                "g_obs",
                "g_mask",
                "ef_eta",
                "ef_obs",
                "ef_mask",
            )
        }
    )

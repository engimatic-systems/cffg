"""One exact schedule execution and its fixed-point certificate."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from cffg._immutable import FrozenArray
from cffg.failure_facts import FailureAt, FailurePhase, empty_failure_at, record_failure
from cffg.runtime_types import State, _replace
from cffg.schedule_contracts import ExactUnit


class ExactReason(IntEnum):
    CERTIFIED = 0
    BUDGET = 1
    INITIAL_NONFINITE = 2
    PRE_EMPTY_SUPPORT = 3
    UPDATE_NONFINITE = 4
    POST_EMPTY_SUPPORT = 5
    OBJECTIVE_NONFINITE = 6


@dataclass(frozen=True)
class ExactOutcome:
    """Exact execution counts need not equal the length of its delta trace."""

    state: State
    reason: ExactReason
    iterations: int
    message_deltas: FrozenArray = ()
    free_energies: FrozenArray = ()
    residual: float = float("inf")
    structural: bool = False
    by_tolerance: bool = False
    failure: FailureAt = field(default_factory=empty_failure_at)

    def __post_init__(self):
        object.__setattr__(self, "reason", ExactReason(self.reason))
        for name in ("message_deltas", "free_energies"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional trace")
            object.__setattr__(self, name, FrozenArray(values))


@dataclass(frozen=True)
class ExactRunStrategy:
    """Consume the fused two-pass/support/ledger unit, without an outer solver.

    Each directed message is computed once, then a certification sweep checks
    every entry against its own magnitude. A true tree fixed point is within
    recomputation rounding, independent of graph diameter or another block's
    scale. A caller tolerance may additionally certify the raw delta.
    """

    unit: ExactUnit
    semiring: str
    finite: Callable[[State], bool]
    objective: Callable[[State], float]

    def run(self, state: State, tol: float | None) -> ExactOutcome:
        if not self.finite(state):
            return ExactOutcome(state, ExactReason.INITIAL_NONFINITE, 0)

        msgs, g_eta, g_lam, ef_eta, delta_dev, rel_dev, fe_dev, pre_bad, post_bad, *auxiliary = (
            self.unit(
                state.messages,
                state.g_eta,
                state.g_lam,
                state.g_obs,
                state.g_mask,
                state.ef_eta,
                state.ef_obs,
                state.ef_mask,
                state.params,
            )
        )
        if bool(pre_bad):
            return ExactOutcome(
                state,
                ExactReason.PRE_EMPTY_SUPPORT,
                0,
                free_energies=(float("inf"),),
            )
        st = _replace(state, messages=msgs, g_eta=g_eta, g_lam=g_lam, ef_eta=ef_eta)
        failure = empty_failure_at()
        if auxiliary:
            failure = record_failure(failure, auxiliary[0], FailurePhase.PROPAGATION, 1)
        delta, rel = float(delta_dev), float(rel_dev)
        if not np.isfinite(delta) or not np.isfinite(rel) or not self.finite(st):
            return ExactOutcome(st, ExactReason.UPDATE_NONFINITE, 1, (delta,), failure=failure)
        if bool(post_bad):
            # Empty support discards the fused sentinel-valued ledger.
            return ExactOutcome(
                st,
                ExactReason.POST_EMPTY_SUPPORT,
                1,
                free_energies=(float("inf"),),
                failure=failure,
            )

        eps = float(np.finfo(np.asarray(st.messages).dtype).eps)
        structural = rel <= 64.0 * eps
        by_tol = tol is not None and delta <= max(float(tol), 64.0 * eps)
        certified = structural or by_tol
        reason = ExactReason.CERTIFIED if certified else ExactReason.BUDGET
        if self.semiring == "max":
            if certified and not np.isfinite(self.objective(st)):
                return ExactOutcome(st, ExactReason.OBJECTIVE_NONFINITE, 1, (delta,), failure=failure)
            energies = ()
        else:
            fe = float(fe_dev)
            if not np.isfinite(fe):
                return ExactOutcome(st, ExactReason.OBJECTIVE_NONFINITE, 1, (delta,), failure=failure)
            energies = (fe,)
        return ExactOutcome(
            st, reason, 1, (delta,), energies, delta, structural, by_tol, failure
        )

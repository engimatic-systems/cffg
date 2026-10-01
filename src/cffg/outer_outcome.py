"""Outer-control evidence without a public inference status decision."""

from dataclasses import dataclass

import numpy as np

from cffg._immutable import FrozenArray
from cffg.outer import OuterSolveResult
from cffg.runtime_types import OuterTerminationReason, State


@dataclass(frozen=True)
class OuterOutcome:
    """Accumulated work plus the last solve's own numerical diagnostics."""

    state: State
    reason: OuterTerminationReason
    iterations: int = 0
    message_deltas: FrozenArray = ()
    free_energies: FrozenArray = ()
    residual: float = float("inf")
    inner_code: int = 0
    inner_residual: float = 0.0
    tolerance_requested: bool = False
    functional_pending: bool = False
    functional_failure: bool = False
    last_solve: OuterSolveResult | None = None
    accepted_steps: int = 0
    rejected_steps: int = 0
    outer_map_evaluations: int = 0
    trial_evaluations: int = 0
    linear_iterations: int = 0
    solver_compilations: int = 0
    initial_rejection: bool = False

    def __post_init__(self):
        for name in ("message_deltas", "free_energies"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional trace")
            object.__setattr__(self, name, FrozenArray(values))

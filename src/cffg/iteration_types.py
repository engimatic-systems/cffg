"""Internal iteration outcomes and array-valued facts, without public statuses."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple

import jax
import numpy as np

from cffg._immutable import FrozenArray
from cffg.failure_facts import FailureAt, FailurePhase, empty_failure_at
from cffg.runtime_types import State

type ScalarArray = jax.Array | np.ndarray


class IterationReason(IntEnum):
    RUNNING = 0
    BUDGET = 1
    FE_STATIONARY = 2
    MAX_STATIONARY = 3
    INITIAL_NONFINITE = 4
    UPDATE_NONFINITE = 5
    EMPTY_SUPPORT = 6
    INNER_BUDGET = 7
    INNER_DIVERGED = 8
    FUNCTIONAL_BUDGET = 9
    FUNCTIONAL_DIVERGED = 10
    OBJECTIVE_NONFINITE = 11
    UNDEFINED_LEDGER = 12
    INITIAL_EMPTY_SUPPORT = 13


class StepFacts(NamedTuple):
    delta: ScalarArray
    inner_residual: ScalarArray
    inner_code: ScalarArray


class TerminationFacts(NamedTuple):
    """Fixed-shape scalar facts; an unchecked state/support predicate is -1."""

    reason: ScalarArray
    residual: ScalarArray
    inner_code: ScalarArray
    inner_residual: ScalarArray
    state_finite: ScalarArray
    zero_support: ScalarArray


@dataclass(frozen=True)
class IterationOutcome:
    """Internal state and evidence; the public result owner chooses RunStatus."""

    state: State
    reason: IterationReason
    message_deltas: FrozenArray
    free_energies: FrozenArray
    residual: float
    inner_code: int
    inner_residual: float
    state_finite: bool | None
    zero_support: bool | None
    failure: FailureAt = field(default_factory=empty_failure_at)
    phase: FailurePhase = FailurePhase.SOFT

    def __post_init__(self):
        object.__setattr__(self, "reason", IterationReason(self.reason))
        for name in ("message_deltas", "free_energies"):
            # Host traces keep Python-float precision even with JAX x64 disabled.
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional trace")
            object.__setattr__(self, name, FrozenArray(values))

    @property
    def iterations(self) -> int:
        return len(self.message_deltas)

    @property
    def initial_rejection(self) -> bool:
        return self.reason in (
            IterationReason.INITIAL_NONFINITE,
            IterationReason.INITIAL_EMPTY_SUPPORT,
        )

    @property
    def numerical_facts(self) -> TerminationFacts:
        """Detached scalar arrays for JAX batching, without Python enum leaves."""
        return TerminationFacts(
            np.asarray(self.reason, dtype=np.int32),
            np.asarray(self.residual, dtype=np.float64),
            np.asarray(self.inner_code, dtype=np.int32),
            np.asarray(self.inner_residual, dtype=np.float64),
            np.asarray(
                -1 if self.state_finite is None else self.state_finite, dtype=np.int8
            ),
            np.asarray(
                -1 if self.zero_support is None else self.zero_support, dtype=np.int8
            ),
        )

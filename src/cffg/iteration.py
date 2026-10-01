"""Ordinary iteration and incremental termination assessment, without statuses."""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from cffg.iteration_types import IterationOutcome, IterationReason, StepFacts
from cffg.runtime_types import State, _replace
from cffg.schedule_program import ScheduleProgram

type StatePredicate = Callable[[State], bool]
type StateObjective = Callable[[State], float]
type FunctionalUpdate = Callable[[State], tuple[State, float, int, float]]


@dataclass(frozen=True)
class IterationPolicy:
    semiring: str
    allow_initial_undefined_energy: bool
    effective_tolerance: float | None

    @classmethod
    def from_dtype(cls, dtype, *, semiring, allow_initial_undefined_energy, tol):
        # Smaller tolerances cannot be falsified at the active precision.
        tolerance = None if tol is None else max(float(tol), 64.0 * np.finfo(dtype).eps)
        return cls(semiring, allow_initial_undefined_energy, tolerance)


@dataclass(frozen=True)
class IterationServices:
    """Trusted host callbacks, not serialized model or State data."""

    finite: StatePredicate
    zero_support: StatePredicate
    free_energy: StateObjective
    objective: StateObjective
    update_functional: FunctionalUpdate


def step_facts(
    old: State, new: State, inner_residual, inner_code, *, functional_delta=0.0
) -> StepFacts:
    """Preserve non-finite changes and host precision in fixed-shape facts."""
    changes = [
        functional_delta,
        *(
            float(jnp.max(jnp.abs(getattr(new, name) - getattr(old, name))))
            for name in ("messages", "g_eta", "g_lam", "ef_eta")
            if getattr(old, name).size
        ),
    ]
    return StepFacts(
        np.asarray(np.max(changes)), np.asarray(inner_residual), np.asarray(inner_code)
    )


class IterationProgress:
    """One instance's host-side progress, shared by standalone and batched loops."""

    def __init__(self, policy: IterationPolicy):
        self.policy = policy
        self.reason = IterationReason.RUNNING
        self.deltas: list[float] = []
        self.energies: list[float] = []
        self.residual = float("inf")
        self.inner_code = 0
        self.inner_residual = 0.0
        self.previous_fe: float | None = None
        self.state_finite: bool | None = None
        self.zero_support: bool | None = None

    @property
    def stopped(self) -> bool:
        return self.reason is not IterationReason.RUNNING

    @property
    def converged(self) -> bool:
        return self.reason in (
            IterationReason.FE_STATIONARY,
            IterationReason.MAX_STATIONARY,
        )

    def invalidate_checks(self) -> None:
        self.state_finite = None
        self.zero_support = None

    def initial(self, state: State, services: IterationServices) -> None:
        self.state_finite = bool(services.finite(state))
        if not self.state_finite:
            self.reason = IterationReason.INITIAL_NONFINITE
            self.inner_code = 2
            self.inner_residual = float("inf")
            return
        self.zero_support = bool(services.zero_support(state))
        if self.zero_support:
            self.reason = IterationReason.INITIAL_EMPTY_SUPPORT
            self.energies.append(float("inf"))

    def begin_phase(self) -> None:
        self.reason = IterationReason.RUNNING
        self.previous_fe = None

    def reject_functional(self, code: int, residual: float) -> None:
        self.invalidate_checks()
        self.inner_code = 1 if code == 1 else 2
        self.inner_residual = residual
        self.residual = residual
        self.reason = (
            IterationReason.FUNCTIONAL_BUDGET
            if code == 1
            else IterationReason.FUNCTIONAL_DIVERGED
        )

    def consume(
        self, state: State, facts: StepFacts, services: IterationServices
    ) -> None:
        delta = float(facts.delta)
        self.deltas.append(delta)
        self.inner_residual = float(facts.inner_residual)
        self.inner_code = int(facts.inner_code)
        self.invalidate_checks()

        if not np.isfinite(delta):
            self.reason = IterationReason.UPDATE_NONFINITE
            self.residual = float("inf")
            return
        self.state_finite = bool(services.finite(state))
        if not self.state_finite or not np.isfinite(self.inner_residual):
            self.reason = IterationReason.UPDATE_NONFINITE
            self.residual = float("inf")
            return
        if self.inner_code not in (0, 1):
            self.reason = IterationReason.INNER_DIVERGED
            self.residual = float("inf")
            return
        self.zero_support = bool(services.zero_support(state))
        if self.zero_support:
            self.reason = IterationReason.EMPTY_SUPPORT
            self.residual = float("inf")
            self.energies.append(float("inf"))
            return
        if self.inner_code == 1:
            self.reason = IterationReason.INNER_BUDGET
            self.residual = self.inner_residual
            return

        tolerance = self.policy.effective_tolerance
        if self.policy.semiring == "max":
            self.residual = delta
            if tolerance is not None and delta <= tolerance:
                if np.isfinite(services.objective(state)):
                    self.reason = IterationReason.MAX_STATIONARY
                else:
                    self.reason = IterationReason.OBJECTIVE_NONFINITE
                    self.residual = float("inf")
            return

        fe = services.free_energy(state)
        if np.isfinite(fe):
            self.energies.append(fe)
            if self.previous_fe is not None:
                self.residual = abs(fe - self.previous_fe)
                if (
                    tolerance is not None
                    and self.residual <= tolerance
                    and delta <= tolerance
                ):
                    self.reason = IterationReason.FE_STATIONARY
                    return
            self.previous_fe = fe
        elif (
            self.previous_fe is not None
            or not self.policy.allow_initial_undefined_energy
        ):
            self.reason = IterationReason.OBJECTIVE_NONFINITE
            self.residual = float("inf")

    def finish(self, state: State) -> IterationOutcome:
        if self.reason is IterationReason.RUNNING:
            self.reason = IterationReason.BUDGET
        if (
            self.reason
            in (
                IterationReason.BUDGET,
                IterationReason.INNER_BUDGET,
                IterationReason.FUNCTIONAL_BUDGET,
            )
            and self.policy.semiring != "max"
            and not self.energies
            and self.deltas
            and self.policy.allow_initial_undefined_energy
        ):
            self.reason = IterationReason.UNDEFINED_LEDGER
            self.residual = float("inf")
        return IterationOutcome(
            state,
            self.reason,
            self.deltas,
            self.energies,
            self.residual,
            self.inner_code,
            self.inner_residual,
            self.state_finite,
            self.zero_support,
        )


def run_iterations(
    state: State,
    program: ScheduleProgram,
    *,
    services: IterationServices,
    policy: IterationPolicy,
    n_iters: int,
    damp: float,
    harden_last: int,
) -> IterationOutcome:
    """Run the existing soft/hard phases and return facts, not a public status."""
    progress = IterationProgress(policy)
    progress.initial(state, services)
    if progress.stopped:
        return progress.finish(state)

    if program.prime is not None and n_iters >= 1:
        msgs, eta, lam, ef_eta, *_ = program.prime(
            state.messages,
            state.g_eta,
            state.g_lam,
            state.ef_eta,
            state.g_obs,
            state.g_mask,
            state.ef_obs,
            state.ef_mask,
            state.params,
        )
        state = _replace(state, messages=msgs, g_eta=eta, g_lam=lam, ef_eta=ef_eta)
        progress.invalidate_checks()

    def phase(current: State, count: int, *, harden: bool) -> State:
        progress.begin_phase()
        for _ in range(count):
            current, functional_delta, code, residual = services.update_functional(
                current
            )
            if code:
                progress.reject_functional(code, residual)
                return current
            msgs, eta, lam, ef_eta, inner_residual, inner_code, *_ = program.iter_step(
                current.messages,
                current.g_eta,
                current.g_lam,
                current.g_obs,
                current.g_mask,
                current.ef_eta,
                current.ef_obs,
                current.ef_mask,
                current.params,
                damp,
                harden,
            )
            candidate = _replace(
                current, messages=msgs, g_eta=eta, g_lam=lam, ef_eta=ef_eta
            )
            facts = step_facts(
                current,
                candidate,
                inner_residual,
                inner_code,
                functional_delta=functional_delta,
            )
            current = candidate
            progress.consume(current, facts, services)
            if progress.stopped:
                break
        return current

    state = phase(state, n_iters - harden_last, harden=False)
    if harden_last and (not progress.stopped or progress.converged):
        state = phase(state, harden_last, harden=True)
    return progress.finish(state)

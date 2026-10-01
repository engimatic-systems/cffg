"""Once-only, budgeted adaptive admission before strict trust-region solving."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from cffg.execution.startup import StartupProgram
from cffg.failure_facts import FailurePhase, FailureReason, empty_failure, first_failure, in_phase
from cffg.outer_chart import OuterArrays, build_outer_chart
from cffg.outer_evidence import terminal_failure
from cffg.outer_outcome import OuterOutcome
from cffg.runtime_types import OuterTerminationReason, State, _replace


@dataclass(frozen=True)
class StartupOutcome:
    state: State
    iterations: int = 0
    deltas: tuple[float, ...] = ()
    terminal: OuterOutcome | None = None


def run_startup(program: StartupProgram, structure, eliminated, services, state, budget):
    """Wait only at neutral, unadmitted sites; never certify convergence here."""
    if budget == 0:
        return StartupOutcome(state)
    eligible = program.eligibility(state.g_eta, state.g_lam)
    admitted = tuple(jnp.zeros_like(row) for row in eligible)
    deltas = []
    iterations = 0

    def terminal(reason, fact, *, code=0, residual=float("inf")):
        outcome = OuterOutcome(
            state, reason, iterations=iterations, message_deltas=deltas,
            inner_code=code, inner_residual=residual,
            failure=terminal_failure(in_phase(fact, FailurePhase.STARTUP), iterations),
        )
        return StartupOutcome(state, iterations, tuple(deltas), outcome)

    while True:
        check = program.inspect(state.g_eta, state.g_lam, state.g_obs, state.g_mask,
                                state.params, eligible, admitted)
        if int(check.failure.reason) != 0:
            if iterations == 0:
                iterations = 1
            return terminal(OuterTerminationReason.INNER_FAILURE, check.failure, code=2)
        arrays = OuterArrays(state.messages, state.g_eta, state.g_lam, state.ef_eta)
        if bool(check.ready) and build_outer_chart(structure, arrays, eliminated).proper_reference:
            return StartupOutcome(state, iterations, tuple(deltas))
        if iterations >= budget:
            pending = check.pending
            if int(pending.reason) == 0:
                pending = empty_failure()._replace(reason=jnp.int32(FailureReason.LOCAL_BUDGET))
            return terminal(OuterTerminationReason.ITERATION_BUDGET, pending, code=1)

        result = program.step(
            state.messages, state.g_eta, state.g_lam, state.g_obs, state.g_mask,
            state.ef_eta, state.ef_obs, state.ef_mask, state.params, eligible, admitted,
        )
        sweep = result.sweep
        state = _replace(state, messages=sweep[0], g_eta=sweep[1], g_lam=sweep[2], ef_eta=sweep[3])
        admitted = result.admitted
        iterations += 1
        deltas.append(max((float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                           for a, b in zip(arrays, sweep[:4], strict=True) if a.size), default=0.))
        fact = first_failure(sweep[6], result.check.failure)
        if int(sweep[5]) != 0 or int(fact.reason) != 0:
            return terminal(OuterTerminationReason.INNER_FAILURE, fact,
                            code=int(sweep[5]) or 2, residual=float(sweep[4]))
        if not services.finite(state):
            return terminal(OuterTerminationReason.NONFINITE_EVALUATION, fact, code=2)
        if services.zero_support(state):
            return terminal(OuterTerminationReason.INFEASIBLE, fact)

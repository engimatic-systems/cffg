"""Decode retained execution evidence once at the public result boundary."""

import jax.numpy as jnp
import numpy as np

from cffg.exact_strategy import ExactReason
from cffg.failure_facts import FailurePhase, FailureReason
from cffg.iteration_types import IterationReason
from cffg.runtime_types import FailureDiagnostic, OuterTerminationReason


def local_diagnostic(retained, names):
    factor, reason, phase, iteration = np.asarray(
        jnp.stack((*retained.fact, retained.iteration))
    ).tolist()
    if reason == FailureReason.NONE:
        return None
    return FailureDiagnostic(
        FailureReason(reason).name.lower(),
        names[factor] if 0 <= factor < len(names) else None,
        FailurePhase(phase).name.lower(),
        iteration,
    )


def iteration_diagnostic(outcome, names):
    local = local_diagnostic(outcome.failure, names)
    if local is not None:
        return local
    phase = outcome.phase.name.lower()
    iteration = outcome.iterations
    reason, phase = {
        IterationReason.INITIAL_NONFINITE: ("nonfinite_state", "initialization"),
        IterationReason.INITIAL_EMPTY_SUPPORT: ("empty_support", "initialization"),
        IterationReason.UPDATE_NONFINITE: ("nonfinite_update", phase),
        IterationReason.EMPTY_SUPPORT: ("empty_support", phase),
        IterationReason.INNER_BUDGET: ("local_budget", phase),
        IterationReason.INNER_DIVERGED: ("local_divergence", phase),
        IterationReason.FUNCTIONAL_BUDGET: ("functional_budget", "functional"),
        IterationReason.FUNCTIONAL_DIVERGED: ("functional_divergence", "functional"),
        IterationReason.OBJECTIVE_NONFINITE: ("nonfinite_objective", "ledger"),
        IterationReason.UNDEFINED_LEDGER: ("undefined_objective", "ledger"),
        IterationReason.BUDGET: ("iteration_budget", phase),
    }[outcome.reason]
    if phase == "functional":
        iteration += 1
    return FailureDiagnostic(reason, None, phase, iteration)


def exact_diagnostic(outcome, names):
    local = local_diagnostic(outcome.failure, names)
    if local is not None:
        return local
    reason, phase = {
        ExactReason.INITIAL_NONFINITE: ("nonfinite_state", "initialization"),
        ExactReason.PRE_EMPTY_SUPPORT: ("empty_support", "initialization"),
        ExactReason.UPDATE_NONFINITE: ("nonfinite_update", "propagation"),
        ExactReason.POST_EMPTY_SUPPORT: ("empty_support", "propagation"),
        ExactReason.OBJECTIVE_NONFINITE: ("nonfinite_objective", "ledger"),
        ExactReason.BUDGET: ("exact_certificate", "certification"),
    }[outcome.reason]
    return FailureDiagnostic(reason, None, phase, outcome.iterations)


def outer_diagnostic(outcome):
    """Only terminal evidence is known here; never invent a rejected-trial cause."""
    if outcome.initial_rejection:
        reason = ("empty_support" if outcome.reason is OuterTerminationReason.INFEASIBLE
                  else "nonfinite_state")
        return FailureDiagnostic(reason, None, "initialization", 0)
    if outcome.functional_failure:
        reason = "functional_budget" if outcome.inner_code == 1 else "functional_divergence"
        return FailureDiagnostic(reason, None, "functional", outcome.iterations + 1)
    return FailureDiagnostic("outer_" + outcome.reason.value, None, "outer", outcome.iterations)

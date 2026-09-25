"""Auxiliary outer-map evidence, independent of numerical solver decisions."""

import jax.numpy as jnp

from cffg.failure_facts import (
    FailureAt, FailureFact, FailurePhase, empty_failure, in_phase,
)


def map_evidence(output):
    """Accept legacy numerical triples and maps with an auxiliary fourth leaf."""
    arrays, residual, code, *auxiliary = output
    return arrays, residual, code, auxiliary[0] if auxiliary else empty_failure()


def terminal_failure(fact: FailureFact, attempt: int) -> FailureAt:
    """Call only at terminal host decisions, never inside derivative traces."""
    return FailureAt(
        in_phase(fact, FailurePhase.OUTER),
        jnp.where(fact.reason != 0, jnp.int32(attempt), jnp.int32(-1)),
    )


def offset_failure(failure: FailureAt, completed: int) -> FailureAt:
    """Translate an inner solve's attempt without turning absence into evidence."""
    return failure._replace(iteration=jnp.where(
        failure.fact.reason != 0, failure.iteration + completed, failure.iteration,
    ))

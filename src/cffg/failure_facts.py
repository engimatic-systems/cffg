"""Auxiliary failure facts; these never decide numerical convergence."""

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp


class FailureReason(IntEnum):
    NONE = 0
    NONFINITE_OUTPUT = 1
    LOCAL_BUDGET = 2
    LOCAL_DIVERGENCE = 3
    IMPROPER_CAVITY = 4
    INVALID_NOISE = 5
    NONFINITE_FUNCTION = 6
    IMPROPER_TILT = 7
    UNREPRESENTABLE_SIGMA_POINTS = 8


class FailurePhase(IntEnum):
    NONE = 0
    INITIALIZATION = 1
    PRIMING = 2
    PROPAGATION = 3
    CERTIFICATION = 4
    SOFT = 5
    HARD = 6
    OUTER = 7
    RECONSTRUCTION = 8
    LEDGER = 9


class FailureFact(NamedTuple):
    """Scalar int32 leaves, or stacked leaves under vmap.

    Factor identifiers are ranks in a stable model-local name table. A
    reason of NONE is the only empty-record predicate; factor -1 denotes
    unavailable ownership. These facts are run-local, never serialized.
    """

    factor: jax.Array
    reason: jax.Array
    phase: jax.Array


def empty_failure() -> FailureFact:
    return FailureFact(
        jnp.asarray(-1, jnp.int32),
        jnp.asarray(FailureReason.NONE, jnp.int32),
        jnp.asarray(FailurePhase.NONE, jnp.int32),
    )


def first_failure(earlier: FailureFact, later: FailureFact) -> FailureFact:
    """Keep chronology, even when the later factor has a smaller name rank."""
    return FailureFact(*(
        jnp.where(earlier.reason != FailureReason.NONE, old, new)
        for old, new in zip(earlier, later, strict=True)
    ))


def in_phase(fact: FailureFact, phase: FailurePhase) -> FailureFact:
    """Label a local cause once; enclosing stages cannot replace its phase."""
    unset = (fact.reason != FailureReason.NONE) & (fact.phase == FailurePhase.NONE)
    return fact._replace(phase=jnp.where(unset, jnp.int32(phase), fact.phase))


def nonfinite_rows(values: jax.Array, *, log_support: bool = False) -> jax.Array:
    """Reduce message coordinates, retaining the leading factor-member axis.

    Negative infinity is valid exact zero support for categorical log messages.
    It is not a valid Gaussian information coordinate.
    """
    if values.ndim < 1:
        raise ValueError("outgoing values require a leading member axis")
    invalid = jnp.isnan(values) | jnp.isposinf(values)
    if not log_support:
        invalid |= jnp.isneginf(values)
    return jnp.any(invalid, axis=tuple(range(1, values.ndim)))


def local_failure(
    factor_ids: jax.Array,
    *,
    status: jax.Array | None = None,
    reasons: jax.Array | None = None,
    nonfinite: jax.Array | None = None,
) -> FailureFact:
    """Select the lowest factor rank among simultaneous failed updates.

    Inputs have one entry per member. Structured reasons apply only when that
    member reports a failed status; informational codes on successful updates
    cannot become failures. A structured cause takes precedence over its
    downstream non-finite outgoing values. The existing status codes remain
    authoritative: 0 succeeded, 1 exhausted, and 2 diverged.
    """
    factor_ids = jnp.asarray(factor_ids, jnp.int32)
    if factor_ids.ndim != 1:
        raise ValueError("factor identifiers require one member axis")
    for name, value in (("status", status), ("reasons", reasons), ("nonfinite", nonfinite)):
        if value is not None and value.shape != factor_ids.shape:
            raise ValueError(f"{name} must have one value per factor member")
    if not factor_ids.size:
        return empty_failure()
    status = jnp.zeros_like(factor_ids) if status is None else status
    reason = jnp.where(
        status == 1, FailureReason.LOCAL_BUDGET,
        jnp.where(status != 0, FailureReason.LOCAL_DIVERGENCE, FailureReason.NONE),
    ).astype(jnp.int32)
    if nonfinite is not None:
        reason = jnp.where(nonfinite, FailureReason.NONFINITE_OUTPUT, reason)
    if reasons is not None:
        reason = jnp.where(
            (status != 0) & (reasons != FailureReason.NONE), reasons, reason
        ).astype(jnp.int32)
    failed = reason != FailureReason.NONE
    index = jnp.argmin(jnp.where(failed, factor_ids, jnp.iinfo(jnp.int32).max))
    return FailureFact(
        jnp.where(jnp.any(failed), factor_ids[index], jnp.int32(-1)),
        reason[index],
        jnp.asarray(FailurePhase.NONE, jnp.int32),
    )

"""Typed outputs and numerical facts at the local-update boundary."""

from typing import NamedTuple

import jax


class ExecutionDelta[T](NamedTuple):
    """One plan's outgoing blocks and its existing local solver facts.

    Numerical facts stay arrays for JAX transformations. ``None`` means the
    rule supplies no such fact, not that a solver converged. Exact support
    remains encoded in the outgoing blocks; a successful solver status does
    not replace downstream support checks. The output structure is static for
    each compiled plan, including rules that return no solver diagnostics.
    ``failure_reason`` optionally carries one ``FailureReason`` code per
    member. It is auxiliary provenance, meaningful only for a failed status;
    it never replaces the status or a support check.
    """

    outgoing: T
    residual: jax.Array | None = None
    status: jax.Array | None = None
    failure_reason: jax.Array | None = None


class GaussianMessage(NamedTuple):
    eta: jax.Array
    precision: jax.Array


class HybridMessages(NamedTuple):
    selector: jax.Array
    gaussian: tuple[GaussianMessage, ...]


class RecurrenceMessages(NamedTuple):
    destination: jax.Array
    source: jax.Array
    drive: GaussianMessage


class PrecisionMessages(NamedTuple):
    gaussian: GaussianMessage
    precision_parameter: jax.Array


class DirichletMessages(NamedTuple):
    structural: jax.Array
    prior: jax.Array
    counts: jax.Array


class GfeMessages(NamedTuple):
    latent: jax.Array
    log_prediction: jax.Array
    goal_prior: jax.Array | None
    goal_prediction: jax.Array | None


class TransitionMessages(NamedTuple):
    destination: jax.Array
    source: jax.Array
    selector: jax.Array


class FunctionalEstimate(NamedTuple):
    mean: jax.Array
    column_entropy: jax.Array
    weights: jax.Array
    log_normalizer: jax.Array
    expected_phi: jax.Array
    kl: jax.Array
    ess: jax.Array

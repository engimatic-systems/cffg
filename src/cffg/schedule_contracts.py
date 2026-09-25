"""Callable contracts for static schedule programs, independent of Engine."""

from collections.abc import Mapping
from typing import Protocol

import jax

from cffg.failure_facts import FailureFact

type Parameters = Mapping[str, jax.Array]
type MessageArrays = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
type PropagationResult = tuple[jax.Array, jax.Array, jax.Array, jax.Array, FailureFact]
type SweepResult = tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, FailureFact
]
type ExactResult = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    FailureFact,
]


class Sweep(Protocol):
    """Message arrays, unchanged solver facts, and auxiliary failure provenance."""

    def __call__(
        self,
        messages: jax.Array,
        g_eta: jax.Array,
        g_lam: jax.Array,
        g_obs: jax.Array,
        g_mask: jax.Array,
        ef_eta: jax.Array,
        ef_obs: jax.Array,
        ef_mask: jax.Array,
        params: Parameters,
        damp: float = 0.0,
        harden: bool = True,
    ) -> SweepResult: ...


class Propagation(Protocol):
    """Ordered messages and first cause, without damping or termination policy."""

    def __call__(
        self,
        messages: jax.Array,
        g_eta: jax.Array,
        g_lam: jax.Array,
        ef_eta: jax.Array,
        g_obs: jax.Array,
        g_mask: jax.Array,
        ef_obs: jax.Array,
        ef_mask: jax.Array,
        params: Parameters,
    ) -> PropagationResult: ...


class FreeEnergy(Protocol):
    """Objective callback supplied by the ledger owner."""

    def __call__(
        self,
        messages: jax.Array,
        g_eta: jax.Array,
        g_lam: jax.Array,
        g_obs: jax.Array,
        g_mask: jax.Array,
        ef_eta: jax.Array,
        ef_obs: jax.Array,
        ef_mask: jax.Array,
        params: Parameters,
    ) -> jax.Array: ...


class ExactUnit(Protocol):
    """Messages, residuals, objective and support facts; not public statuses."""

    def __call__(
        self,
        messages: jax.Array,
        g_eta: jax.Array,
        g_lam: jax.Array,
        g_obs: jax.Array,
        g_mask: jax.Array,
        ef_eta: jax.Array,
        ef_obs: jax.Array,
        ef_mask: jax.Array,
        params: Parameters,
    ) -> ExactResult: ...

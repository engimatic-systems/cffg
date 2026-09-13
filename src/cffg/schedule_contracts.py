"""Callable contracts for static schedule programs, independent of Engine."""

from collections.abc import Mapping
from typing import Protocol

import jax

type Parameters = Mapping[str, jax.Array]
type MessageArrays = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
type SweepResult = tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
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
]


class Sweep(Protocol):
    """One update returning message arrays and unchanged inner solver facts."""

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
    """Ordered stages, without damping or a public termination decision."""

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
    ) -> MessageArrays: ...


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

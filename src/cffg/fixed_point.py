"""The adaptive residual and energy composition shared by solving and training."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from cffg.outer import OuterArrays, OuterChart, chart_arrays, chart_coordinates
from cffg.runtime_types import State
from cffg.schedule_contracts import Propagation, Sweep
from cffg.failure_facts import empty_failure
from cffg.outer_coordinates import ChartMetrics


class RootContext(NamedTuple):
    base: OuterArrays
    g_obs: jax.Array
    g_mask: jax.Array
    ef_obs: jax.Array
    ef_mask: jax.Array
    params: Mapping[str, jax.Array]
    hard_values: tuple[jax.Array, ...] = ()
    metrics: ChartMetrics | None = None


def root_context(chart: OuterChart, state: State, hard_values=()) -> RootContext:
    return RootContext(chart.base, state.g_obs, state.g_mask, state.ef_obs,
                       state.ef_mask, state.params, tuple(hard_values), chart.metrics)


@dataclass(frozen=True)
class RootProgram:
    """Fixed chart, dynamic model context, and complete approximation dependence."""

    chart: OuterChart
    iter_step: Sweep
    reconstruct: Propagation | None
    hard_indices: tuple[tuple[np.ndarray, np.ndarray], ...] = ()

    def reconstruct_arrays(self, arrays, context: RootContext):
        if self.reconstruct is None:
            return arrays, empty_failure()
        values = self.reconstruct(
            arrays.messages, arrays.g_eta, arrays.g_lam, arrays.ef_eta,
            context.g_obs, context.g_mask, context.ef_obs, context.ef_mask, context.params,
        )
        return OuterArrays(*values[:4]), values[4] if len(values) > 4 else empty_failure()

    def arrays(self, value, context: RootContext) -> OuterArrays:
        arrays = chart_arrays(self.chart, value, context.base, context.metrics)
        return self.reconstruct_arrays(arrays, context)[0]

    def map_arrays(self, arrays, context: RootContext):
        messages, g_eta, g_lam, ef_eta, residual, status, *auxiliary = self.iter_step(
            arrays.messages, arrays.g_eta, arrays.g_lam, context.g_obs, context.g_mask,
            arrays.ef_eta, context.ef_obs, context.ef_mask, context.params, 0.0, False,
        )
        for (i0, i1), hot in zip(self.hard_indices, context.hard_values, strict=True):
            messages = messages.at[i0].set(hot).at[i1].set(hot)
        return (OuterArrays(messages, g_eta, g_lam, ef_eta), residual, status,
                auxiliary[0] if auxiliary else empty_failure())

    def update(self, value, context: RootContext):
        return self.map_arrays(self.arrays(value, context), context)[:3]

    def array_energy(self, arrays, context: RootContext, belief_kernel: Callable):
        return belief_kernel(
            arrays.messages, arrays.g_eta, arrays.g_lam, context.g_obs, context.g_mask,
            arrays.ef_eta, context.ef_obs, context.ef_mask, context.params,
        )

    def residual(self, value, context: RootContext):
        mapped, _, _ = self.update(value, context)
        return chart_coordinates(self.chart, mapped, context.metrics) - value

    def energy(self, value, context: RootContext, belief_kernel: Callable):
        arrays = self.arrays(value, context)
        return self.array_energy(arrays, context, belief_kernel)

    def certificate(self, value, context: RootContext, belief_kernel: Callable):
        current = self.arrays(value, context)
        mapped, local_residual, local_status, _ = self.map_arrays(current, context)
        mapped_value = chart_coordinates(self.chart, mapped, context.metrics)
        successor = self.arrays(mapped_value, context)
        residual = jnp.max(jnp.abs(mapped_value - value) / (1 + jnp.abs(value)), initial=0.)
        return (self.array_energy(current, context, belief_kernel),
                self.array_energy(successor, context, belief_kernel),
                residual, local_residual, local_status, mapped.messages)

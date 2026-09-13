"""Lazy derivative kernels keyed by structural charts, with dynamic context."""

from collections.abc import Callable
from typing import Protocol

import jax
import numpy as np

from cffg.outer import (
    OuterArrays,
    OuterChart,
    OuterKernelPlan,
    _make_root_diagnostics,
    _make_trust_kernel,
    chart_arrays,
    chart_coordinates,
    outer_chart_signature,
)
from cffg.runtime_types import State
from cffg.schedule_contracts import Propagation, Sweep

type HardBlock = tuple[np.ndarray, np.ndarray, jax.Array]


class KernelPlanProvider(Protocol):
    def __call__(
        self, chart: OuterChart, state: State, hard_blocks: list[HardBlock]
    ) -> OuterKernelPlan: ...


class OuterKernelProvider:
    """Per-engine cache; schedule handles remain current at kernel tracing time."""

    def __init__(self):
        self.cache: dict[tuple, tuple[Callable, Callable]] = {}
        self.iter_step: Sweep
        self.reconstruct: Propagation | None = None

    def plan(
        self,
        chart: OuterChart,
        state: State,
        hard_blocks: list[HardBlock],
        *,
        iter_step: Sweep,
        reconstruct: Propagation | None,
    ) -> OuterKernelPlan:
        """Reuse derivative kernels while keeping every runtime value dynamic."""
        self.iter_step = iter_step
        self.reconstruct = reconstruct
        hard_indices = tuple(
            (
                np.asarray(i0).dtype.str,
                np.asarray(i0).shape,
                np.asarray(i0).tobytes(),
                np.asarray(i1).dtype.str,
                np.asarray(i1).shape,
                np.asarray(i1).tobytes(),
            )
            for i0, i1, _ in hard_blocks
        )
        key = outer_chart_signature(chart), hard_indices
        compiled = key not in self.cache
        if compiled:
            static_indices = tuple((i0, i1) for i0, i1, _ in hard_blocks)

            def residual(value, context):
                (
                    base,
                    g_obs,
                    g_mask,
                    ef_obs,
                    ef_mask,
                    params,
                    hard_values,
                ) = context
                arrays = chart_arrays(chart, value, base)
                if self.reconstruct is not None:
                    messages, g_eta, g_lam, ef_eta = self.reconstruct(
                        arrays.messages,
                        arrays.g_eta,
                        arrays.g_lam,
                        arrays.ef_eta,
                        g_obs,
                        g_mask,
                        ef_obs,
                        ef_mask,
                        params,
                    )
                    arrays = OuterArrays(messages, g_eta, g_lam, ef_eta)
                messages, g_eta, g_lam, ef_eta, _, _ = self.iter_step(
                    arrays.messages,
                    arrays.g_eta,
                    arrays.g_lam,
                    g_obs,
                    g_mask,
                    arrays.ef_eta,
                    ef_obs,
                    ef_mask,
                    params,
                    0.0,
                    False,
                )
                for (i0, i1), hot in zip(static_indices, hard_values):
                    messages = messages.at[i0].set(hot).at[i1].set(hot)
                mapped = OuterArrays(messages, g_eta, g_lam, ef_eta)
                return chart_coordinates(chart, mapped) - value

            self.cache[key] = (
                _make_trust_kernel(residual, chart.size),
                _make_root_diagnostics(residual),
            )
        trust, root_diagnostics = self.cache[key]
        context = (
            chart.base,
            state.g_obs,
            state.g_mask,
            state.ef_obs,
            state.ef_mask,
            state.params,
            tuple(hot for _, _, hot in hard_blocks),
        )
        return OuterKernelPlan(trust, root_diagnostics, context, compiled)

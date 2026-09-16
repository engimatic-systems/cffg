"""Lazy derivative kernels keyed by structural charts, with dynamic context."""

from collections.abc import Callable
from typing import Protocol

import jax
import numpy as np

from cffg.fixed_point import RootProgram, root_context
from cffg.outer import (
    OuterChart,
    OuterKernelPlan,
    _make_root_diagnostics,
    _make_trust_kernel,
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

            program = RootProgram(chart, self.iter_step, self.reconstruct, static_indices)
            self.cache[key] = (
                _make_trust_kernel(program.residual, chart.size),
                _make_root_diagnostics(program.residual),
            )
        trust, root_diagnostics = self.cache[key]
        context = root_context(chart, state, tuple(hot for _, _, hot in hard_blocks))
        return OuterKernelPlan(trust, root_diagnostics, context, compiled)

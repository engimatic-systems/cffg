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
)
from cffg.runtime_types import State
from cffg.outer_coordinates import structural_signature
from cffg.adaptive_proposals import make_adaptive_proposals
from cffg.outer_batch import ChartKernel
from cffg.schedule_contracts import Propagation, Sweep

type HardBlock = tuple[np.ndarray, np.ndarray, jax.Array]


def _hard_signature(blocks):
    def index_key(array):
        values = np.asarray(array)
        return values.dtype.str, values.shape, values.tobytes()

    return tuple(index_key(i0) + index_key(i1) for i0, i1, _ in blocks)


class KernelPlanProvider(Protocol):
    def __call__(
        self, chart: OuterChart, state: State, hard_blocks: list[HardBlock]
    ) -> OuterKernelPlan: ...


class OuterKernelProvider:
    """Per-engine cache; schedule handles remain current at kernel tracing time."""

    def __init__(self):
        self.cache: dict[tuple, tuple[Callable, Callable, Callable | None]] = {}
        self.stages = {}
        self.iter_step: Sweep
        self.reconstruct: Propagation | None = None

    def stage(self, chart, state, hard_blocks, *, iter_step, reconstruct, capacity, trust=None, ledger=None):
        from cffg.outer_candidate import make_candidate
        from cffg.outer_fixed import make_fixed_controller

        indices = tuple((i0, i1) for i0, i1, _ in hard_blocks)
        hard_key = _hard_signature(hard_blocks)
        key = (structural_signature(chart), hard_key, id(iter_step), id(reconstruct), capacity, trust, ledger)
        if key not in self.stages:
            program = RootProgram(chart, iter_step, reconstruct, indices)
            operation = (make_candidate(program, capacity) if trust is None
                         else make_fixed_controller(program, trust, ledger, capacity))
            self.stages[key] = ChartKernel.build(operation)
        context = root_context(chart, state, tuple(hot for _, _, hot in hard_blocks))
        return self.stages[key], context

    def plan(
        self,
        chart: OuterChart,
        state: State,
        hard_blocks: list[HardBlock],
        *,
        iter_step: Sweep,
        reconstruct: Propagation | None,
        belief_kernel: Callable | None = None,
    ) -> OuterKernelPlan:
        """Reuse derivative kernels while keeping every runtime value dynamic."""
        self.iter_step = iter_step
        self.reconstruct = reconstruct
        hard_indices = _hard_signature(hard_blocks)
        key = structural_signature(chart), hard_indices, id(iter_step), id(reconstruct), id(belief_kernel)
        compiled = key not in self.cache
        if compiled:
            static_indices = tuple((i0, i1) for i0, i1, _ in hard_blocks)

            program = RootProgram(chart, self.iter_step, self.reconstruct, static_indices)
            self.cache[key] = (
                _make_trust_kernel(program.residual, chart.size),
                _make_root_diagnostics(program.residual),
                None if belief_kernel is None else make_adaptive_proposals(program, belief_kernel),
            )
        trust, root_diagnostics, proposal = self.cache[key]
        context = root_context(chart, state, tuple(hot for _, _, hot in hard_blocks))
        return OuterKernelPlan(trust, root_diagnostics, context, compiled, proposal)

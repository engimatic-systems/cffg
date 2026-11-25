"""Run independent fixed-chart stages together at host-controlled phase boundaries."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import jax
import numpy as np


@dataclass(frozen=True)
class ChartKernel:
    one: Callable
    many: Callable

    @classmethod
    def build(cls, operation):
        return cls(jax.jit(operation), jax.jit(jax.vmap(operation)))


@dataclass(frozen=True)
class ChartCall:
    kernel: ChartKernel
    arguments: tuple


def run_phases(members):
    """Advance each member to its next device call, keeping result ownership."""
    results = [None] * len(members)
    pending = [(i, None) for i in range(len(members))]
    while pending:
        groups = defaultdict(list)
        for index, reply in pending:
            try:
                call = members[index].send(reply)
            except StopIteration as finished:
                results[index] = finished.value
            else:
                groups[call.kernel].append((index, call.arguments))
        pending = []
        for kernel, group in groups.items():
            if len(group) == 1:
                index, arguments = group[0]
                pending.append((index, kernel.one(*arguments)))
                continue
            inputs = jax.tree.map(lambda *rows: np.stack(rows), *(args for _, args in group))
            rows = jax.device_get(kernel.many(*inputs))
            for i, (index, _) in enumerate(group):
                pending.append((index, jax.tree.map(lambda values: values[i], rows)))
    return results

"""Host labels and signed contributions for the optional free-energy readout."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class EnergyEntry:
    owner_kind: str
    owner_name: str
    total: float
    components: Mapping[str, float]
    bound_direction: str | None = None


@dataclass(frozen=True)
class FreeEnergyBreakdown:
    """Only the owner totals are additive. Components explain their owner."""

    total: float
    entries: tuple[EnergyEntry, ...]


class EnergyRows:
    """Collect already-computed rows while tracing the detailed JAX kernel."""

    def __init__(self, arena):
        self.blocks = {}
        self.edge_names = {}
        for name, edge in arena.edges.items():
            if edge.interior and not edge.param_shape:
                family = "ef" if edge.ef else "gaussian" if edge.dim else "categorical"
                self.edge_names[family, edge.slot0] = name

    def add(self, kind, names, values, *, components=None, direction=""):
        key = (len(self.blocks), kind, tuple(names), direction)
        self.blocks[key] = (jnp.atleast_1d(values), components or {})

    def edges(self, family, pairs, values):
        names = [self.edge_names[family, int(pair[0])] for pair in pairs]
        self.add("edge_entropy", names, values)


def sum_nodes(group, values, report, *, kind="factor", direction=""):
    if report is not None:
        report.add(kind, group.members, values, direction=direction)
    return values.sum()


def sum_edges(family, pairs, values, report):
    if report is not None:
        report.edges(family, pairs, values)
    return values.sum()


def unpack(total, blocks):
    entries = []
    for (_, kind, names, direction), (values, parts) in sorted(blocks.items()):
        values = np.asarray(values)
        parts = {label: np.asarray(items) for label, items in parts.items()}
        for i, name in enumerate(names):
            components = MappingProxyType({label: float(items[i]) for label, items in parts.items()})
            entries.append(EnergyEntry(kind, name, float(values[i]), components, direction or None))
    return FreeEnergyBreakdown(float(total), tuple(entries))


def infeasible():
    proof = EnergyEntry("support", "empty_exact_support", float("inf"), MappingProxyType({}))
    return FreeEnergyBreakdown(float("inf"), (proof,))

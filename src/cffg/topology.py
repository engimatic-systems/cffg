"""Declared graph incidence and storage layout, before factor interpretation.

The caller expands plates and performs core validation. This owner records
declared domains and wiring, not conditioning, parameter-table meaning,
factor admission, or free-energy ownership.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from cffg._immutable import ImmutableRecord, freeze
from cffg.compiled import EdgeInfo
from cffg.core.ir import (
    ConstraintSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)

Endpoint = tuple[str, str]


@dataclass(frozen=True)
class FactorDeclaration(ImmutableRecord):
    """Read-only role bindings copied from a core FactorSpec, not a new type."""

    name: str
    type: str
    edges: Mapping[str, str]
    param_edges: Mapping[str, str]
    param_blobs: Mapping[str, str]

    def all_edges(self) -> dict[str, str]:
        return {**self.edges, **self.param_edges}


@dataclass(frozen=True)
class DataCandidate(ImmutableRecord):
    """A declared exterior DATA port, without an observation-row assignment."""

    port: str
    variable: str
    endpoint: Endpoint
    external_slot: int


@dataclass(frozen=True)
class TopologyLayout(ImmutableRecord):
    """Immutable declaration snapshot and three independent logical layouts.

    Parameter carriers retain their declared discrete cardinality. Later
    semantic discovery attaches shape/axis tags to new edge records; this
    snapshot never acquires those tags or observation-row assignments.
    """

    variables: Mapping[str, VariableSpec]
    factors: Mapping[str, FactorDeclaration]
    constraints: tuple[ConstraintSpec, ...]
    ports: Mapping[str, PortSpec]
    endpoints: Mapping[str, tuple[Endpoint, ...]]
    degrees: Mapping[str, int]
    ports_by_variable: Mapping[str, tuple[PortSpec, ...]]
    edges: Mapping[str, EdgeInfo]
    slot_of: Mapping[Endpoint, int]
    opposite: Mapping[Endpoint, int]
    n_slots: int
    n_gslots: int
    n_efslots: int
    message_lengths: tuple[int, ...]
    g_eta_lengths: tuple[int, ...]
    g_lam_lengths: tuple[int, ...]
    data_candidates: tuple[DataCandidate, ...]
    port_problems: tuple[str, ...]


def prefix_offsets(lengths: tuple[int, ...] | np.ndarray) -> np.ndarray:
    """Offsets for consecutive variable-width segments, including the end."""
    return np.concatenate([
        np.zeros(1, dtype=np.int64), np.cumsum(np.asarray(lengths, dtype=np.int64)),
    ])


def build_topology(template: Template) -> TopologyLayout:
    """Lay out an already expanded, core-validated template.

    Executable-port diagnostics are returned in declaration order. Compiler
    orchestration appends them after factor admission at its existing error
    gate; moving graph construction must not move observable error timing.
    """
    factors = {
        name: FactorDeclaration(
            factor.name, factor.type, factor.edges, factor.param_edges, factor.param_blobs,
        )
        for name, factor in template.factors.items()
    }
    constraints = tuple(
        replace(constraint, detail=freeze(constraint.detail))
        for constraint in template.constraints
    )
    endpoints: dict[str, list[Endpoint]] = {name: [] for name in template.variables}
    for factor in sorted(factors.values(), key=lambda factor: factor.name):
        for role, variable in sorted(factor.all_edges().items()):
            endpoints[variable].append((factor.name, role))
    degrees = {name: len(ends) for name, ends in endpoints.items()}
    ports_by_variable: dict[str, list[PortSpec]] = {name: [] for name in template.variables}
    port_problems = []
    for port in template.ports.values():
        ports_by_variable[port.variable].append(port)
        if port.role in (PortRole.PRIOR, PortRole.GOAL):
            port_problems.append(
                f"port {port.name!r} ({port.role.value}) is unterminated: compose the missing "
                "factor before compiling"
            )

    edges, slot_of, opposite = {}, {}, {}
    data_candidates = []
    next_slot = next_gslot = next_efslot = 0
    for name in sorted(template.variables):
        variable, ends = template.variables[name], endpoints[name]
        if variable.dim:
            s0, s1 = next_gslot, next_gslot + 1
            next_gslot += 2
        elif variable.ef:
            s0, s1 = next_efslot, next_efslot + 1
            next_efslot += 2
        else:
            s0, s1 = next_slot, next_slot + 1
            next_slot += 2
        edges[name] = EdgeInfo(
            name, variable.cardinality, s0, s1, len(ends) == 2,
            variable.dim, variable.ef,
        )
        slot_of[ends[0]] = s0
        opposite[ends[0]] = s1
        if len(ends) == 2:
            slot_of[ends[1]] = s1
            opposite[ends[1]] = s0
        else:
            data_candidates.extend(
                DataCandidate(port.name, name, ends[0], s1)
                for port in ports_by_variable[name] if port.role is PortRole.DATA
            )

    # Keep declared lengths as Python values. int64 conversion belongs at the
    # existing assembly gate, after semantic and array-resolution diagnostics.
    message_lengths = []
    g_eta_lengths = []
    g_lam_lengths = []
    for edge in edges.values():
        if edge.dim:
            g_eta_lengths.extend((edge.dim, edge.dim))
            g_lam_lengths.extend((edge.dim * edge.dim, edge.dim * edge.dim))
        elif not edge.ef:
            message_lengths.extend((edge.card, edge.card))

    return TopologyLayout(
        variables=template.variables, factors=factors, constraints=constraints,
        ports=template.ports, endpoints=endpoints, degrees=degrees,
        ports_by_variable=ports_by_variable, edges=edges, slot_of=slot_of,
        opposite=opposite, n_slots=next_slot, n_gslots=next_gslot,
        n_efslots=next_efslot, message_lengths=message_lengths,
        g_eta_lengths=g_eta_lengths, g_lam_lengths=g_lam_lengths,
        data_candidates=data_candidates, port_problems=port_problems,
    )

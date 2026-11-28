"""Declared graph incidence and storage layout, before factor interpretation.

Flat templates and compact plates supply resolved endpoints to the same slot
allocator. Factor admission, conditioning and parameter layout run afterward.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from cffg._immutable import ImmutableRecord, freeze
from cffg.compiled import EdgeInfo
from cffg.compiler_errors import CompileError
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
    function: object = None

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
class ExteriorPort:
    name: str
    variable: str
    role: PortRole


@dataclass(frozen=True)
class BoundConstraint:
    declaration: ConstraintSpec
    target: str

    @property
    def kind(self):
        return self.declaration.kind

    @property
    def detail(self):
        return self.declaration.detail

    def detail_dict(self):
        return self.declaration.detail_dict()


@dataclass(frozen=True)
class TopologyLayout(ImmutableRecord):
    """Immutable declaration snapshot and three independent logical layouts.

    Parameter carriers retain their declared discrete cardinality. Later
    semantic discovery attaches shape/axis tags to new edge records; this
    snapshot never acquires those tags or observation-row assignments.
    """

    variables: Mapping[str, VariableSpec | EdgeInfo]
    factors: Mapping[str, FactorDeclaration]
    constraints: tuple[ConstraintSpec | BoundConstraint, ...]
    ports: Mapping[str, PortSpec | ExteriorPort]
    endpoints: Mapping[str, tuple[Endpoint, ...]]
    degrees: Mapping[str, int]
    ports_by_variable: Mapping[str, tuple[PortSpec | ExteriorPort, ...]]
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


def validate_slot_addresses(layout: TopologyLayout) -> None:
    """Check generated bindings against the domain's layout before packing arrays."""
    if len(layout.message_lengths) != layout.n_slots:
        raise CompileError("discrete slot count differs from the message layout")
    if len(layout.g_eta_lengths) != layout.n_gslots or len(layout.g_lam_lengths) != layout.n_gslots:
        raise CompileError("Gaussian slot count differs from the natural parameter layout")
    for name, edge in layout.edges.items():
        count = layout.n_slots
        if edge.dim:
            count = layout.n_gslots
        elif edge.ef:
            count = layout.n_efslots
        if not (0 <= edge.slot0 < count and 0 <= edge.slot1 < count) or edge.slot0 == edge.slot1:
            raise CompileError(f"edge {name!r} has invalid slots {(edge.slot0, edge.slot1)} for {count} slots")
        for index, endpoint in enumerate(layout.endpoints[name]):
            outgoing = edge.slot0 if index == 0 else edge.slot1
            incoming = edge.slot1 if index == 0 else edge.slot0
            if layout.slot_of.get(endpoint) != outgoing or layout.opposite.get(endpoint) != incoming:
                raise CompileError(f"endpoint {endpoint!r} does not address its edge {name!r}")
    for candidate in layout.data_candidates:
        declared = layout.ports.get(candidate.port)
        edge = layout.edges.get(candidate.variable)
        if (declared is None or declared.role is not PortRole.DATA or declared.variable != candidate.variable or edge is None
                or layout.endpoints[candidate.variable] != (candidate.endpoint,)
                or candidate.external_slot != edge.slot1):
            raise CompileError(
                f"DATA port {candidate.port!r} has invalid external slot {candidate.external_slot} "
                f"for edge {candidate.variable!r}"
            )


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
            factor.function,
        )
        for name, factor in template.factors.items()
    }
    constraints = tuple(
        replace(constraint, detail=freeze(constraint.detail))
        for constraint in template.constraints
    )
    return layout_addresses(template.variables, factors, constraints, template.ports)


def layout_addresses(domains, factors, constraints, ports, *, use_edge_domains=False):
    """Assign slots in lexical order to resolved, fully connected factor declarations."""
    endpoints: dict[str, list[Endpoint]] = {name: [] for name in domains}
    for factor in sorted(factors.values(), key=lambda factor: factor.name):
        for role, variable in sorted(factor.all_edges().items()):
            endpoints[variable].append((factor.name, role))
    degrees = {name: len(ends) for name, ends in endpoints.items()}
    ports_by_variable: dict[str, list[PortSpec | ExteriorPort]] = {name: [] for name in domains}
    port_problems = []
    for port in ports.values():
        ports_by_variable[port.variable].append(port)
        if port.role in (PortRole.PRIOR, PortRole.GOAL):
            port_problems.append(
                f"port {port.name!r} ({port.role.value}) is unterminated: compose the missing "
                "factor before compiling"
            )

    edges, slot_of, opposite = {}, {}, {}
    data_candidates = []
    next_slot = next_gslot = next_efslot = 0
    for name in sorted(domains):
        variable, ends = domains[name], endpoints[name]
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

    if use_edge_domains:
        edges = freeze(edges)
    return TopologyLayout(
        variables=edges if use_edge_domains else domains, factors=factors, constraints=constraints,
        ports=ports, endpoints=endpoints, degrees=degrees,
        ports_by_variable=ports_by_variable, edges=edges, slot_of=slot_of,
        opposite=opposite, n_slots=next_slot, n_gslots=next_gslot,
        n_efslots=next_efslot, message_lengths=message_lengths,
        g_eta_lengths=g_eta_lengths, g_lam_lengths=g_lam_lengths,
        data_candidates=data_candidates, port_problems=port_problems,
    )

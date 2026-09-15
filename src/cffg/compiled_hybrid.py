"""Immutable D12 ownership contracts, without numerical state.

Serialized declarations select the approximation. These derived records map
their named owners to indexes and ordinary slots. Parameter members refer to
``Arena.member_of`` and its existing addresses into ``Arena.log_params``;
accepted beliefs and learning statistics belong only to future runtime State.
Construction checks record identity, not graph admission or resolved arrays.
Declaration-only graph lowering lives in specialized.hybrid.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from cffg._immutable import FrozenArray, ImmutableRecord
from cffg.core.hybrid import map_component
from cffg.core.ir import (
    HYBRID_APPROXIMATION,
    HybridAdapterSpec,
    HybridComponentSpec,
    HybridRegionSpec,
    HybridSeparatorSpec,
)


@dataclass(frozen=True)
class HybridSeparatorPlan(ImmutableRecord):
    """A labeled entropy owner and its explicitly identified graph copies.

    Label support remains in the matching serialized separator. Variable names
    reference Arena.edges; slot numbers retain their ordinary arena meanings.
    """

    name: str
    cardinality: int
    dim: int
    regime_variables: tuple[str, ...]
    state_variables: tuple[str, ...]


@dataclass(frozen=True)
class HybridRegionPlan(ImmutableRecord):
    """Ordered ports of one declared factor-energy/region-entropy owner.

    ``separator_indices`` indexes the enclosing plan's separators, in declared
    port order. The declared factor is its key in Arena.member_of, when it has
    parameters; this record does not copy parameter arrays or learning targets.
    """

    name: str
    separator_indices: FrozenArray


@dataclass(frozen=True)
class HybridAdapterPlan(ImmutableRecord):
    """One zero-energy identity region facing an ordinary edge copy.

    Gather/scatter are logical slots in the categorical or Gaussian arena,
    just as for existing group plans. The adapter writes ``scatter`` and reads
    the other slot; ``gather`` preserves the edge's (slot0, slot1) order.
    """

    name: str
    separator_index: int
    domain: str
    gather: tuple[int, int]
    scatter: int


@dataclass(frozen=True)
class HybridBufferRequirement(ImmutableRecord):
    """A declared future logical buffer, never an allocated or peak byte count.

    ``dtype=None`` uses the inventory's explicit floating dtype. Fixed bool or
    integer dtypes describe support/status/index requirements. Unknown layouts
    use None on the component's requirement collection, not an empty shape.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str | None = None


def _integer(value, minimum=0):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value >= minimum


def _names(values):
    return all(isinstance(v, str) and v for v in values) and len(set(values)) == len(values)


@dataclass(frozen=True)
class HybridComponentPlan(ImmutableRecord):
    """A reconstructible ownership description, with execution still deferred.

    The compiler must derive the mapping from graph incidence, resolve support
    and parameter arrays, and certify exclusive ownership before publication.
    This constructor certifies only the identity and index contract. In
    particular, it does not map arbitrary graphs through Coordinate.from_sites.

    Coordinate/scratch shapes may be declared once their layout is known.
    None means unavailable; an empty tuple explicitly declares no buffers.
    No accepted coordinate, belief, derivative capture or learned statistic is
    stored here. Publishing a plan cannot enable the ordinary D9 execution path.
    """

    declaration: HybridComponentSpec
    separators: tuple[HybridSeparatorPlan, ...]
    regions: tuple[HybridRegionPlan, ...]
    adapters: tuple[HybridAdapterPlan, ...] = ()
    coordinate_requirements: tuple[HybridBufferRequirement, ...] | None = None
    scratch_requirements: tuple[HybridBufferRequirement, ...] | None = None

    def __post_init__(self):
        super().__post_init__()
        spec = self.declaration
        prefix = f"hybrid component {getattr(spec, 'name', '<unnamed>')!r}"

        def fail(message):
            raise ValueError(f"{prefix}: {message}")

        if not isinstance(spec, HybridComponentSpec) or not _names((spec.name,)):
            fail("requires a named HybridComponentSpec declaration")
        if spec.approximation != HYBRID_APPROXIMATION:
            fail(f"unknown approximation {spec.approximation!r}")
        if spec.support_semantics != "fixed":
            fail(f"unsupported support semantics {spec.support_semantics!r}")
        for label, declared, plans, spec_type, plan_type in (
            ("separator", spec.separators, self.separators, HybridSeparatorSpec, HybridSeparatorPlan),
            ("region", spec.regions, self.regions, HybridRegionSpec, HybridRegionPlan),
            ("adapter", spec.adapters, self.adapters, HybridAdapterSpec, HybridAdapterPlan),
        ):
            if (not all(isinstance(p, spec_type) for p in declared)
                    or not all(isinstance(p, plan_type) for p in plans)
                    or not _names(tuple(p.name for p in plans))
                    or not _names(tuple(p.name for p in declared))
                    or {p.name for p in plans} != {p.name for p in declared}):
                fail(f"{label} plan identities disagree with the declaration")
        if not _names(tuple(p.name for p in (*spec.separators, *spec.regions, *spec.adapters))):
            fail("duplicate declaration names across owners")
        object.__setattr__(self, "declaration", map_component(spec))
        if not self.separators or not self.regions:
            fail("requires nonempty separator and region ownership")
        if not _names(self.members):
            fail("factor/equality ownership is unnamed or duplicated")

        separator_index = {p.name: i for i, p in enumerate(self.separators)}
        separator_specs = {s.name: s for s in spec.separators}
        for plan in self.separators:
            declared = separator_specs[plan.name]
            if not _integer(plan.cardinality, 2) or not _integer(plan.dim, 1):
                fail(f"separator {plan.name!r}: invalid cardinality or dimension")
            support = declared.active_support
            if (not support or not all(_integer(i) and i < plan.cardinality for i in support)
                    or len(set(support)) != len(support)):
                fail(f"separator {plan.name!r}: invalid fixed label support")
            if (not _names(plan.regime_variables) or not _names(plan.state_variables)
                    or declared.regime not in plan.regime_variables
                    or declared.state not in plan.state_variables
                    or set(plan.regime_variables) & set(plan.state_variables)):
                fail(f"separator {plan.name!r}: graph-copy identities disagree")

        region_specs = {r.name: r for r in spec.regions}
        for plan in self.regions:
            declared = region_specs[plan.name]
            ports = declared.ports
            if not isinstance(declared.kind, str) or not _names(ports):
                fail(f"region {plan.name!r}: invalid kind or separator ports")
            arity = {"recurrence": 2, "dynamics": 2, "unary": 1}.get(declared.kind)
            if len(ports) != arity or any(p not in separator_index for p in ports):
                fail(f"region {plan.name!r}: invalid kind or separator ports")
            indices = plan.separator_indices
            if (not isinstance(indices, FrozenArray) or indices.shape != (len(ports),)
                    or not np.issubdtype(indices.dtype, np.integer)
                    or tuple(indices) != tuple(separator_index[p] for p in ports)):
                fail(f"region {plan.name!r}: ordered separator indexes disagree")

        adapter_specs = {a.name: a for a in spec.adapters}
        for plan in self.adapters:
            declared = adapter_specs[plan.name]
            if (not isinstance(declared.separator, str) or not _integer(plan.separator_index)
                    or separator_index.get(declared.separator) != plan.separator_index):
                fail(f"adapter {plan.name!r}: separator index disagrees")
            separator = self.separators[plan.separator_index]
            variables = {"categorical": separator.regime_variables, "gaussian": separator.state_variables}
            if not isinstance(plan.domain, str) or declared.variable not in variables.get(plan.domain, ()):
                fail(f"adapter {plan.name!r}: ordinary domain/copy disagrees")
            if (len(plan.gather) != 2 or not all(_integer(i) for i in plan.gather)
                    or plan.gather[0] == plan.gather[1]
                    or not _integer(plan.scatter) or plan.scatter not in plan.gather):
                fail(f"adapter {plan.name!r}: invalid ordinary gather/scatter slots")

        for kind, requirements in (("coordinate", self.coordinate_requirements),
                                   ("scratch", self.scratch_requirements)):
            if requirements is None:
                continue
            if (not all(isinstance(r, HybridBufferRequirement) for r in requirements)
                    or not _names(tuple(r.name for r in requirements))):
                fail(f"invalid {kind} requirement identities")
            for requirement in requirements:
                if (not all(_integer(d) for d in requirement.shape)
                        or requirement.dtype not in (None, "bool", "int32", "int64")):
                    fail(f"invalid {kind} requirement {requirement.name!r} shape/dtype")

    @property
    def name(self) -> str:
        return self.declaration.name

    @property
    def approximation(self) -> str:
        return self.declaration.approximation

    @property
    def members(self) -> tuple[str, ...]:
        """Exclusive execution/factor-energy ownership, including copy wiring."""
        return tuple(r.factor for r in self.declaration.regions) + self.declaration.equalities

    @property
    def parameter_members(self) -> tuple[str, ...]:
        """Declared factor keys for existing Arena.member_of address lookup."""
        return tuple(r.factor for r in self.declaration.regions)

    @property
    def schedule_stage(self) -> str:
        return "deferred"

"""Registered structural plans, before observation rows and arrays resolve."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from cffg._immutable import FrozenArray, ImmutableRecord
from cffg.array_requirements import ArrayRequirement, SourceRequirement
from cffg.compiled import GroupPlan
from cffg.kernels import FactorKernel


@dataclass(frozen=True)
class RegistrySnapshot(ImmutableRecord):
    bindings: Mapping[str, FactorKernel]
    conditioned_out_types: frozenset[str]
    configured: Mapping[str, FactorKernel] = field(default_factory=dict)

    def for_factor(self, factor):
        return self.configured.get(factor.name, self.bindings.get(factor.type))


@dataclass(frozen=True)
class ObservationSource(ImmutableRecord):
    """A missing port retains intent until conditioning or row resolution fails."""

    member: str
    port: str | None
    variable: str
    dim: int
    family: str


@dataclass(frozen=True)
class ObservationIntent(ImmutableRecord):
    """Observed role and member order, without claiming resolved row numbers."""

    role: str
    role_index: int | None
    sources: tuple[ObservationSource, ...]


@dataclass(frozen=True)
class EqualityReference(ImmutableRecord):
    """Opaque specialized route, ordered beside registered groups."""

    key: tuple
    members: tuple[str, ...]


@dataclass(frozen=True)
class DiscreteDraft(ImmutableRecord):
    plan: GroupPlan
    arrays: ArrayRequirement
    problem: str | None = None
    path_problem: str | None = None


@dataclass(frozen=True)
class GaussianDraft(ImmutableRecord):
    name: str
    type: str
    roles: tuple[str, ...]
    dims: tuple[int, ...]
    members: tuple[str, ...]
    gather: FrozenArray
    scatter: FrozenArray
    kernel: FactorKernel
    parameter_paths: tuple[tuple[str, str], ...]
    observation: ObservationIntent | None
    arrays: ArrayRequirement
    path_problem: str | None = None


@dataclass(frozen=True)
class EfDraft(ImmutableRecord):
    name: str
    type: str
    family: str
    roles: tuple[str, ...]
    members: tuple[str, ...]
    gather: FrozenArray
    scatter: FrozenArray
    kernel: FactorKernel
    observation: ObservationIntent | None
    arrays: ArrayRequirement


@dataclass(frozen=True)
class RegisteredPlanDrafts(ImmutableRecord):
    sources: tuple[SourceRequirement, ...]
    discrete: tuple[DiscreteDraft | EqualityReference, ...]
    gaussian: tuple[GaussianDraft | EqualityReference, ...]
    ef: tuple[EfDraft | EqualityReference, ...]
    bp_members: frozenset[str]
    data_slots: Mapping[str, int]
    conditioning_problems: tuple[str, ...]
    grouping_problem: str | None

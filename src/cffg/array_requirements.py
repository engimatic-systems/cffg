"""Host-only array declarations shared by structural lowering stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

from cffg._immutable import ImmutableRecord
from cffg.kernel_bindings import ParameterSpec

if TYPE_CHECKING:
    from cffg.kernels import FactorKernel


class ArrayRecipe(str, Enum):
    """Family-owned interpretations, not executable or serialized callbacks."""

    DIRICHLET = "dirichlet"
    GFE = "gfe"
    GFE_TM = "gfe_tm"
    HYBRID_GFE = "hybrid_gfe"
    MIXTURE = "mixture"
    RECURRENCE = "recurrence"
    PRECISION = "precision"
    CONSTRAINT = "constraint"
    REGISTERED = "registered"


@dataclass(frozen=True)
class ShapeRequirement(ImmutableRecord):
    """An unevaluated trusted shape provider and its declared role sizes."""

    parameter: ParameterSpec
    role_sizes: Mapping[str, int]


@dataclass(frozen=True)
class SourceRequirement(ImmutableRecord):
    """One source role, in original traversal order, before any blob loads."""

    factor: str
    role: str
    blob: str
    shape: ShapeRequirement | None
    covariance: bool
    initializer_fallback: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ParameterRuntime(ImmutableRecord):
    """A declared packed address and its existing write/learning interpretation."""

    key: str
    transform: str = "identity"
    probability_axis: int | None = None
    permutation: tuple[int, ...] = ()
    learning_form: Literal["expected", "counts"] = "expected"
    learning_layout: Literal["as_is", "tm"] = "as_is"


@dataclass(frozen=True)
class MemberRequirement(ImmutableRecord):
    """Address intent, checked and materialized only after the member packs."""

    member: str
    index: int
    runtime: Mapping[str, ParameterRuntime]
    writable: bool = False
    statistic_source: bool = True


@dataclass(frozen=True)
class ArrayInput(ImmutableRecord):
    factor: str
    role: str
    blob: str
    initializer_shape: tuple[int, ...] | None
    shape_requirement: ShapeRequirement | None = None


@dataclass(frozen=True)
class ArrayDestination(ImmutableRecord):
    kind: Literal["parameter", "payload"]
    name: str


@dataclass(frozen=True)
class ArrayRequirement(ImmutableRecord):
    """Ordered source roles and destinations for one family packing operation."""

    owner: str
    recipe: ArrayRecipe
    inputs: tuple[ArrayInput, ...]
    destinations: tuple[ArrayDestination, ...]
    members: tuple[MemberRequirement, ...] = ()
    auxiliary_kernels: Mapping[str, FactorKernel | None] = field(default_factory=dict)

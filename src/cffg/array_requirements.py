"""Host-only array declarations shared by structural lowering stages."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from cffg._immutable import ImmutableRecord
from cffg.kernel_bindings import ParameterSpec


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

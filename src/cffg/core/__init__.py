"""JAX-free core: CFFG intermediate representation, validation, serialization.

Policy (docs/ARCHITECTURE.md): this package imports neither JAX nor Equinox.
Templates must be loadable, validatable, and editable on a machine with no
accelerator stack. JAX enters at the compiler boundary (cffg.compile).
"""

from cffg.core.ir import (
    FACTOR_TYPES,
    AgentSpec,
    BeliefBinding,
    BlobRef,
    BoundaryFactorization,
    Broadcast,
    Chain,
    ConstraintKind,
    ConstraintSpec,
    Expose,
    FactorSpec,
    FactorType,
    LearnableSpec,
    PlateSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
    register_factor_type,
)
from cffg.core.plates import expand_plates, plate_observations
from cffg.core.validate import ModelError, validate

__all__ = [
    "FACTOR_TYPES",
    "AgentSpec",
    "BeliefBinding",
    "BlobRef",
    "BoundaryFactorization",
    "Broadcast",
    "Chain",
    "ConstraintKind",
    "ConstraintSpec",
    "Expose",
    "FactorSpec",
    "FactorType",
    "LearnableSpec",
    "ModelError",
    "PlateSpec",
    "PortRole",
    "PortSpec",
    "Template",
    "VariableSpec",
    "expand_plates",
    "plate_observations",
    "register_factor_type",
    "validate",
]

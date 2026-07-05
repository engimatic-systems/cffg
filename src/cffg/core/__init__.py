"""JAX-free core: CFFG intermediate representation, validation, serialization.

Policy (docs/ARCHITECTURE.md): this package imports neither JAX nor Equinox.
Templates must be loadable, validatable, and editable on a machine with no
accelerator stack. JAX enters at the compiler boundary (cffg.compile).
"""

from cffg.core.ir import (
    FACTOR_TYPES,
    BlobRef,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    FactorType,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
    register_factor_type,
)
from cffg.core.validate import ModelError, validate

__all__ = [
    "FACTOR_TYPES",
    "BlobRef",
    "ConstraintKind",
    "ConstraintSpec",
    "FactorSpec",
    "FactorType",
    "ModelError",
    "PortRole",
    "PortSpec",
    "Template",
    "VariableSpec",
    "register_factor_type",
    "validate",
]

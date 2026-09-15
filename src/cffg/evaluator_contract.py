"""Future D12 evaluator and policy interface; no numerical implementation.

The existing policy remains in hybrid_joint_solver. Binding it to this
interface requires new implementation and qualification, with an explicit
inventory of every numerical source regardless of module naming. Metadata
here does not change the historical producer or its accepted evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cffg._immutable import ImmutableRecord
from cffg.incidence_contract import IncidenceContract, Requirement


@dataclass(frozen=True)
class Partition(ImmutableRecord):
    """A contiguous block in the unbound internal coordinate or residual."""

    name: str
    offset: int
    size: int
    constraint: str = "unrestricted"


@dataclass(frozen=True)
class ValueRequirement(ImmutableRecord):
    """A future input/output, not a stored State or temporary allocation.

    None shape means the producer must still specify its payload. The logical
    size excludes containers, compiled memory and derivative captures. A
    floating leaf dtype is explicit and never inferred from an absent array.
    An unbound structured payload has dtype None because it may mix numeric,
    support and status leaves; its floating leaves use the enclosing dtype.
    """

    name: str
    owners: tuple[str, ...]
    shape: tuple[int, ...] | None
    dtype: str | None
    meaning: str

    @property
    def logical_bytes(self) -> int | None:
        if self.shape is None or self.dtype is None:
            return None
        count = 1
        for dimension in self.shape:
            count *= dimension
        return count * {"float32": 4, "float64": 8, "int32": 4, "bool": 1}[self.dtype]


@dataclass(frozen=True)
class EvaluatorContract(ImmutableRecord):
    incidence: IncidenceContract
    dtype: str
    coordinate_partitions: tuple[Partition, ...]
    internal_residual_partitions: tuple[Partition, ...]
    inputs: tuple[ValueRequirement, ...]
    outputs: tuple[ValueRequirement, ...]
    certificates: tuple[str, ...]
    obligations: tuple[Requirement, ...]

    @property
    def internal_residual_size(self) -> int:
        return sum(p.size for p in self.internal_residual_partitions)

    @property
    def coupled_coordinate_size(self) -> int | None:
        return None if self.incidence.adapters else self.incidence.coordinate_size

    @property
    def coupled_residual_size(self) -> int | None:
        return None if self.incidence.adapters else self.internal_residual_size

    @property
    def accepted_state_requirements(self) -> None:
        """Runtime retention/lifecycle has not been specified or allocated."""
        return None

    @property
    def scratch_requirements(self) -> None:
        """Includes unknown evaluation payloads and derivative captures."""
        return None


def describe_evaluator(incidence: IncidenceContract, *, dtype: str) -> EvaluatorContract:
    """Inventory the interface without evaluating parameters, sites or shapes.

    Internal residual order follows the existing certificate families, using
    every directed hybrid incidence. It is rectangular: all active label mass
    residuals remain, even though one message constant is gauge. Ordinary
    adapter/exterior residuals must be bound separately before claiming a
    complete coupled solve. Empty auxiliary blocks are valid for unary graphs.
    """
    if dtype not in ("float32", "float64"):
        raise ValueError(f"hybrid component {incidence.name!r}: evaluator requires explicit float32 or float64")
    n = incidence.coordinate_size
    aux = n - incidence.message_size
    coordinates = (Partition("messages", 0, incidence.message_size),
                   Partition("squared_auxiliaries", incidence.message_size, aux, "nonnegative"))
    sizes = (
        ("natural", incidence.message_size),
        ("label_mass", sum(len(p.coefficients.labels) for p in incidence.ports)),
        ("conditional_mean", sum(p.coefficients.linear_count for p in incidence.ports)),
        ("conditional_covariance", sum(p.coefficients.precision_count for p in incidence.ports)),
        ("auxiliary", aux),
    )
    residuals, offset = [], 0
    for name, size in sizes:
        residuals.append(Partition(name, offset, size))
        offset += size
    inputs = [ValueRequirement("coordinate", (incidence.name,), (n,), dtype,
                               "One internal vector: directed messages followed by squared auxiliaries; no accepted State is created.")]
    for separator in incidence.separators:
        inputs.append(ValueRequirement("fixed_origins", (separator.name,), separator.origin_shape, dtype,
                                       "Frozen per-label origins; rebuilding is required if origins or declared support change."))
    for region in incidence.regions:
        for source in region.parameters.inputs:
            inputs.append(ValueRequirement(f"parameter:{source.role}", (region.name, region.factor), source.initializer_shape,
                                           dtype, "Original fixed factor input; retain source/role/normalizer provenance and unresolved shape provider."))
    for adapter in incidence.adapters:
        # Slot intentions and domain-specific layouts live on the adapter.
        for name, meaning in (
            ("ordinary_incoming_cavity", "Read incoming_slot; declare held or jointly updated exterior policy before binding."),
            ("ordinary_port_belief", "Ordinary equality-port moments/support from the exterior binding, distinct from its incoming cavity."),
            ("ordinary_outgoing_message", "Current ordinary quotient at outgoing_slot for complete adapter stationarity, distinct from hybrid unknowns."),
        ):
            inputs.append(ValueRequirement(name, (adapter.name,), None, None, meaning))
    outputs = [
        ValueRequirement("target", (incidence.name,), (n,), dtype,
                         "Complete simultaneous internal quotient/auxiliary target in coordinate order, independent of the rectangular residual."),
        ValueRequirement("residual", (incidence.name,), (offset,), dtype, "Unscaled internal residual with named partitions."),
        ValueRequirement("scales", (incidence.name,), (offset,), dtype,
                         "Current positive finite residual scales; freeze accepted scales across one linear solve and its trials."),
        ValueRequirement("free_energy", (incidence.name,), (), dtype,
                         "D12 primal objective; regional/separator/adapter terms retain ownership. Held exterior and coupled objectives stay distinct."),
        ValueRequirement("dual_energy", (incidence.name,), (), dtype, "Separate dual diagnostic; never a replacement for the primal objective."),
        ValueRequirement("status", (incidence.name,), (), "int32", "Array-valued properness, finite-value, support and context failure flags."),
    ]
    for name, owners, meaning in (
        ("regional_beliefs", tuple(r.name for r in incidence.regions), "Proper local beliefs in declared region scopes, including unary regions."),
        ("separator_beliefs", tuple(s.name for s in incidence.separators), "Labeled separator moments with original active support."),
        ("adapter_beliefs", tuple(a.name for a in incidence.adapters), "Reduced equality-support beliefs, never entropy of a delta distribution."),
        ("learning_statistics", tuple(r.factor for r in incidence.regions), "Factor-owned D12 expected sufficient statistics; no published learning address or write."),
        ("objective_terms", tuple(o.owner for o in incidence.objectives), "Every original region energy/entropy and degree-based correction, including zero-energy owners."),
        ("auxiliary_statistics", tuple(a.owner for a in incidence.auxiliaries), "At-risk moments, squared-auxiliary targets and stationarity diagnostics."),
        ("ordinary_outgoing_quotients", tuple(a.name for a in incidence.adapters), "One domain-specific projection-before-division quotient per ordinary outgoing slot."),
        ("ordinary_port_residuals", tuple(a.name for a in incidence.adapters), "Both adapter port equations plus applicable exterior stationarity; full coupled layout remains unresolved."),
    ):
        outputs.append(ValueRequirement(name, owners, None if owners else (0,), None if owners else dtype, meaning))
    certificates = ("natural", "label_mass", "conditional_mean", "conditional_covariance", "auxiliary", "primal_dual")
    if incidence.adapters:
        certificates += ("ordinary_adapter_natural", "ordinary_adapter_moments", "exterior_stationarity")
    outputs.append(ValueRequirement("certificates", (incidence.name,), (len(certificates),), dtype,
                                    "Dynamic maxima over all applicable families; absent families have empty domains, never hide missing exterior evidence."))
    obligations = (
        Requirement("source_inventory", (incidence.name,),
                    "Every future numerical evaluator, initializer, derivative or policy adapter needs an explicit complete source inventory and new qualification."),
        Requirement("policy_binding", (incidence.name,),
                    "Reuse hybrid_joint_solver policy through an independently reviewed binding. Adapt complete target, FE, certificates.maximum(), "
                    "status replacement and explicit auxiliary partition; do not append exterior unknowns to a presumed nonnegative tail."),
        Requirement("fixed_runtime_structure", (incidence.name,),
                    "Accepted, trial and returned evaluations need identical pytree/leaf shapes and dtypes; retain array-valued failure status replacement."),
        Requirement("initialization", (incidence.name,),
                    "Specify feasible generated-start and explicit-coordinate starts separately. The current generated start assumes adjacent transitions. "
                    "Valid empty blocks differ from invalid empty support; traced zero-size paths still require target/derivative outputs and complete certificates."),
        Requirement("derivative_actions", (incidence.name,),
                    "At a proper accepted point, forward maps scaled N-coordinate steps to M residuals with frozen accepted scales; "
                    "transpose is the adjoint of that same action/capture. No dense Jacobian or derivative through rejected trials."),
        Requirement("properness_and_convergence", (incidence.name,),
                    "Signed/vacuous cavities are allowed; proper tilted/accepted beliefs, exact fixed support, finite FE and residuals, complete "
                    "dynamic certificates and FE stability are required. Preserve converged/exhausted/diverged outcomes and rejection reasons; "
                    "zero motion or empty coordinates alone do not certify convergence."),
        Requirement("storage", (incidence.name,),
                    "Interface values are not an allocation inventory. Account accepted/trial evaluations, target/scales, evaluator temporaries and "
                    "derivative captures separately from the existing three N plus one M CGLS vectors. Logical, compiled and process peaks are distinct and unknown here."),
    )
    return EvaluatorContract(incidence, dtype, coordinates, tuple(residuals), tuple(inputs), tuple(outputs), certificates, obligations)


class CertificateValues[Array](Protocol):
    """All named certificate families from EvaluatorContract, at one point."""

    def maximum(self) -> Array: ...


class EvaluationValues[Array, Payload](Protocol):
    """Future fixed-structure outputs; no instance is constructed here.

    Payload holds the named belief/statistic/objective output inventory. A
    binding must supply every certificate applicable to its exterior policy.
    """

    target: Array
    residual: Array
    scales: Array
    free_energy: Array
    dual_energy: Array
    certificates: CertificateValues[Array]
    status: Array
    payload: Payload

    def with_status(self, status: Array) -> EvaluationValues[Array, Payload]: ...


class DerivativeActions[Array](Protocol):
    """A single accepted capture with frozen residual/coordinate scales."""

    residual: Array

    def forward(self, step: Array) -> Array: ...

    def transpose(self, cotangent: Array) -> Array: ...


class Evaluator[Context, Array, Payload, StartOptions](Protocol):
    """Required future methods, not a wrapper around the qualified chain.

    Context binds fixed parameters, origins/support and the explicitly chosen
    exterior policy. An explicit start goes directly to evaluate; initialize
    owns a separately qualified generated feasible start. All arrays conform
    to the bound coordinate/residual partitions, including empty blocks.
    """

    def evaluate(self, context: Context, coordinate: Array) -> EvaluationValues[Array, Payload]: ...

    def initialize(self, context: Context, options: StartOptions) -> tuple[Array, EvaluationValues[Array, Payload]]: ...

    def linearize(self, context: Context, coordinate: Array, residual_scales: Array,
                  coordinate_scales: Array) -> DerivativeActions[Array]: ...

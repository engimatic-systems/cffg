"""Host-side declarations for single-arena executable factors.

Role sizes come from the template, in the binding's role order. Parameter
shapes describe serialized blobs, not the possibly transformed runtime pack.
This module deliberately does not import JAX.
Construction snapshots declaration mappings and sequences. Executable callbacks
remain trusted identity-bearing objects; their closure state is not frozen.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from cffg._immutable import ImmutableRecord


@dataclass(frozen=True)
class ParameterSpec(ImmutableRecord):
    """Shape inference and optional direct replacement of a serialized blob.

    ``shape`` receives a mapping from role names to cardinalities/dimensions.
    ``key`` names the packed row when direct replacement is supported. Leave it
    unset for joint/nonlinear packs; those are reconstructed by compilation.
    ``permutation`` applies after the numerical transform, excluding the member
    axis. ``group_key`` preserves an existing bare group-name storage address.
    """

    shape: Callable
    key: str | None = None
    transform: str = "identity"
    probability_axis: int | None = None
    permutation: tuple[int, ...] = ()
    group_key: bool = False


@dataclass(frozen=True)
class EFBinding(ImmutableRecord):
    """Scalar natural-parameter messages; legacy callback signatures preserved."""

    family: str
    message_roles: tuple[str, ...]
    edge_families: Mapping[str, str]
    message: Callable
    energy: Callable
    param_keys: tuple[str, ...] = ()
    pack_params: Callable | None = None
    observed_out: str | None = None
    validate_observation: Callable | None = None
    constraint_modes: frozenset[str] = frozenset()
    readout: Callable | None = None
    parameters: Mapping[str, ParameterSpec] = field(default_factory=dict)
    arena: str = field(default="ef", init=False)


@dataclass(frozen=True)
class DiscreteMode(ImmutableRecord):
    """Callbacks for one explicitly selected inference mode.

    BP input is opposite-side cavities [M,R,Klocal]. Its message receives
    ``(input, params, cards, Klocal, reduction)``. VMP input is a tuple of
    ``(q, log_q)`` pairs, one [M,card] pair per role, normalized from both
    edge messages. Its message receives ``(input, params, cards, Klocal)``.
    Energy and optional readout receive ``(input, params, cards)`` in either
    mode; energies return [M], readouts a mapping of names to [M,...] arrays.
    """

    message: Callable
    energy: Callable
    readout: Callable | None = None


@dataclass(frozen=True)
class DiscreteBinding(ImmutableRecord):
    """Finite discrete factors using local padded log-message blocks.

    ``potential(params, cards)`` returns [M,*cards] log potentials in role
    order. It owns support, joint BP beliefs and assignment scoring; the
    runtime never assumes a particular packed parameter key. Exact zeros use
    the arena's NEG sentinel (literal -inf is also accepted as input support).
    Discrete DATA ports retain structural edge clamping, not node absorption.
    """

    roles: tuple[str, ...]
    parameters: Mapping[str, ParameterSpec]
    param_keys: tuple[str, ...]
    pack_params: Callable | None
    potential: Callable
    bp: DiscreteMode
    vmp: DiscreteMode | None = None
    semirings: frozenset[str] = frozenset({"sum", "max"})
    arena: str = field(default="discrete", init=False)

    @property
    def message_roles(self) -> tuple[str, ...]:
        return self.roles

    @property
    def constraint_modes(self) -> frozenset[str]:
        return frozenset({"mean_field"}) if self.vmp is not None else frozenset()


@dataclass(frozen=True)
class GaussianConditioning(ImmutableRecord):
    """One legal DATA role and its host validator ``(value, dimension)``.

    ``plan_type`` is reserved for built-in compatibility labels. External
    labels are derived from the factor type and observed role. Neither
    changes the serialized factor type or implementation identity.
    """

    role: str
    validate: Callable
    plan_type: str | None = None


@dataclass(frozen=True)
class GaussianBinding(ImmutableRecord):
    """Information-form callbacks retaining every structural role.

    Callbacks receive ``(eta, precision, params, dims, observed_role, y, mask)``.
    Incoming blocks are [M,R,Dlocal] and [M,R,Dlocal,Dlocal]; dimensions and
    the optional observed-role index are static. Observation arrays are
    [M,d_observed] and [M], or None for unobserved groups. Messages return
    one exact-sized ``(eta [M,d], precision [M,d,d])`` pair per role,
    including the cavity predictive on a conditioned role. Energy returns
    [M]. Optional readout/statistics callbacks return named [M,...] arrays.

    ``statistics`` feeds gaussian_node_stats (conjugate sufficient statistics);
    ``readout`` feeds kernel_readouts (extension-specific summaries). They
    may use host NumPy; message and energy callbacks must be JAX trace-safe.
    A missing observation means marginalization of its role, not necessarily
    deletion of the remaining factor. The callback owns that mathematical
    distinction and the node's normalization constant.
    """

    roles: tuple[str, ...]
    parameters: Mapping[str, ParameterSpec]
    param_keys: tuple[str, ...]
    pack_params: Callable | None
    message: Callable
    energy: Callable
    conditioning: GaussianConditioning | None = None
    statistics: Callable | None = None
    readout: Callable | None = None
    arena: str = field(default="gaussian", init=False)
    constraint_modes: frozenset[str] = field(default=frozenset(), init=False)

    @property
    def message_roles(self) -> tuple[str, ...]:
        return self.roles

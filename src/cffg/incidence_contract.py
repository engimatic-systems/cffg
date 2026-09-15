"""Unbound D12 incidence metadata derived from reviewed structural drafts.

This is not a numerical producer or an executable component plan. In
particular, neither a directed-path mapping nor a coefficient count qualifies
an evaluator. Array resolution, fixed origins and runtime storage are later
contracts. D12.4/D12.7 in DERIVATIONS own the counting and gauge conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cffg._immutable import ImmutableRecord
from cffg.array_requirements import ArrayRequirement
from cffg.compiled_hybrid import HybridComponentPlan
from cffg.specialized.contracts import HybridComponentDraft


@dataclass(frozen=True)
class Coefficients(ImmutableRecord):
    """One outgoing message, ordered constants, linear, upper precision.

    Labels keep their original integer identities. Only the first active
    constant is fixed to zero. A Gaussian ordinary message has no label axis
    and its sole constant is gauge. Inactive hybrid constants reconstruct as
    -inf; other inactive coefficients reconstruct as zero. No correction word
    or fixed origin is an unknown. Precision order is row-major upper triangle.
    """

    domain: Literal["hybrid", "categorical", "gaussian"]
    labels: tuple[int, ...]
    dim: int

    @property
    def gauge_label(self) -> int | None:
        return self.labels[0] if self.labels else None

    @property
    def constant_count(self) -> int:
        return max(len(self.labels) - 1, 0)

    @property
    def linear_count(self) -> int:
        return (len(self.labels) if self.domain == "hybrid" else 1) * self.dim

    @property
    def precision_count(self) -> int:
        return (len(self.labels) if self.domain == "hybrid" else 1) * self.dim * (self.dim + 1) // 2

    @property
    def size(self) -> int:
        return self.constant_count + self.linear_count + self.precision_count


@dataclass(frozen=True)
class Port(ImmutableRecord):
    """A directed region/adapter -> hybrid separator incidence.

    Cavities are future evaluator inputs derived from the other incidences,
    not a second bank of independent separator -> region messages.
    """

    owner: str
    position: int
    separator: str
    coefficients: Coefficients
    offset: int


@dataclass(frozen=True)
class Separator(ImmutableRecord):
    name: str
    cardinality: int
    dim: int
    support: tuple[int, ...]
    regime_variables: tuple[str, ...]
    state_variables: tuple[str, ...]
    ports: tuple[int, ...]

    @property
    def origin_shape(self) -> tuple[int, int]:
        """Fixed floating input, including original inactive label positions."""
        return self.cardinality, self.dim


@dataclass(frozen=True)
class Region(ImmutableRecord):
    name: str
    kind: str
    factor: str
    ports: tuple[int, ...]
    parameters: ArrayRequirement


@dataclass(frozen=True)
class Auxiliary(ImmutableRecord):
    """Squared JJ coordinates in source-label, at-risk-stick order.

    A stick remains at risk when any active destination label is at or beyond
    it. Thus sparse labels do not renumber the stick-breaking model. This is
    the support inventory used by Coordinate.from_sites, without allocating
    its dense [K,K-1] mask. Values must later be finite and nonnegative.
    """

    owner: str
    source_labels: tuple[int, ...]
    sticks: tuple[int, ...]
    full_shape: tuple[int, int]
    offset: int

    @property
    def size(self) -> int:
        return len(self.source_labels) * len(self.sticks)


@dataclass(frozen=True)
class Adapter(ImmutableRecord):
    """An identity region with hybrid port 0 and ordinary port 1.

    Its incoming cavity and outgoing quotient are ordinary slot intentions,
    not held unary evidence or additional accepted component State. Both port
    moment equations and the reduced identity entropy remain explicit. The
    +1 ordinary entropy contribution is only this adapter's degree increment;
    the full ordinary degree/correction belongs to exterior assembly.
    """

    name: str
    hybrid_port: int
    variable: str
    domain: str
    gather: tuple[int, int]
    incoming_slot: int
    outgoing_slot: int
    outgoing_coefficients: Coefficients
    ordinary_entropy_increment: int = 1
    ordinary_entropy_coefficient: int | None = None


@dataclass(frozen=True)
class Objective(ImmutableRecord):
    """One original owner in the alternating-region D12 objective.

    Entropy coefficients multiply H, not -H. A separator uses its full
    region-plus-adapter degree minus one, before any unary cancellation.
    Equalities are copy wiring with zero energy, not extra entropy regions.
    """

    owner: str
    kind: Literal["region", "separator", "adapter", "equality"]
    factor: str | None
    energy: Literal["negative_log_factor_expectation", "zero", "none"]
    entropy_coefficient: int
    port_count: int


@dataclass(frozen=True)
class IncidenceContract(ImmutableRecord):
    plan: HybridComponentPlan
    separators: tuple[Separator, ...]
    regions: tuple[Region, ...]
    ports: tuple[Port, ...]
    auxiliaries: tuple[Auxiliary, ...]
    adapters: tuple[Adapter, ...]
    objectives: tuple[Objective, ...]

    @property
    def name(self) -> str:
        return self.plan.name

    @property
    def message_size(self) -> int:
        return sum(p.coefficients.size for p in self.ports)

    @property
    def coordinate_size(self) -> int:
        return self.message_size + sum(a.size for a in self.auxiliaries)


def from_draft(draft: HybridComponentDraft) -> IncidenceContract:
    """Describe a reviewed draft without resolving arrays or invoking callbacks.

    Graph admission remains core.hybrid's responsibility. Here named errors
    detect malformed ownership, parameter and local scope contracts. A valid
    unary-only, branching or disconnected graph is not a chain-mapping error.
    The result is separate from the draft and does not publish an Arena plan.
    """
    if not isinstance(draft, HybridComponentDraft):
        raise TypeError("D12 incidence requires a reviewed HybridComponentDraft")
    plan = draft.plan
    if not isinstance(plan, HybridComponentPlan):
        raise ValueError("hybrid component '<unnamed>': missing compiled ownership contract")

    def fail(message):
        raise ValueError(f"hybrid component {plan.name!r}: {message}")

    if (len(draft.ownership.executed) != len(plan.members)
            or set(draft.ownership.executed) != set(plan.members)):
        fail("factor/equality execution ownership disagrees with declaration")
    expected = {(r.name, "region", r.factor) for r in plan.declaration.regions}
    expected |= {(s.name, "separator", None) for s in plan.separators}
    expected |= {(a.name, "adapter", None) for a in plan.adapters}
    if (len(draft.objectives) != len(expected)
            or {(o.name, o.kind, o.factor) for o in draft.objectives} != expected):
        fail("objective ownership disagrees with declaration")
    parameters, parameter_owners, destinations = {}, set(), set()
    for requirement in draft.arrays:
        if not isinstance(requirement, ArrayRequirement) or len(requirement.members) != 1:
            fail("each factor requires one unresolved parameter member")
        member = requirement.members[0]
        if member.member in parameters or member.member not in plan.parameter_members:
            fail(f"parameter member {member.member!r} is duplicated or unowned")
        if member.writable or member.statistic_source:
            fail(f"parameter member {member.member!r} is not an unresolved fixed input")
        if any(source.factor != member.member for source in requirement.inputs):
            fail(f"parameter member {member.member!r} has another factor's source")
        roles = tuple(source.role for source in requirement.inputs)
        keys = tuple(destination.name for destination in requirement.destinations)
        if (requirement.owner in parameter_owners or len(set(roles)) != len(roles)
                or len(set(keys)) != len(keys) or destinations.intersection(keys)):
            fail(f"parameter member {member.member!r} has duplicate owner/source/destination intentions")
        parameter_owners.add(requirement.owner)
        destinations.update(keys)
        parameters[member.member] = requirement
    if set(parameters) != set(plan.parameter_members):
        fail("parameter inputs do not cover every declared factor")

    specs = {s.name: s for s in plan.declaration.separators}
    separators = {s.name: s for s in plan.separators}
    regime_copies, state_copies = {}, {}
    for separator in plan.separators:
        for domain, variables, registry, identity in (
            ("regime", separator.regime_variables, regime_copies,
             (frozenset(separator.regime_variables), separator.cardinality, frozenset(specs[separator.name].active_support))),
            ("state", separator.state_variables, state_copies,
             (frozenset(separator.state_variables), separator.dim)),
        ):
            for variable in variables:
                previous, owner = registry.setdefault(variable, (identity, separator.name))
                if previous != identity:
                    fail(f"separators {owner!r}/{separator.name!r}: shared {domain} copies/support/dimensions disagree")
    uses = {s.name: [] for s in plan.separators}
    ports, regions, adapters = [], [], []
    offset = 0

    def add_port(owner, position, name):
        nonlocal offset
        separator = separators[name]
        coefficients = Coefficients("hybrid", tuple(sorted(specs[name].active_support)), separator.dim)
        index = len(ports)
        ports.append(Port(owner, position, name, coefficients, offset))
        uses[name].append(index)
        offset += coefficients.size
        return index

    declared_regions = sorted(plan.declaration.regions, key=lambda r: r.name)
    pairs = {}
    for region in declared_regions:
        if region.kind in ("recurrence", "dynamics"):
            middle = region.ports[1 if region.kind == "recurrence" else 0]
            pairs.setdefault(middle, []).append(region.kind)
    for middle, kinds in pairs.items():
        if sorted(kinds) != ["dynamics", "recurrence"]:
            fail(f"separator {middle!r}: malformed R/H/D pairing")
    for region in declared_regions:
        if len({(separators[p].cardinality, separators[p].dim) for p in region.ports}) != 1:
            fail(f"region {region.name!r}: port dimensions disagree")
        if region.kind != "unary":
            left, right = (separators[p] for p in region.ports)
            same_regime = set(left.regime_variables) == set(right.regime_variables)
            same_state = set(left.state_variables) == set(right.state_variables)
            if (region.kind == "recurrence" and (same_regime or not same_state)
                    or region.kind == "dynamics" and (not same_regime or same_state)):
                fail(f"region {region.name!r}: ordered R/H/D copy scopes disagree")
        indices = tuple(add_port(region.name, i, p) for i, p in enumerate(region.ports))
        regions.append(Region(region.name, region.kind, region.factor, indices, parameters[region.factor]))
    adapter_specs = {a.name: a for a in plan.declaration.adapters}
    for adapter in sorted(plan.adapters, key=lambda a: a.name):
        spec = adapter_specs[adapter.name]
        index = add_port(adapter.name, 0, spec.separator)
        separator = separators[spec.separator]
        labels = tuple(sorted(specs[spec.separator].active_support)) if adapter.domain == "categorical" else ()
        coefficients = Coefficients(adapter.domain, labels, separator.dim if adapter.domain == "gaussian" else 0)
        incoming, = (slot for slot in adapter.gather if slot != adapter.scatter)
        adapters.append(Adapter(adapter.name, index, spec.variable, adapter.domain,
                                adapter.gather, incoming, adapter.scatter, coefficients))
    auxiliaries = []
    for region in declared_regions:
        if region.kind == "recurrence":
            left, right = (specs[p] for p in region.ports)
            k = separators[left.name].cardinality
            sticks = tuple(range(min(max(right.active_support) + 1, k - 1)))
            auxiliary = Auxiliary(region.name, tuple(sorted(left.active_support)), sticks, (k, k - 1), offset)
            auxiliaries.append(auxiliary)
            offset += auxiliary.size
    boundary = tuple(Separator(s.name, s.cardinality, s.dim, tuple(sorted(specs[s.name].active_support)),
                               s.regime_variables, s.state_variables, tuple(uses[s.name]))
                     for s in sorted(plan.separators, key=lambda s: s.name))
    objectives = (
        tuple(Objective(r.name, "region", r.factor, "negative_log_factor_expectation", -1, len(r.ports)) for r in regions)
        + tuple(Objective(s.name, "separator", None, "none", len(s.ports) - 1, len(s.ports)) for s in boundary)
        + tuple(Objective(a.name, "adapter", None, "zero", -1, 2) for a in adapters)
        + tuple(Objective(name, "equality", name, "zero", 0, 0) for name in sorted(plan.declaration.equalities))
    )
    return IncidenceContract(plan, boundary, tuple(regions), tuple(ports), tuple(auxiliaries), tuple(adapters), objectives)


@dataclass(frozen=True)
class Requirement(ImmutableRecord):
    code: str
    owners: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class ChainTransition(ImmutableRecord):
    recurrence: str
    dynamics: str
    left: str
    middle: str
    right: str


@dataclass(frozen=True)
class ChainSite(ImmutableRecord):
    separator: str
    unary_owners: tuple[str, ...]
    held_adapter_inputs: tuple[str, ...]


@dataclass(frozen=True)
class ChainMapping(ImmutableRecord):
    """Ordered internal mapping, conditional on held adapter cavities if any.

    Transition t maps to bank positions (left,t) and (right,t); its recurrence
    owns xi[t]. Unary sites retain each original energy and normalizer owner.
    D12.3 lifts the eliminated H beliefs; this is not a permutation of the
    general coordinate. No arrays or original objective terms are discarded.
    """

    transitions: tuple[ChainTransition, ...]
    sites: tuple[ChainSite, ...]
    shape: tuple[int, int, int]


@dataclass(frozen=True)
class ChainApplicability(ImmutableRecord):
    """Structural applicability only, never complete numerical qualification."""

    internal_mapping: ChainMapping | None
    general_requirements: tuple[Requirement, ...]
    unresolved_inputs: tuple[Requirement, ...]
    exterior_obligations: tuple[Requirement, ...]


def chain_applicability(contract: IncidenceContract) -> ChainApplicability:
    """Find a directed path for the unchanged TransitionParameters/sites API.

    An H unary need not destroy D12.3's conditional identity, but that API has
    no H-site input. Even separable H sites require a separately implemented
    and qualified transport. Held ordinary inputs only define a conditional
    internal objective; adapter/ordinary stationarity is not chain-certified.
    """
    regions = {r.name: r for r in contract.regions}
    separators = {s.name: s for s in contract.separators}
    scopes = {r.name: tuple(contract.ports[p].separator for p in r.ports) for r in contract.regions}
    requirements = []

    def need(code, owners, detail):
        requirements.append(Requirement(code, tuple(owners), detail))

    transitions = []
    dynamics = {}
    for region in contract.regions:
        if region.kind == "dynamics":
            dynamics.setdefault(scopes[region.name][0], []).append(region.name)
    recurrences = {}
    for region in contract.regions:
        if region.kind == "recurrence":
            left, middle = scopes[region.name]
            recurrences.setdefault(middle, []).append(region.name)
            paired = dynamics.get(middle, [])
            if len(paired) == 1:
                dynamics_name, = paired
                transitions.append(ChainTransition(region.name, dynamics_name, left, middle, scopes[dynamics_name][1]))
    for middle in sorted(recurrences.keys() | dynamics.keys()):
        if len(recurrences.get(middle, ())) != 1 or len(dynamics.get(middle, ())) != 1:
            raise ValueError(f"hybrid component {contract.name!r}: separator {middle!r} has malformed R/H/D pairing")
    for transition in transitions:
        extra = tuple(contract.ports[p].owner for p in separators[transition.middle].ports
                      if contract.ports[p].owner not in (transition.recurrence, transition.dynamics))
        if extra:
            need("intermediate_inputs", (transition.middle, *extra),
                 "The unchanged grouped API has no H unary/adapter input; preserve its scope and objective.")
    domains = {(s.cardinality, s.dim) for s in contract.separators}
    if len(domains) != 1:
        need("heterogeneous_shapes", separators, "The chain API requires homogeneous total K and d.")
    if any(s.cardinality < 2 for s in contract.separators):
        need("total_cardinality", separators, "Coordinate.from_sites requires total K>=2; singleton active support is distinct.")
    if not transitions:
        need("no_transitions", regions, "Unary-only incidence needs its own evaluator; no fictitious transition may be added.")
    incoming, outgoing = {}, {}
    for transition in transitions:
        incoming.setdefault(transition.right, []).append(transition)
        outgoing.setdefault(transition.left, []).append(transition)
    endpoints = set(incoming) | set(outgoing)
    branching = sorted(s for s in endpoints if len(incoming.get(s, ())) > 1 or len(outgoing.get(s, ())) > 1)
    if branching:
        need("directed_branching", branching, "A path needs at most one incoming and one outgoing transition; reversal is not a mapping.")
    starts = sorted(endpoints - incoming.keys())
    ordered, site_names = [], []
    if not branching and len(starts) == 1:
        current = starts[0]
        site_names.append(current)
        visited = set()
        while current in outgoing and current not in visited:
            visited.add(current)
            transition, = outgoing[current]
            ordered.append(transition)
            current = transition.right
            site_names.append(current)
    if transitions and (len(ordered) != len(transitions) or len(set(site_names)) != len(site_names)):
        need("not_one_directed_path", sorted(endpoints), "Cycles and independent paths require separate evaluation; components cannot be concatenated.")
    covered = endpoints | {t.middle for t in transitions}
    if set(separators) - covered:
        need("separate_separators", sorted(set(separators) - covered), "Keep disconnected unary/separator ownership in the general evaluator.")
    exterior = tuple(Requirement(
        "ordinary_adapter_stationarity", (a.name, a.variable),
        "Held incoming cavities bind only an internal conditional solve. Preserve both port moments, outgoing quotient, "
        "reduced entropy, ordinary correction and exterior factor objective; coupled ordinary coordinates/certificates are unresolved.")
        for a in contract.adapters)
    unresolved = (
        Requirement("resolved_parameters", tuple(r.factor for r in contract.regions),
                    "Resolve original factor arrays and each unary normalizer without changing ownership."),
        Requirement("fixed_support_origins_dtype", tuple(separators),
                    "Verify exact declared support and original label positions; choose fixed origins and one float32/float64 dtype. "
                    "Support/origin changes require specialization; shapes alone do not certify numerical validity."),
        Requirement("numerical_qualification", (contract.name,),
                    "Proper local beliefs, finite primal/dual energy and complete returned-point certificates need numerical evidence."),
    )
    mapping = None
    if not requirements:
        sites = tuple(ChainSite(name,
                               tuple(r.name for r in contract.regions if r.kind == "unary" and scopes[r.name] == (name,)),
                               tuple(a.name for a in contract.adapters if contract.ports[a.hybrid_port].separator == name))
                      for name in site_names)
        k, d = next(iter(domains))
        mapping = ChainMapping(tuple(ordered), sites, (len(sites), k, d))
    return ChainApplicability(mapping, tuple(requirements), unresolved, exterior)

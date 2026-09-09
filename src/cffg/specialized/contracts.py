"""Immutable inputs and outputs of specialized structural lowering."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from cffg._immutable import FrozenArray, ImmutableRecord
from cffg.array_requirements import ArrayRequirement
from cffg.compiled import (
    DirichletGroupPlan,
    EfGroupPlan,
    GaussGroupPlan,
    GfePlan,
    GfeTmPlan,
    GoalParameterPlan,
    GroupPlan,
    HybridGfePlan,
    MixGroupPlan,
    Mu3Plan,
    PrecGroupPlan,
    RecurrenceComponentPlan,
    RecurrencePlan,
)
from cffg.topology import TopologyLayout


@dataclass(frozen=True)
class PlanDraft[Plan](ImmutableRecord):
    """A structural plan and the still-unresolved arrays it requires."""

    plan: Plan
    arrays: ArrayRequirement


@dataclass(frozen=True)
class ParameterLayout(ImmutableRecord):
    shape: tuple[int, ...]
    axis: int
    likelihood: str


@dataclass(frozen=True)
class ParameterClaim(ImmutableRecord):
    variable: str
    layout: ParameterLayout
    prior: str | None
    is_goal: bool
    problem: str | None = None


@dataclass(frozen=True)
class GfeClaim(ImmutableRecord):
    cards: tuple[int, int]
    learned: bool
    mu3: bool
    goal_learned: bool
    pairs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class TransitionGfeClaim(ImmutableRecord):
    cards: tuple[int, int, int]
    learned: bool
    members: tuple[str, ...]


@dataclass(frozen=True)
class MixtureClaim(ImmutableRecord):
    type: str
    dims: tuple[int, ...]
    selector_cardinality: int
    members: tuple[str, ...]
    goals: tuple[str, ...] = ()


@dataclass(frozen=True)
class EqualityRoute(ImmutableRecord):
    """Declared-domain routing; ordinary compatibility grouping uses this key."""

    arena: Literal["discrete", "gaussian", "ef"]
    key: tuple


@dataclass(frozen=True)
class SpecializedClaims(ImmutableRecord):
    topology: TopologyLayout
    semiring: str
    mean_field: frozenset[str]
    parameter_layouts: Mapping[str, ParameterLayout]
    parameters: tuple[ParameterClaim, ...]
    gfe: tuple[GfeClaim, ...]
    transition_gfe: tuple[TransitionGfeClaim, ...]
    hybrid_gfe: tuple[MixtureClaim, ...]
    mixtures: tuple[MixtureClaim, ...]
    recurrence: Mapping[tuple[int, int], tuple[str, ...]]
    precision: tuple[str, ...]
    constraints: tuple[str, ...]
    equalities: Mapping[str, EqualityRoute]
    consumed: frozenset[str]
    excluded_edges: frozenset[str]
    admission_problems: tuple[str, ...]
    layout_problems: tuple[str, ...]
    composite_problems: tuple[str, ...]
    grouping_problems: Mapping[str, str]
    conditioning_problems: Mapping[str, str]
    initializer_shapes: Mapping[tuple[str, str], tuple[int, ...] | None]


@dataclass(frozen=True)
class OrdinaryMembership(ImmutableRecord):
    """Categorical BP members, including routed structural equalities."""

    bp_members: frozenset[str]


@dataclass(frozen=True)
class ConstraintDraft(ImmutableRecord):
    """No numeric k/table is published until its family validator resolves it."""

    name: str
    type: str
    roles: tuple[str, ...]
    cards: tuple[int, ...]
    gather: FrozenArray
    scatter: FrozenArray
    arrays: ArrayRequirement


@dataclass(frozen=True)
class CompositeOwnership(ImmutableRecord):
    """Factor execution is distinct from shared parameter/objective overlays."""

    plan: str
    executed: tuple[str, ...]
    absorbed: tuple[str, ...] = ()
    constraints: tuple[tuple[str, str], ...] = ()
    excluded_edges: tuple[str, ...] = ()
    parameter_owners: tuple[str, ...] = ()
    parameter_edges: tuple[str, ...] = ()


@dataclass(frozen=True)
class SpecializedPlanDrafts(ImmutableRecord):
    dirichlet: tuple[PlanDraft[DirichletGroupPlan], ...]
    gfe: tuple[PlanDraft[GfePlan], ...]
    gfe_goals: Mapping[str, tuple[str, ...]]
    transition_gfe: tuple[PlanDraft[GfeTmPlan], ...]
    hybrid_gfe: tuple[PlanDraft[HybridGfePlan], ...]
    mixtures: tuple[PlanDraft[MixGroupPlan], ...]
    recurrence: tuple[PlanDraft[RecurrencePlan], ...]
    recurrence_components: tuple[RecurrenceComponentPlan, ...]
    precision: tuple[PlanDraft[PrecGroupPlan], ...]
    constraints: tuple[ConstraintDraft, ...]
    equalities: Mapping[str, GroupPlan | GaussGroupPlan | EfGroupPlan]
    mu3: tuple[Mu3Plan, ...]
    goal_parameters: tuple[GoalParameterPlan, ...]
    goal_parameter_problem: str | None
    ownership: tuple[CompositeOwnership, ...]
    ownership_problem: str | None
    delta_pairs: FrozenArray
    delta_problem: str | None

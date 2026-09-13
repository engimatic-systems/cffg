"""Read-only structural contracts; Arena supplies them without copies."""

from collections.abc import Mapping
from typing import Protocol

from cffg.compiled import (
    DirichletGroupPlan,
    EdgeInfo,
    EfGroupPlan,
    GaussGroupPlan,
    GfePlan,
    GoalParameterPlan,
    GroupPlan,
    HybridGfePlan,
    MixGroupPlan,
    RecurrencePlan,
)
from cffg.execution.transport import GaussianLayout, MessageLayout, ObservationLayout


class EdgesView(Protocol):
    @property
    def edges(self) -> Mapping[str, EdgeInfo]: ...


class DiscreteView(MessageLayout, EdgesView, Protocol):
    pass


class NodeView(MessageLayout, Protocol):
    @property
    def groups(self) -> tuple[GroupPlan, ...]: ...

    @property
    def dirichlet_groups(self) -> tuple[DirichletGroupPlan, ...]: ...


class GoalView(Protocol):
    @property
    def gfe_groups(self) -> tuple[GfePlan, ...]: ...

    @property
    def goal_parameter_plans(self) -> tuple[GoalParameterPlan, ...]: ...


class GaussianView(GaussianLayout, EdgesView, Protocol):
    pass


class MixtureView(MessageLayout, GaussianLayout, Protocol):
    @property
    def mix_groups(self) -> tuple[MixGroupPlan, ...]: ...


class GaussianStatisticsView(MixtureView, ObservationLayout, Protocol):
    @property
    def gauss_groups(self) -> tuple[GaussGroupPlan, ...]: ...


class RecurrenceView(MessageLayout, GaussianLayout, Protocol):
    @property
    def recurrence_groups(self) -> tuple[RecurrencePlan, ...]: ...


class HybridGfeView(MessageLayout, GaussianLayout, Protocol):
    @property
    def hybrid_gfe_groups(self) -> tuple[HybridGfePlan, ...]: ...


class KernelView(MessageLayout, GaussianLayout, ObservationLayout, Protocol):
    @property
    def groups(self) -> tuple[GroupPlan, ...]: ...

    @property
    def gauss_groups(self) -> tuple[GaussGroupPlan, ...]: ...

    @property
    def ef_groups(self) -> tuple[EfGroupPlan, ...]: ...

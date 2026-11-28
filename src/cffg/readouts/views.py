"""Read-only structural contracts; Arena supplies them without copies."""

from collections.abc import Mapping
from typing import Protocol

from cffg._immutable import FrozenArray
from cffg.core.ir import PortSpec
from cffg.compiled import (
    ConstraintPlan,
    DirichletGroupPlan,
    EdgeInfo,
    EfGroupPlan,
    GaussGroupPlan,
    GfePlan,
    GfeTmPlan,
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


class SensingView(EdgesView, NodeView, Protocol):
    @property
    def endpoints(self) -> Mapping[str, tuple[tuple[str, str], ...]]: ...

    @property
    def ports(self) -> Mapping[str, PortSpec]: ...

    @property
    def data_slots(self) -> Mapping[str, int]: ...

    @property
    def message_starts(self) -> FrozenArray: ...

    @property
    def semiring(self) -> str: ...

    @property
    def gfe_groups(self) -> tuple[GfePlan, ...]: ...

    @property
    def gfe_tm_groups(self) -> tuple[GfeTmPlan, ...]: ...

    @property
    def hybrid_gfe_groups(self) -> tuple[HybridGfePlan, ...]: ...

    @property
    def constraint_groups(self) -> tuple[ConstraintPlan, ...]: ...

    @property
    def delta_pairs(self) -> FrozenArray: ...


class GoalView(Protocol):
    @property
    def gfe_groups(self) -> tuple[GfePlan, ...]: ...

    @property
    def goal_parameter_plans(self) -> tuple[GoalParameterPlan, ...]: ...


class GaussianView(GaussianLayout, ObservationLayout, EdgesView, Protocol):
    @property
    def gaussian_taps(self) -> Mapping[str, int]: ...


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

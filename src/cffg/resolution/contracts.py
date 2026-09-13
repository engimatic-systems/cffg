"""Complete immutable numeric and address data, without an arena or schedule."""

from collections.abc import Mapping
from dataclasses import dataclass

from cffg._immutable import FrozenArray, ImmutableRecord
from cffg.compiled import (
    ConstraintPlan,
    EfGroupPlan,
    GaussGroupPlan,
    GoalParameterPlan,
    GroupPlan,
    LearnPlan,
    MemberAddress,
    Mu3Plan,
)


@dataclass(frozen=True)
class ResolvedCompileData(ImmutableRecord):
    template_name: str
    log_params: Mapping[str, FrozenArray]
    groups: tuple[GroupPlan, ...]
    gauss_groups: tuple[GaussGroupPlan, ...]
    ef_groups: tuple[EfGroupPlan, ...]
    constraint_groups: tuple[ConstraintPlan, ...]
    gauss_data: Mapping[str, tuple[int, int]]
    n_obs_rows: int
    ef_data: Mapping[str, tuple[int, str]]
    n_ef_obs: int
    message_starts: FrozenArray
    g_eta_starts: FrozenArray
    g_lam_starts: FrozenArray
    g_obs_starts: FrozenArray
    member_of: Mapping[str, MemberAddress]
    writable_members: frozenset[str]
    learn_plans: tuple[LearnPlan, ...]
    mu3: tuple[Mu3Plan, ...]
    goal_parameters: tuple[GoalParameterPlan, ...]

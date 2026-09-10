"""Select the numerical parameter blocks consumed by explicit plan owners."""

from collections.abc import Mapping
from typing import NamedTuple

import jax

from cffg.compiled import (
    ConstraintPlan,
    DirichletGroupPlan,
    GfePlan,
    GfeTmPlan,
    HybridGfePlan,
    MixGroupPlan,
    PrecGroupPlan,
    RecurrencePlan,
)

type Parameters = Mapping[str, jax.Array]


class MixtureParameters(NamedTuple):
    precision: jax.Array
    matrix: jax.Array | None = None
    offset: jax.Array | None = None
    mean: jax.Array | None = None


class HybridParameters(NamedTuple):
    channel: MixtureParameters
    goal_mean: jax.Array
    goal_precision: jax.Array


class RecurrenceParameters(NamedTuple):
    weights: jax.Array
    offsets: jax.Array


class GfeParameters(NamedTuple):
    """Plan-selected channel representation and fixed or learned goal data.

    A channel is fixed log probabilities or Dirichlet counts, selected by the
    plan's learned flag. Functional plans instead use mean/column entropy.
    Goal data is either log probabilities or prior concentrations. Absent
    fields describe static plan variants, not missing runtime observations.
    """

    channel: jax.Array | None
    mean: jax.Array | None
    column_entropy: jax.Array | None
    log_goal: jax.Array | None
    goal_prior: jax.Array | None


class ConstraintParameters(NamedTuple):
    threshold: jax.Array | None = None
    table: jax.Array | None = None


def mixture_parameters(
    plan: MixGroupPlan | HybridGfePlan, values: Parameters
) -> MixtureParameters:
    precision = values[plan.name + ":W"]
    if plan.type == "gaussian_mixture_linear":
        return MixtureParameters(
            precision, values[plan.name + ":A"], values[plan.name + ":b"]
        )
    return MixtureParameters(precision, mean=values[plan.name + ":m"])


def hybrid_parameters(plan: HybridGfePlan, values: Parameters) -> HybridParameters:
    return HybridParameters(
        mixture_parameters(plan, values),
        values[plan.name + ":goal_m"],
        values[plan.name + ":goal_W"],
    )


def recurrence_parameters(
    plan: RecurrencePlan, values: Parameters
) -> RecurrenceParameters:
    return RecurrenceParameters(
        values[plan.name + ":weights"], values[plan.name + ":offsets"]
    )


def precision_mean(plan: PrecGroupPlan, values: Parameters) -> jax.Array:
    return values[plan.name + ":m"]


def dirichlet_prior(plan: DirichletGroupPlan, values: Parameters) -> jax.Array:
    return values[plan.name + ":alpha0"]


def transition_parameters(plan: GfeTmPlan, values: Parameters) -> jax.Array:
    return values[plan.name + (":counts" if plan.learned else ":B")]


def gfe_parameters(plan: GfePlan, values: Parameters) -> GfeParameters:
    return GfeParameters(
        None if plan.mu3 else values[plan.name + (":counts" if plan.learned else ":A")],
        values[plan.name + ":Abar"] if plan.mu3 else None,
        values[plan.name + ":hbar"] if plan.mu3 else None,
        None if plan.goal_learned else values[plan.name + ":c"],
        values[plan.name + ":c_alpha0"] if plan.goal_learned else None,
    )


def constraint_parameters(
    plan: ConstraintPlan, values: Parameters
) -> ConstraintParameters:
    """Select this instance's data without capturing another arena's defaults."""
    if plan.type in ("exactly_k", "at_most_k"):
        return ConstraintParameters(threshold=values[plan.parameter_key])
    if plan.type == "table":
        return ConstraintParameters(table=values[plan.parameter_key])
    return ConstraintParameters()

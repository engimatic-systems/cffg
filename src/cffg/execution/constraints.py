"""Hard-constraint updates from explicit numerical threshold/table inputs."""

from collections.abc import Callable

import jax

from cffg import csp
from cffg.compiled import ConstraintPlan
from cffg.execution.contracts import ExecutionDelta
from cffg.execution.parameters import ConstraintParameters


def execute_constraint(
    plan: ConstraintPlan,
    incoming: jax.Array,
    params: ConstraintParameters,
    reduction: Callable[..., jax.Array],
) -> ExecutionDelta[jax.Array]:
    rule = csp.CONSTRAINT_MESSAGES[plan.type]
    if plan.type in ("exactly_k", "at_most_k"):
        outgoing = rule(
            incoming, plan.cards, max(plan.cards), reduction, k=params.threshold
        )
    elif plan.type == "table":
        outgoing = rule(
            incoming, plan.cards, max(plan.cards), reduction, table=params.table
        )
    else:
        outgoing = rule(incoming, plan.cards, max(plan.cards), reduction)
    return ExecutionDelta(outgoing)

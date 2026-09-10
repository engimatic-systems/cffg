"""Registered local updates shared by flat and register-carried transport."""

from collections.abc import Callable, Mapping

import jax
import numpy as np

from cffg import efrules, kernel_runtime, kernels
from cffg._immutable import FrozenArray
from cffg.compiled import EfGroupPlan, GaussGroupPlan, GroupPlan
from cffg.execution.contracts import ExecutionDelta, GaussianMessage

type Parameters = Mapping[str, jax.Array]
type Rows = jax.Array | np.ndarray | FrozenArray | None
type GaussianMessages = tuple[GaussianMessage, ...]


def execute_discrete(
    plan: GroupPlan,
    incoming: jax.Array,
    params: Parameters,
    reduction: Callable[..., jax.Array],
) -> ExecutionDelta[jax.Array]:
    return ExecutionDelta(
        kernel_runtime.discrete_message(plan, incoming, params, reduction)
    )


def execute_gaussian(
    plan: GaussGroupPlan,
    eta: jax.Array,
    precision: jax.Array,
    params: Parameters,
    observation: jax.Array | None = None,
    mask: jax.Array | None = None,
) -> ExecutionDelta[GaussianMessages]:
    outgoing = kernel_runtime.gaussian_message(
        plan, eta, precision, params, observation, mask
    )
    return ExecutionDelta(tuple(GaussianMessage(e, p) for e, p in outgoing))


def ef_parameters(
    plan: EfGroupPlan, params: Parameters, rows: Rows = None
) -> dict[str, jax.Array]:
    if plan.type == "equality":
        return {}
    kernel = kernels.KERNELS[plan.type]
    return {
        key: params[f"{plan.name}:{key}"]
        if rows is None
        else params[f"{plan.name}:{key}"][rows]
        for key in kernel.param_keys
    }


def execute_ef(
    plan: EfGroupPlan,
    incoming: jax.Array,
    params: Parameters,
    observation: jax.Array | None = None,
    mask: jax.Array | None = None,
) -> ExecutionDelta[jax.Array]:
    if plan.type == "equality":
        outgoing = efrules.equality_messages(incoming)
    else:
        outgoing = kernels.KERNELS[plan.type].message(
            incoming, params, observation, mask
        )
    return ExecutionDelta(outgoing)

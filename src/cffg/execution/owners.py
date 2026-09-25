"""Typed local execution capabilities available to schedule construction."""

from collections.abc import Callable
from dataclasses import dataclass

import jax

from cffg.compiled import EfGroupPlan, GaussGroupPlan, GroupPlan
from cffg.execution.contracts import ExecutionDelta
from cffg.execution.registered import (
    GaussianMessages,
    Parameters,
    Rows,
    ef_parameters,
    execute_discrete,
    execute_ef,
    execute_gaussian,
)
from cffg.execution.support import discrete_zero_support
from cffg.kernel_runtime import parameters

type Reduction = Callable[..., jax.Array]
type LocalParameters = dict[str, jax.Array]
type OptionalArray = jax.Array | None


@dataclass(frozen=True)
class ScheduledExecutors:
    """Trusted host callables, never part of serialized model or State data.

    These are the same local owners used by flooding. The schedule transports
    their inputs and outputs; parameter meanings and factor mathematics stay
    behind these capabilities. Optional local facts do not decide termination.
    """

    discrete: Callable[
        [GroupPlan, jax.Array, Parameters, Reduction], ExecutionDelta[jax.Array]
    ] = execute_discrete
    gaussian: Callable[
        [
            GaussGroupPlan,
            jax.Array,
            jax.Array,
            Parameters,
            OptionalArray,
            OptionalArray,
        ],
        ExecutionDelta[GaussianMessages],
    ] = execute_gaussian
    exponential_family: Callable[
        [EfGroupPlan, jax.Array, Parameters, OptionalArray, OptionalArray],
        ExecutionDelta[jax.Array],
    ] = execute_ef
    parameters: Callable[
        [GroupPlan | GaussGroupPlan, Parameters, Rows], LocalParameters
    ] = parameters
    ef_parameters: Callable[[EfGroupPlan, Parameters, Rows], LocalParameters] = (
        ef_parameters
    )
    discrete_zero_support: Callable[
        [GroupPlan, tuple[jax.Array, ...], LocalParameters], jax.Array
    ] = discrete_zero_support


SCHEDULED_EXECUTORS = ScheduledExecutors()

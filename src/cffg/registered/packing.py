"""Registered packing adapters; numerical definitions stay in kernel owners."""

import numpy as np

from cffg import kernels
from cffg.compiler_errors import CompileError
from cffg.core.ir import Template


def pack_discrete(template, draft, resolved, semiring):
    if draft.problem:
        raise CompileError(draft.problem)
    if draft.path_problem:
        raise ValueError(draft.path_problem)
    plan, kernel = draft.plan, draft.plan.kernel
    try:
        packed = kernels.packed_parameters(
            kernel, template, list(plan.members), resolved
        )
        from cffg.kernel_runtime import validate_discrete_shapes

        validate_discrete_shapes(plan, packed, semiring)
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise CompileError(f"kernel {kernel.impl_id!r}: {error}") from error
    paths = dict(plan.parameter_paths)
    return {paths[key]: value for key, value in packed.items()}


def pack_gaussian(template, draft, plan, resolved):
    kernel = draft.kernel
    try:
        from cffg.kernel_runtime import validate_gaussian_shapes

        packed = kernels.packed_parameters(
            kernel, template, list(draft.members), resolved
        )
        validate_gaussian_shapes(plan, packed)
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise CompileError(f"kernel {kernel.impl_id!r}: {error}") from error
    paths = dict(draft.parameter_paths)
    return {paths[key]: value for key, value in packed.items()}


def pack_ef(template, draft, resolved):
    """Retain the legacy callback and key checks, without new array validation."""
    kernel = draft.kernel
    if kernel.pack_params is None:
        return {}
    try:
        packed = kernel.pack_params(template, list(draft.members), resolved)
    except ValueError as error:
        raise CompileError(str(error)) from error
    if set(packed) != set(kernel.param_keys):
        raise CompileError(
            f"kernel {kernel.impl_id!r}: pack_params returned keys {sorted(packed)}, "
            f"declared param_keys {sorted(kernel.param_keys)}"
        )
    return {f"{draft.name}:{key}": arr for key, arr in packed.items()}


def pack_gaussian_parameters(kernel, template, members, resolved):
    """An auxiliary Gaussian goal uses the original restricted source mapping."""
    if kernel is None or kernel.arena != "gaussian":
        raise CompileError("factor type 'gaussian': no Gaussian binding")
    try:
        return kernels.packed_parameters(kernel, template, members, resolved)
    except ValueError as error:
        raise CompileError(str(error)) from error


def _stack_params(
    t: Template,
    ftype: str,
    cards: tuple[int, ...],
    members: list[str],
    resolved: dict[str, np.ndarray],
) -> np.ndarray | None:
    """Stack per-member parameter arrays into [M, ...] log-space, with shape
    and column-stochasticity checks ("true FE" needs normalized conditionals)."""
    if ftype == "equality":
        return None
    kern = kernels.kernel_for(ftype)
    if kern is None or kern.arena != "discrete":
        raise CompileError(f"factor type {ftype!r}: no discrete binding")
    packed = kernels.packed_parameters(kern, t, members, resolved)
    return np.asarray(kern.binding.potential(packed, cards))


def _stack_gauss_params(
    t: Template,
    ftype: str,
    dims: tuple[int, ...],
    members: list[str],
    resolved: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Per-member Gaussian parameters, stacked [M, ...]. Covariances are
    validated SPD and stored as precisions W = V⁻¹ (the form every rule
    consumes); the compile-time Cholesky is also the SPD check."""
    if ftype == "equality":
        return {}
    kern = kernels.kernel_for(ftype)
    if kern is None or kern.arena != "gaussian":
        raise CompileError(f"factor type {ftype!r}: no Gaussian binding")
    try:
        return kernels.packed_parameters(kern, t, members, resolved)
    except ValueError as err:
        raise CompileError(str(err)) from err

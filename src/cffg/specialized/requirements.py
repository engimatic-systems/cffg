"""Describe source arrays without loading them or choosing factor meaning."""

from cffg import kernels

from .contracts import ArrayDestination, ArrayInput, ArrayRequirement


def array_requirement(claims, owner, recipe, members, keys=(), *, payload=None):
    inputs = []
    topology = claims.topology
    for name in members:
        factor = topology.factors[name]
        kernel = kernels.kernel_for(factor.type)
        sizes = {
            role: topology.variables[variable].dim
            or topology.variables[variable].cardinality
            for role, variable in factor.edges.items()
        }
        for role, blob in sorted(factor.param_blobs.items()):
            shape = claims.initializer_shapes.get((name, role))
            if kernel is not None and role in kernel.binding.parameters:
                shape = tuple(kernel.binding.parameters[role].shape(sizes))
            inputs.append(ArrayInput(name, role, blob, shape))
    destinations = tuple(ArrayDestination("parameter", key) for key in keys)
    if payload is not None:
        destinations += (ArrayDestination("payload", payload),)
    return ArrayRequirement(owner, recipe, tuple(inputs), destinations)

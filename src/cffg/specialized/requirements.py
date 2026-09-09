"""Describe source arrays without loading them or choosing factor meaning."""

from .contracts import ArrayDestination, ArrayInput, ArrayRequirement


def array_requirement(claims, owner, recipe, members, keys=(), *, payload=None):
    inputs = []
    topology = claims.topology
    for name in members:
        factor = topology.factors[name]
        for role, blob in sorted(factor.param_blobs.items()):
            shape = claims.initializer_shapes.get((name, role))
            inputs.append(ArrayInput(name, role, blob, shape))
    destinations = tuple(ArrayDestination("parameter", key) for key in keys)
    if payload is not None:
        destinations += (ArrayDestination("payload", payload),)
    return ArrayRequirement(owner, recipe, tuple(inputs), destinations)

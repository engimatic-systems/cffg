"""Describe source arrays without loading them or choosing factor meaning."""

from cffg._immutable import FrozenMap
from cffg.array_requirements import (
    ArrayDestination,
    ArrayInput,
    ArrayRequirement,
    MemberRequirement,
)


def array_requirement(
    claims,
    owner,
    recipe,
    members,
    keys=(),
    *,
    payload=None,
    addresses=(),
    auxiliary_kernels=None,
):
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
    return ArrayRequirement(
        owner,
        recipe,
        tuple(inputs),
        destinations,
        addresses,
        auxiliary_kernels or {},
    )


def member_requirements(members, runtime, *, writable=True, statistic_source=True):
    runtime = FrozenMap(runtime)
    return tuple(
        MemberRequirement(member, index, runtime, writable, statistic_source)
        for index, member in enumerate(members)
    )

"""Declared source interpretation and member writes for registered factors."""

from cffg._immutable import FrozenMap
from cffg.array_requirements import (
    MemberRequirement,
    ParameterRuntime,
    ShapeRequirement,
    SourceRequirement,
)


def source_requirements(topology, registry):
    """Include absorbed members, preserving original factor/role traversal."""
    sources = []
    for factor in topology.factors.values():
        kernel = registry.bindings.get(factor.type)
        parameters = kernel.binding.parameters if kernel is not None else {}
        sizes = FrozenMap(
            {
                role: topology.variables[variable].dim
                or topology.variables[variable].cardinality
                for role, variable in factor.edges.items()
            }
        )
        for role, blob in factor.param_blobs.items():
            parameter = parameters.get(role)
            sources.append(
                SourceRequirement(
                    factor.name,
                    role,
                    blob,
                    ShapeRequirement(parameter, sizes)
                    if parameter is not None
                    else None,
                    parameter is not None
                    and parameter.transform == "covariance_to_precision",
                    () if factor.type in ("gamma", "beta") else None,
                )
            )
    return tuple(sources)


def member_requirements(topology, name, members, kernel, paths):
    """Keep legacy direct-write and learning layout decisions with the binding."""
    if kernel is None or kernel.arena == "ef":
        return ()
    specs = kernel.binding.parameters
    if not all(spec.key is not None for spec in specs.values()):
        return ()
    runtime = FrozenMap(
        {
            role: ParameterRuntime(
                paths[spec.key],
                spec.transform,
                spec.probability_axis,
                spec.permutation,
                learning_layout=(
                    "tm"
                    if kernel.factor_type.name == "transition_mixture"
                    and paths[spec.key] == name
                    else "as_is"
                ),
            )
            for role, spec in specs.items()
        }
    )
    return tuple(
        MemberRequirement(
            member,
            index,
            runtime,
            writable=kernel.arena == "discrete"
            or bool(topology.factors[member].param_blobs),
        )
        for index, member in enumerate(members)
    )

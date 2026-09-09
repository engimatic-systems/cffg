"""Registry snapshots and declaration checks at existing admission points."""

from cffg import kernels
from cffg.core.ir import FACTOR_TYPES, ConstraintKind, PortRole

from .contracts import RegistrySnapshot


def snapshot_registry() -> RegistrySnapshot:
    bindings = dict(kernels.KERNELS)
    return RegistrySnapshot(
        bindings,
        frozenset(
            name for name, kernel in bindings.items() if kernel.observed_out is not None
        ),
    )


def constraint_problems(factor, registry, *, mean_field, psub, delegated_psub):
    kernel = registry.bindings.get(factor.type)
    if kernel is not None and not delegated_psub:
        requested = set()
        if mean_field:
            requested.add(ConstraintKind.MEAN_FIELD.value)
        if psub:
            requested.add(ConstraintKind.P_SUBSTITUTION.value)
        bad = sorted(requested - set(kernel.constraint_modes))
        if bad:
            return (
                (
                    f"factor {factor.name!r}: kernel {kernel.impl_id!r} does not accept "
                    f"{bad} beads (conjugate BP is already exact on EF subgraphs)"
                ),
            )
    elif FACTOR_TYPES[factor.type].family == "ef" and (mean_field or psub):
        return (
            (
                f"factor {factor.name!r}: beads on EF nodes are undefined in M-scope — "
                "conjugate BP is already exact on EF subgraphs (cff-8s6a)"
            ),
        )
    return ()


def conditioned_output_problems(factor, topology, registry):
    if factor.type not in registry.conditioned_out_types:
        return ()
    variable = factor.edges["out"]
    ports = topology.ports_by_variable.get(variable, ())
    if topology.degrees.get(variable, 0) == 2 or not any(
        p.role is PortRole.DATA for p in ports
    ):
        return (
            (
                f"factor {factor.name!r}: {factor.type} is a conditioned likelihood — its 'out' "
                "edge must be a DATA port (unobserved predictives leave the family)"
            ),
        )
    return ()

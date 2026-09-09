"""Registered same-arena structure, without loading arrays or assigning rows."""

import numpy as np

from cffg import kernels
from cffg._immutable import FrozenMap
from cffg.array_requirements import (
    ArrayDestination,
    ArrayInput,
    ArrayRecipe,
    ArrayRequirement,
    ShapeRequirement,
)
from cffg.compiled import GroupPlan
from cffg.core.ir import FACTOR_TYPES, Template
from cffg.specialized.contracts import SpecializedClaims
from cffg.topology import TopologyLayout

from .contracts import (
    DiscreteDraft,
    EfDraft,
    EqualityReference,
    GaussianDraft,
    ObservationIntent,
    ObservationSource,
    RegisteredPlanDrafts,
    RegistrySnapshot,
)


def _conditioning(topology, claims, registry):
    gaussian, ef, discrete, problems = {}, {}, {}, []
    for candidate in topology.data_candidates:
        variable = topology.variables[candidate.variable]
        member, role = candidate.endpoint
        kind = topology.factors[member].type
        if variable.dim:
            kernel = registry.bindings.get(kind)
            conditioning = (
                kernel.binding.conditioning
                if kernel is not None and kernel.arena == "gaussian"
                else None
            )
            if conditioning is not None and role == conditioning.role:
                gaussian[candidate.variable] = candidate.port
            elif candidate.port in claims.conditioning_problems:
                problems.append(claims.conditioning_problems[candidate.port])
            else:
                problems.append(
                    f"port {candidate.port!r}: Gaussian DATA port on {member!r}/{role!r} "
                    "requires a registered conditioning role"
                )
        elif variable.ef:
            if kind in registry.conditioned_out_types and role == "out":
                ef[candidate.variable] = candidate.port
            else:
                problems.append(
                    f"port {candidate.port!r}: EF DATA ports must clamp a conditioned "
                    f"observation kernel's 'out' edge "
                    f"({sorted(registry.conditioned_out_types)}), not "
                    f"{member!r}/{role!r}"
                )
        else:
            discrete[candidate.port] = candidate.external_slot
    return gaussian, ef, discrete, tuple(problems)


def _group_keys(t, claims, gauss_data_ports, registry):
    """Compatibility membership product; specialized routes are already chosen.

    This read-only pass runs before draft completion. Its first error is raised
    only at the former remaining-factor grouping checkpoint, after GFE packing.
    """
    buckets, gauss_buckets, ef_buckets = {}, {}, {}
    problem = None
    specialized_members = set(claims.precision) | set(claims.constraints)
    specialized_members.update(
        member for claim in claims.mixtures for member in claim.members
    )
    specialized_members.update(
        member for members in claims.recurrence.values() for member in members
    )
    for fname in sorted(t.factors):
        if fname in claims.consumed:
            continue
        if fname in claims.grouping_problems:
            problem = problem or claims.grouping_problems[fname]
            continue
        if fname in specialized_members:
            continue
        if fname in claims.equalities:
            route = claims.equalities[fname]
            destination = {
                "discrete": buckets,
                "gaussian": gauss_buckets,
                "ef": ef_buckets,
            }[route.arena]
            destination.setdefault(route.key, []).append(fname)
            continue
        f = t.factors[fname]
        ft = FACTOR_TYPES[f.type]
        roles = ft.edge_roles
        if ft.family == "gaussian" or (
            ft.family == "any" and t.variables[f.edges[roles[0]]].dim > 0
        ):
            dims = tuple(t.variables[f.edges[r]].dim for r in roles)
            kern = registry.bindings.get(f.type)
            if kern is None or kern.arena != "gaussian":
                problem = (
                    problem or f"factor type {f.type!r} has no Gaussian kernel binding"
                )
                continue
            conditioning = kern.binding.conditioning
            observed = (
                conditioning is not None
                and f.edges[conditioning.role] in gauss_data_ports
            )
            gtype = (
                (conditioning.plan_type or f"{f.type}:observed:{conditioning.role}")
                if observed
                else f.type
            )
            gauss_buckets.setdefault((gtype, dims, f.type, observed), []).append(fname)
        elif ft.family == "ef" or (
            ft.family == "any" and bool(t.variables[f.edges[roles[0]]].ef)
        ):
            kern = registry.bindings.get(f.type)
            if kern is None:
                problem = problem or (
                    f"factor {fname!r}: EF type {f.type!r} has no registered "
                    "FactorKernel — register one before compiling (cff-ri8m)"
                )
                continue
            ef_buckets.setdefault((f.type, kern.family), []).append(fname)
        else:
            cards = tuple(t.variables[f.edges[r]].cardinality for r in roles)
            mode = "vmp" if fname in claims.mean_field else "bp"
            buckets.setdefault((f.type, cards, mode), []).append(fname)
    return buckets, gauss_buckets, ef_buckets, problem


def _addresses(topology, members, roles):
    gather = np.array(
        [[topology.opposite[member, role] for role in roles] for member in members],
        dtype=np.int64,
    )
    scatter = np.array(
        [[topology.slot_of[member, role] for role in roles] for member in members],
        dtype=np.int64,
    )
    return gather, scatter


def _parameter_paths(kernel, name):
    if kernel is None:
        return {}, None
    try:
        return kernels.parameter_paths(kernel, name), None
    except ValueError as error:
        return {}, str(error)


def _arrays(topology, name, members, kernel, paths):
    inputs = []
    for member in members:
        factor = topology.factors[member]
        sizes = FrozenMap(
            {
                role: topology.variables[variable].dim
                or topology.variables[variable].cardinality
                for role, variable in factor.edges.items()
            }
        )
        for role, blob in sorted(factor.param_blobs.items()):
            parameter = (
                kernel.binding.parameters.get(role) if kernel is not None else None
            )
            pending = (
                ShapeRequirement(parameter, sizes) if parameter is not None else None
            )
            inputs.append(ArrayInput(member, role, blob, None, pending))
    return ArrayRequirement(
        name,
        ArrayRecipe.REGISTERED,
        tuple(inputs),
        tuple(ArrayDestination("parameter", value) for value in paths.values()),
    )


def _observation(topology, members, role, role_index, ports):
    return ObservationIntent(
        role,
        role_index,
        tuple(
            ObservationSource(
                member,
                ports.get(topology.factors[member].edges[role]),
                topology.factors[member].edges[role],
                topology.variables[topology.factors[member].edges[role]].dim,
                topology.variables[topology.factors[member].edges[role]].ef,
            )
            for member in members
        ),
    )


def lower_registered(
    template: Template,
    topology: TopologyLayout,
    claims: SpecializedClaims,
    semiring: str,
    registry: RegistrySnapshot,
) -> RegisteredPlanDrafts:
    """Capture registered metadata; preserve all numeric/callback checkpoints."""
    gauss_ports, ef_ports, data_slots, conditioning_problems = _conditioning(
        topology, claims, registry
    )
    discrete_keys, gaussian_keys, ef_keys, grouping_problem = _group_keys(
        topology,
        claims,
        gauss_ports,
        registry,
    )
    discrete, gaussian, ef = [], [], []
    for key, members in sorted(discrete_keys.items()):
        if members[0] in claims.equalities:
            discrete.append(EqualityReference(key, tuple(members)))
            continue
        kind, cards, mode = key
        roles = FACTOR_TYPES[kind].edge_roles
        name = f"{mode}:{kind}:{'x'.join(map(str, cards))}"
        gather, scatter = _addresses(topology, members, roles)
        kernel = registry.bindings.get(kind)
        problem = None
        if kernel is None or kernel.arena != "discrete":
            problem = f"factor type {kind!r} has no discrete kernel binding"
        elif semiring not in kernel.binding.semirings:
            problem = f"kernel {kernel.impl_id!r} does not support {semiring!r}"
        paths, path_problem = (
            _parameter_paths(kernel, name) if problem is None else ({}, None)
        )
        plan = GroupPlan(
            name,
            kind,
            roles,
            cards,
            tuple(members),
            gather,
            scatter,
            mode,
            np.stack([scatter, gather], axis=-1),
            kernel,
            tuple(paths.items()),
        )
        discrete.append(
            DiscreteDraft(
                plan,
                _arrays(topology, name, members, kernel, paths),
                problem,
                path_problem,
            )
        )
    for key, members in sorted(gaussian_keys.items()):
        if members[0] in claims.equalities:
            gaussian.append(EqualityReference(key, tuple(members)))
            continue
        kind, dims, base, observed = key
        kernel = registry.bindings[base]
        roles = FACTOR_TYPES[base].edge_roles
        name = f"gauss:{kind}:{'x'.join(map(str, dims))}"
        gather, scatter = _addresses(topology, members, roles)
        observation = None
        if observed:
            role = kernel.binding.conditioning.role
            observation = _observation(
                topology, members, role, roles.index(role), gauss_ports
            )
        paths, path_problem = _parameter_paths(kernel, name)
        gaussian.append(
            GaussianDraft(
                name,
                kind,
                roles,
                dims,
                tuple(members),
                gather,
                scatter,
                kernel,
                tuple(paths.items()),
                observation,
                _arrays(topology, name, members, kernel, paths),
                path_problem,
            )
        )
    for key, members in sorted(ef_keys.items()):
        if members[0] in claims.equalities:
            ef.append(EqualityReference(key, tuple(members)))
            continue
        kind, family = key
        kernel = registry.bindings[kind]
        roles = kernel.message_roles
        name = f"ef:{kind}:{family}"
        gather, scatter = _addresses(topology, members, roles)
        observation = None
        if kernel.observed_out is not None:
            observation = _observation(topology, members, "out", None, ef_ports)
        paths = {key: f"{name}:{key}" for key in kernel.param_keys}
        ef.append(
            EfDraft(
                name,
                kind,
                family,
                roles,
                tuple(members),
                gather,
                scatter,
                kernel,
                observation,
                _arrays(topology, name, members, kernel, paths),
            )
        )
    return RegisteredPlanDrafts(
        discrete=discrete,
        gaussian=gaussian,
        ef=ef,
        bp_members=frozenset(
            member
            for (_, _, mode), members in discrete_keys.items()
            if mode == "bp"
            for member in members
        ),
        data_slots=data_slots,
        conditioning_problems=conditioning_problems,
        grouping_problem=grouping_problem,
    )

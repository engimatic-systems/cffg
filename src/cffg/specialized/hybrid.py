"""Declared D12 graph ownership and unresolved per-factor requirements.

This stage preserves general admitted region incidence. It never maps scopes
to a numerical chain or calls an initializer, packer, solver or array loader.
"""

from dataclasses import replace
from urllib.parse import quote

import numpy as np

from cffg.array_requirements import (
    ArrayDestination,
    ArrayInput,
    ArrayRecipe,
    ArrayRequirement,
    ParameterRuntime,
)
from cffg.compiled_hybrid import (
    HybridAdapterPlan,
    HybridComponentPlan,
    HybridRegionPlan,
    HybridSeparatorPlan,
)
from cffg.core.hybrid import hybrid_problems
from cffg.kernels import parameter_paths
from cffg.registered.requirements import (
    member_requirements as registered_members,
    source_requirements,
)

from .contracts import (
    CompositeOwnership,
    HybridComponentClaim,
    HybridComponentDraft,
    HybridObjectiveOwner,
)
from .requirements import member_requirements


def _copies(topology, declaration):
    """Identify copies only through the component's owned equality incidence."""
    neighbors = {}
    for name in declaration.equalities:
        variables = tuple(topology.factors[name].edges.values())
        first = variables[0]
        for variable in variables:
            neighbors.setdefault(first, set()).add(variable)
            neighbors.setdefault(variable, set()).add(first)
    copies = {}
    for separator in declaration.separators:
        for variable in (separator.regime, separator.state):
            if variable in copies:
                continue
            pending, reached = [variable], set()
            while pending:
                current = pending.pop()
                if current in reached:
                    continue
                reached.add(current)
                pending.extend(neighbors.get(current, ()))
            group = tuple(sorted(reached))
            copies.update((copy, group) for copy in group)
    return copies


def discover_hybrid_components(topology):
    """Check snapshot graph consistency before ordinary owners can claim it.

    Diagnostics remain declarations for the specialized admission gate. This
    validates graph scopes, not resolved support or final assembled ownership.
    """
    problems = tuple(hybrid_problems(topology))
    if problems:
        return (), problems
    claims = []
    for _, declaration in sorted(topology.hybrid_components.items()):
        copies = _copies(topology, declaration)
        separators = tuple(
            HybridSeparatorPlan(
                separator.name,
                topology.variables[separator.regime].cardinality,
                topology.variables[separator.state].dim,
                copies[separator.regime],
                copies[separator.state],
            )
            for separator in sorted(declaration.separators, key=lambda s: s.name)
        )
        indices = {separator.name: i for i, separator in enumerate(separators)}
        regions = tuple(
            HybridRegionPlan(region.name, np.asarray(
                [indices[port] for port in region.ports], dtype=np.int64,
            ))
            for region in sorted(declaration.regions, key=lambda r: r.name)
        )
        owned = {region.factor for region in declaration.regions} | set(declaration.equalities)
        adapters = []
        for adapter in sorted(declaration.adapters, key=lambda a: a.name):
            edge = topology.edges[adapter.variable]
            inside = [end for end in topology.endpoints[adapter.variable] if end[0] in owned]
            if len(inside) != 1:
                problems += (f"hybrid component {declaration.name!r}: adapter {adapter.name!r} "
                             "requires exactly one component endpoint",)
                continue
            adapters.append(HybridAdapterPlan(
                adapter.name, indices[adapter.separator],
                "gaussian" if edge.dim else "categorical",
                (edge.slot0, edge.slot1), topology.slot_of[inside[0]],
            ))
        if problems:
            continue
        plan = HybridComponentPlan(declaration, separators, regions, tuple(adapters))
        internal = tuple(sorted(
            variable for variable, ends in topology.endpoints.items()
            if len(ends) == 2 and all(name in owned for name, _ in ends)
        ))
        ownership = CompositeOwnership(
            plan=f"hybrid:{plan.name}", executed=tuple(sorted(owned)), excluded_edges=internal,
        )
        objectives = (
            tuple(HybridObjectiveOwner(r.name, "region", r.factor)
                  for r in sorted(declaration.regions, key=lambda r: r.name))
            + tuple(HybridObjectiveOwner(s.name, "separator") for s in separators)
            + tuple(HybridObjectiveOwner(a.name, "adapter") for a in adapters)
        )
        claims.append(HybridComponentClaim(plan, ownership, objectives))
    return (() if problems else tuple(claims)), problems


def _factor_requirement(claims, factor, sources):
    # Escape the factor segment so names cannot masquerade as parameter suffixes.
    owner = f"hybrid:factor:{quote(factor.name, safe='')}"
    if factor.type in ("categorical", "gaussian"):
        kernel = claims.registry.bindings[factor.type]
        paths = parameter_paths(kernel, owner)
        members = tuple(replace(member, writable=False, statistic_source=False)
                        for member in registered_members(
                            claims.topology, owner, (factor.name,), kernel, paths,
                        ))
        keys, recipe = tuple(paths.values()), ArrayRecipe.REGISTERED
    else:
        if factor.type == "recurrence_link":
            roles = {"weights": ("weights", "identity"), "offsets": ("offsets", "identity")}
            recipe = ArrayRecipe.RECURRENCE
        elif factor.type == "gaussian_mixture":
            roles = {"mean": ("m", "identity"), "cov": ("W", "covariance_to_precision")}
            recipe = ArrayRecipe.MIXTURE
        else:
            roles = {"matrix": ("A", "identity"), "offset": ("b", "identity"),
                     "cov": ("W", "covariance_to_precision")}
            recipe = ArrayRecipe.MIXTURE
        runtime = {role: ParameterRuntime(f"{owner}:{key}", transform)
                   for role, (key, transform) in roles.items()}
        members = member_requirements((factor.name,), runtime, writable=False, statistic_source=False)
        keys = tuple(parameter.key for parameter in runtime.values())
    return ArrayRequirement(
        owner, recipe,
        tuple(ArrayInput(source.factor, source.role, source.blob,
                         claims.initializer_shapes.get((source.factor, source.role)), source.shape)
              for source in sources),
        tuple(ArrayDestination("parameter", key) for key in keys), members,
    )


def lower_hybrid_components(claims):
    """Keep original source traversal and existing parameter-role conventions.

    Member indexes/keys are unresolved intentions, not Arena addresses. Disabled
    writes/statistics do not certify fixed-parameter bundles or create learning
    inventories. Value resolution and publication remain unavailable.
    """
    if not claims.hybrid_components:
        return ()
    sources = {}
    for source in source_requirements(claims.topology, claims.registry):
        sources.setdefault(source.factor, []).append(source)
    drafts = []
    for component in claims.hybrid_components:
        members = set(component.plan.parameter_members)
        arrays = tuple(
            _factor_requirement(claims, factor, sources.get(factor.name, ()))
            for factor in claims.topology.factors.values() if factor.name in members
        )
        drafts.append(HybridComponentDraft(
            component.plan, component.ownership, component.objectives, arrays,
        ))
    return tuple(drafts)

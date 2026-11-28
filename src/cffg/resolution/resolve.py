"""Ordered compilation checkpoints over declared plans and array requirements."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from cffg.compiled import EfGroupPlan
from cffg.compiler_errors import CompileError
from cffg.resolution.gaussian_taps import resolve_taps
from cffg.core.ir import Template
from cffg.registered.contracts import EqualityReference, RegisteredPlanDrafts
from cffg.registered.packing import pack_discrete, pack_ef, pack_gaussian
from cffg.specialized.constraints import ConstraintValidationError, resolve_constraint
from cffg.specialized.contracts import SpecializedPlanDrafts
from cffg.specialized.packing import (
    pack_dirichlet,
    pack_gfe,
    pack_hybrid_gfe,
    pack_mixture,
    pack_precision,
    pack_recurrence,
    pack_transition_gfe,
    parameter_priors,
)
from cffg.specialized.structural import semiring_problem
from cffg.topology import TopologyLayout, prefix_offsets

from . import sources
from .addresses import AddressBuilder, resolve_learnables
from .contracts import ResolvedCompileData
from .observations import ObservationBuilder


def _problems(template, problems):
    if problems:
        raise CompileError(
            f"cannot compile {template.name!r}:\n  - " + "\n  - ".join(problems)
        )


def resolve_compile_data(
    template: Template,
    arrays: dict[str, np.ndarray] | None,
    root: str | Path | None,
    topology: TopologyLayout,
    registered: RegisteredPlanDrafts,
    specialized: SpecializedPlanDrafts,
    *,
    reuse: ResolvedCompileData | None = None,
) -> ResolvedCompileData:
    """Resolve once; publish no numeric output before the last validation gate."""
    resolved = sources.resolve_arrays(
        template,
        arrays,
        root,
        sources=registered.sources,
        fallbacks=specialized.initializer_fallbacks,
    )
    addresses = AddressBuilder() if reuse is None or template.learnables else None
    observations = ObservationBuilder() if reuse is None else None
    layout = {"validated": reuse.log_params} if reuse is not None else {}

    def add(requirement):
        if addresses is not None:
            addresses.add(template, requirement)

    parameters = {}
    concentrations, goal_priors, problems = parameter_priors(
        template, specialized, resolved
    )
    _problems(template, [*registered.conditioning_problems, *problems])
    for draft in specialized.dirichlet:
        parameters.update(pack_dirichlet(draft, concentrations))

    problems = list(specialized.composite_problems)
    if specialized.ownership_problem:
        problems.append(specialized.ownership_problem)
    _problems(template, problems)
    for draft in specialized.gfe:
        parameters.update(
            pack_gfe(
                draft,
                specialized.gfe_goals[draft.plan.name],
                goal_priors,
                resolved,
            )
        )
        add(draft.arrays)
    for draft in specialized.transition_gfe:
        parameters.update(pack_transition_gfe(draft, resolved))
        add(draft.arrays)
    for draft in specialized.hybrid_gfe:
        parameters.update(pack_hybrid_gfe(template, draft, resolved))

    if registered.grouping_problem:
        raise CompileError(registered.grouping_problem)
    equality_of = {
        member: plan
        for plan in specialized.equalities.values()
        for member in plan.members
    }
    gauss_groups, group_names = [], set()
    for index, draft in enumerate(registered.gaussian):
        equality = isinstance(draft, EqualityReference)
        name = equality_of[draft.members[0]].name if equality else draft.name
        if name in group_names:
            raise CompileError(f"Gaussian group name collision: {name!r}")
        group_names.add(name)
        if equality:
            gauss_groups.append(equality_of[draft.members[0]])
            continue
        plan = observations.gaussian_plan(draft) if reuse is None else reuse.gauss_groups[index]
        gauss_groups.append(plan)
        parameters.update(pack_gaussian(template, draft, plan, resolved, **layout))
        add(draft.arrays)

    for draft in specialized.mixtures:
        parameters.update(pack_mixture(template, draft, resolved))
        add(draft.arrays)
    for draft in specialized.recurrence:
        parameters.update(pack_recurrence(template, draft, resolved))
        add(draft.arrays)
    for draft in specialized.precision:
        parameters.update(pack_precision(template, draft, resolved))
        add(draft.arrays)

    ef_groups = []
    for index, draft in enumerate(registered.ef):
        if isinstance(draft, EqualityReference):
            ef_groups.append(equality_of[draft.members[0]])
            continue
        rows = observations.ef_rows(draft) if reuse is None else None
        parameters.update(pack_ef(template, draft, resolved))
        if reuse is not None:
            ef_groups.append(reuse.ef_groups[index])
            continue
        ef_groups.append(
            EfGroupPlan(
                draft.name,
                draft.type,
                draft.family,
                draft.roles,
                draft.members,
                draft.gather,
                draft.scatter,
                rows,
            )
        )

    groups = []
    for draft in registered.discrete:
        if isinstance(draft, EqualityReference):
            groups.append(equality_of[draft.members[0]])
            continue
        parameters.update(
            pack_discrete(template, draft, resolved, specialized.semiring, **layout)
        )
        groups.append(draft.plan)
        add(draft.arrays)

    constraint_groups = []
    for draft in specialized.constraints:
        try:
            constraint = resolve_constraint(draft, resolved)
        except ConstraintValidationError as error:
            raise CompileError(str(error)) from error
        collision = parameters.keys() & constraint.parameters.keys()
        if collision:
            raise CompileError(
                f"constraint parameter address collision: {sorted(collision)}"
            )
        constraint_groups.append(constraint.plan)
        parameters.update(constraint.parameters)
    rejects = semiring_problem(
        specialized,
        any(group.mode == "vmp" for group in groups),
        bool(gauss_groups),
        bool(ef_groups),
    )
    if rejects:
        raise CompileError(
            "semiring='max' (MAP) is defined in v1 for discrete BP groups and hard "
            f"constraints only; template {template.name!r} has: " + rejects
        )
    if specialized.delta_problem:
        raise CompileError(specialized.delta_problem)
    if reuse is not None:
        # All earlier family checks still ran on live numeric sources. The prior
        # assembly certified the shared structural ownership and addresses.
        learn_plans = resolve_learnables(template, resolved, addresses) if addresses is not None else ()
        return replace(reuse, log_params=parameters, learn_plans=learn_plans)

    gauss_groups, tap_edges = resolve_taps(topology, gauss_groups, observations)
    obs_lengths = observations.gaussian_lengths()
    if specialized.goal_parameter_problem:
        raise ValueError(specialized.goal_parameter_problem)

    # Prefix conversion precedes learning; numeric publication follows learning.
    message_starts = prefix_offsets(topology.message_lengths)
    g_eta_starts = prefix_offsets(topology.g_eta_lengths)
    g_lam_starts = prefix_offsets(topology.g_lam_lengths)
    g_obs_starts = prefix_offsets(obs_lengths)
    learn_plans = resolve_learnables(template, resolved, addresses)
    return ResolvedCompileData(
        template_name=template.name,
        log_params=parameters,
        groups=groups,
        gauss_groups=gauss_groups,
        ef_groups=ef_groups,
        constraint_groups=constraint_groups,
        gauss_data=observations.gauss_data,
        gaussian_taps=tap_edges,
        n_obs_rows=observations.n_obs_rows,
        ef_data=observations.ef_data,
        n_ef_obs=observations.n_ef_obs,
        message_starts=message_starts,
        g_eta_starts=g_eta_starts,
        g_lam_starts=g_lam_starts,
        g_obs_starts=g_obs_starts,
        member_of=addresses.member_of,
        writable_members=addresses.writable_members,
        learn_plans=learn_plans,
        mu3=specialized.mu3,
        goal_parameters=specialized.goal_parameters,
    )

"""Two-stage specialized lowering, independent of numeric array resolution."""

from cffg.compiler_errors import CompileError
from cffg.goal_parameters import build_goal_parameter_plans

from .composites import (
    lower_dirichlet,
    lower_gfe,
    lower_hybrid_gfe,
    lower_transition_gfe,
)
from .constraints import lower_constraints
from .contracts import (
    CompositeOwnership,
    OrdinaryMembership,
    SpecializedClaims,
    SpecializedPlanDrafts,
)
from .discovery import discover_specialized, initializer_fallbacks
from .mixed import lower_mixtures, lower_precision, lower_recurrence
from .structural import lower_delta, lower_equalities

__all__ = [
    "OrdinaryMembership",
    "complete_specialized",
    "discover_specialized",
    "validate_admission",
]


def validate_admission(template_name: str, claims: SpecializedClaims) -> None:
    """Publish admission and parameter-layout diagnostics before later stages."""
    for errors in (claims.admission_problems, claims.layout_problems):
        if errors:
            raise CompileError(
                f"cannot compile {template_name!r}:\n  - " + "\n  - ".join(errors)
            )


def _ownership(
    claims: SpecializedClaims,
    *,
    dirichlet,
    gfe,
    goals,
    transition,
    hybrid,
    mixed,
    recurrence,
    precision,
    constraints,
    equalities,
    mu3,
):
    topology = claims.topology
    parameter_of = {
        claim.layout.likelihood: claim.variable for claim in claims.parameters
    }
    parameter_refs = {}
    for overlay in mu3:
        parameter_refs.setdefault(overlay.gfe_name, []).append(overlay.name)
    records = []
    beads_by_factor = {}
    for index, constraint in enumerate(topology.constraints):
        beads_by_factor.setdefault(constraint.target, []).append(
            (index, (constraint.kind.value, constraint.target))
        )

    def add(plan, members, absorbed=(), excluded=(), owners=(), parameter_edges=()):
        factors = set(members) | set(absorbed)
        beads = tuple(
            bead
            for _, bead in sorted(
                item for factor in factors for item in beads_by_factor.get(factor, ())
            )
        )
        records.append(
            CompositeOwnership(
                plan=plan,
                executed=tuple(members),
                absorbed=tuple(absorbed),
                constraints=beads,
                excluded_edges=tuple(excluded),
                parameter_owners=tuple(owners),
                parameter_edges=tuple(parameter_edges),
            )
        )

    for draft in dirichlet:
        plan = draft.plan
        add(
            plan.name,
            plan.members,
            plan.priors,
            owners=(plan.name,),
            parameter_edges=tuple(parameter_of[m] for m in plan.members),
        )
    for draft in gfe:
        plan = draft.plan
        owners = tuple(parameter_refs.get(plan.name, ())) + tuple(
            f"goal:{blob}" for blob in sorted(set(plan.goal_prior_blobs))
        )
        add(
            plan.name,
            plan.members,
            goals[plan.name] + plan.goal_priors,
            tuple(topology.factors[m].edges["out"] for m in plan.members),
            owners,
            plan.goal_param_edges,
        )
    for draft in hybrid:
        plan = draft.plan
        add(
            plan.name,
            plan.members,
            plan.goal_members,
            tuple(topology.factors[m].edges["out"] for m in plan.members),
        )
    for draft in (*transition, *mixed, *recurrence, *precision):
        add(draft.plan.name, draft.plan.members)
    for draft in constraints:
        add(draft.name, (draft.name,))
    for plan in equalities.values():
        add(plan.name, plan.members)
    seen = {}
    problem = None
    for record in records:
        for factor in (*record.executed, *record.absorbed):
            if factor in seen and problem is None:
                problem = f"factor {factor!r}: conflicting specialized owners {seen[factor]!r} and {record.plan!r}"
            seen[factor] = record.plan
    return tuple(records), problem


def complete_specialized(
    claims: SpecializedClaims, ordinary: OrdinaryMembership
) -> SpecializedPlanDrafts:
    """Complete structural plans after ordinary BP membership is available.

    No arrays, files, initializers, or numeric packers enter this operation.
    Shared-parameter errors remain deferred to their existing assembly gate.
    """
    dirichlet = lower_dirichlet(claims)
    gfe, goals, mu3 = lower_gfe(claims)
    transition = lower_transition_gfe(claims)
    hybrid = lower_hybrid_gfe(claims)
    mixed = lower_mixtures(claims)
    recurrence, components = lower_recurrence(claims, ordinary)
    precision = lower_precision(claims)
    constraints = lower_constraints(claims)
    equalities = lower_equalities(claims)
    delta_pairs, delta_problem = lower_delta(claims)
    goal_problem = None
    try:
        goal_parameters = build_goal_parameter_plans(tuple(draft.plan for draft in gfe))
    except ValueError as error:
        goal_parameters, goal_problem = (), str(error)
    ownership, ownership_problem = _ownership(
        claims,
        dirichlet=dirichlet,
        gfe=gfe,
        goals=goals,
        transition=transition,
        hybrid=hybrid,
        mixed=mixed,
        recurrence=recurrence,
        precision=precision,
        constraints=constraints,
        equalities=equalities,
        mu3=mu3,
    )
    return SpecializedPlanDrafts(
        semiring=claims.semiring,
        parameters=claims.parameters,
        parameter_layouts=dict(sorted(claims.parameter_layouts.items())),
        prior_names=tuple(
            sorted(
                f.name
                for f in claims.topology.factors.values()
                if f.type == "dirichlet"
            )
        ),
        composite_problems=claims.composite_problems,
        excluded_edges=claims.excluded_edges,
        initializer_fallbacks=initializer_fallbacks(
            claims.topology, claims.parameter_layouts
        ),
        dirichlet=dirichlet,
        gfe=gfe,
        gfe_goals=goals,
        transition_gfe=transition,
        hybrid_gfe=hybrid,
        mixtures=mixed,
        recurrence=recurrence,
        recurrence_components=components,
        precision=precision,
        constraints=constraints,
        equalities=equalities,
        mu3=mu3,
        goal_parameters=goal_parameters,
        goal_parameter_problem=goal_problem,
        ownership=ownership,
        ownership_problem=ownership_problem,
        delta_pairs=delta_pairs,
        delta_problem=delta_problem,
    )

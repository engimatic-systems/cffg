"""Structural Dirichlet, P-substitution and functional-evaluator lowering."""

import numpy as np

from cffg.compiled import DirichletGroupPlan, GfePlan, GfeTmPlan, HybridGfePlan, Mu3Plan
from cffg.core.ir import FACTOR_TYPES

from .contracts import ArrayRecipe, PlanDraft, SpecializedClaims
from .requirements import array_requirement


def lower_dirichlet(
    claims: SpecializedClaims,
) -> tuple[PlanDraft[DirichletGroupPlan], ...]:
    t = claims.topology
    slot_of, opposite = t.slot_of, t.opposite
    dir_buckets = {}
    for claim in claims.parameters:
        if claim.is_goal or claim.prior is None:
            continue
        f = t.factors[claim.layout.likelihood]
        roles = FACTOR_TYPES[f.type].edge_roles
        cards = tuple(t.variables[f.edges[role]].cardinality for role in roles)
        key = (f.type, cards, claim.layout.shape, claim.layout.axis)
        dir_buckets.setdefault(key, []).append((f.name, claim.prior, claim.variable))
    requirements = []
    dirichlet_groups: list[DirichletGroupPlan] = []
    for (ftype, cards, shape, axis), entries in sorted(dir_buckets.items()):
        roles = FACTOR_TYPES[ftype].edge_roles
        gname = f"dirvmp:{ftype}:{'x'.join(map(str, cards))}"
        members = tuple(e[0] for e in entries)
        priors = tuple(e[1] for e in entries)
        gather_both = np.array(
            [
                [[slot_of[(likelihood, r)], opposite[(likelihood, r)]] for r in roles]
                for likelihood in members
            ],
            dtype=np.int64,
        )
        scatter = gather_both[:, :, 0]
        prior_scatter = np.array(
            [slot_of[(prior, "out")] for prior in priors], dtype=np.int64
        )
        param_roles = {
            "categorical": "statistic",
            "matrix_cat": "matrix",
            "transition_mixture": "matrices",
        }
        count_scatter = np.array(
            [slot_of[(name, param_roles[ftype])] for name in members],
            dtype=np.int64,
        )
        dirichlet_groups.append(
            DirichletGroupPlan(
                gname,
                ftype,
                roles,
                cards,
                members,
                priors,
                gather_both,
                scatter,
                prior_scatter,
                count_scatter,
                shape,
                axis,
            )
        )
        requirements.append(
            array_requirement(
                claims,
                gname,
                ArrayRecipe.DIRICHLET,
                priors,
                (f"{gname}:alpha0",),
            )
        )

    return tuple(
        PlanDraft(plan, requirement)
        for plan, requirement in zip(dirichlet_groups, requirements, strict=True)
    )


def lower_gfe(
    claims: SpecializedClaims,
) -> tuple[
    tuple[PlanDraft[GfePlan], ...], dict[str, tuple[str, ...]], tuple[Mu3Plan, ...]
]:
    t = claims.topology
    slot_of, opposite = t.slot_of, t.opposite
    gfe_buckets = {
        (*claim.cards, claim.learned, claim.mu3, claim.goal_learned): claim.pairs
        for claim in claims.gfe
    }
    gfe_goal_dirichlet = {
        claim.layout.likelihood: (claim.prior, claim.variable)
        for claim in claims.parameters
        if claim.is_goal and claim.prior is not None
    }
    requirements, goals = [], {}
    gfe_groups: list[GfePlan] = []
    # Params keys owned by a GFE composite: write targets for a learnable, but
    # never stat sources (the composite owns its own update path).
    mu3_groups: list[Mu3Plan] = []
    for (ko, ki, learned, mu3, goal_learned), pairs in sorted(gfe_buckets.items()):
        gname = (
            f"gfe{'L' if learned else ''}{'M' if mu3 else ''}"
            f"{'C' if goal_learned else ''}:{ko}x{ki}"
        )
        obs_names = tuple(o for o, _ in pairs)
        param_blobs = tuple(t.factors[o].param_blobs["matrix"] for o, _ in pairs)
        goal_param_edges = (
            tuple(t.factors[g].param_edges["statistic"] for _, g in pairs)
            if goal_learned
            else ()
        )
        goal_priors = (
            tuple(gfe_goal_dirichlet[g][0] for _, g in pairs) if goal_learned else ()
        )
        goal_prior_blobs = (
            tuple(t.factors[prior].param_blobs["counts"] for prior in goal_priors)
            if goal_learned
            else ()
        )
        goal_tie_groups = (
            tuple(
                tuple(i for i, value in enumerate(goal_prior_blobs) if value == blob)
                for blob in sorted(set(goal_prior_blobs))
            )
            if goal_learned
            else ()
        )
        gfe_groups.append(
            GfePlan(
                name=gname,
                cards=(ko, ki),
                members=obs_names,
                gather_z=np.array(
                    [opposite[(o, "in")] for o, _ in pairs], dtype=np.int64
                ),
                scatter_z=np.array(
                    [slot_of[(o, "in")] for o, _ in pairs], dtype=np.int64
                ),
                scatter_x=np.array(
                    [slot_of[(o, "out")] for o, _ in pairs], dtype=np.int64
                ),
                learned=learned,
                mu3=mu3,
                param_blobs=param_blobs,
                goal_learned=goal_learned,
                goal_members=(tuple(goal for _, goal in pairs) if goal_learned else ()),
                goal_param_edges=goal_param_edges,
                goal_priors=goal_priors,
                goal_prior_blobs=goal_prior_blobs,
                goal_tie_groups=goal_tie_groups,
                goal_prior_scatter=(
                    np.array(
                        [slot_of[(prior, "out")] for prior in goal_priors],
                        dtype=np.int64,
                    )
                    if goal_learned
                    else np.zeros(0, dtype=np.int64)
                ),
                goal_count_scatter=(
                    np.array(
                        [slot_of[(goal, "statistic")] for _, goal in pairs],
                        dtype=np.int64,
                    )
                    if goal_learned
                    else np.zeros(0, dtype=np.int64)
                ),
            )
        )
        a_key = f"{gname}:counts" if learned else f"{gname}:A"
        c_key = f"{gname}:c_alpha0" if goal_learned else f"{gname}:c"
        goals[gname] = tuple(goal for _, goal in pairs)
        sources = tuple(name for pair in pairs for name in pair) + goal_priors
        requirements.append(
            array_requirement(
                claims,
                gname,
                ArrayRecipe.GFE,
                sources,
                (a_key, c_key),
            )
        )
        if mu3:
            for blob in sorted(set(param_blobs)):
                rows = np.array(
                    [i for i, value in enumerate(param_blobs) if value == blob],
                    dtype=np.int64,
                )
                members = tuple(obs_names[int(i)] for i in rows)
                mu3_groups.append(
                    Mu3Plan(
                        name=f"mu3:{gname}:{blob}",
                        evaluator="cffg.mu3_is.v1",
                        gfe_name=gname,
                        rows=rows,
                        members=members,
                        param_blob=blob,
                        count_key=a_key,
                        cards=(ko, ki),
                    )
                )

    return (
        tuple(
            PlanDraft(plan, requirement)
            for plan, requirement in zip(gfe_groups, requirements, strict=True)
        ),
        goals,
        tuple(mu3_groups),
    )


def lower_transition_gfe(claims: SpecializedClaims) -> tuple[PlanDraft[GfeTmPlan], ...]:
    t = claims.topology
    slot_of, opposite = t.slot_of, t.opposite
    tm_buckets = {
        (claim.cards, claim.learned): claim.members for claim in claims.transition_gfe
    }
    requirements = []
    gfe_tm_groups: list[GfeTmPlan] = []
    for ((ko, ki, ku), learned), members in sorted(tm_buckets.items()):
        gname = f"gfeTM{'L' if learned else ''}:{ko}x{ki}x{ku}"
        gather = np.array(
            [[opposite[(m, r)] for r in ("out", "in", "selector")] for m in members],
            dtype=np.int64,
        )
        scatter = np.array(
            [[slot_of[(m, r)] for r in ("out", "in", "selector")] for m in members],
            dtype=np.int64,
        )
        plan = GfeTmPlan(gname, (ko, ki, ku), tuple(members), gather, scatter, learned)
        key = f"{gname}:counts" if learned else f"{gname}:B"
        requirements.append(
            array_requirement(claims, gname, ArrayRecipe.GFE_TM, members, (key,))
        )
        gfe_tm_groups.append(plan)

    return tuple(
        PlanDraft(plan, requirement)
        for plan, requirement in zip(gfe_tm_groups, requirements, strict=True)
    )


def lower_hybrid_gfe(claims: SpecializedClaims) -> tuple[PlanDraft[HybridGfePlan], ...]:
    t = claims.topology
    slot_of, opposite = t.slot_of, t.opposite
    hybrid_gfe_buckets = {
        (claim.type, claim.dims, claim.selector_cardinality): tuple(
            zip(claim.members, claim.goals, strict=True)
        )
        for claim in claims.hybrid_gfe
    }
    requirements = []
    hybrid_gfe_groups: list[HybridGfePlan] = []
    for (mtype, dims, k_sel), pairs in sorted(hybrid_gfe_buckets.items()):
        members = [member for member, _ in pairs]
        goals = [goal for _, goal in pairs]
        gauss_roles = tuple(
            role for role in FACTOR_TYPES[mtype].edge_roles if role != "selector"
        )
        gname = f"gfeH:{mtype}:{'x'.join(map(str, dims))}:{k_sel}"
        g_gather = np.array(
            [[opposite[(member, role)] for role in gauss_roles] for member in members],
            dtype=np.int64,
        )
        g_scatter = np.array(
            [[slot_of[(member, role)] for role in gauss_roles] for member in members],
            dtype=np.int64,
        )
        s_gather = np.array(
            [opposite[(member, "selector")] for member in members],
            dtype=np.int64,
        )
        s_scatter = np.array(
            [slot_of[(member, "selector")] for member in members],
            dtype=np.int64,
        )
        hybrid_gfe_groups.append(
            HybridGfePlan(
                gname,
                mtype,
                dims,
                k_sel,
                tuple(members),
                tuple(goals),
                g_gather,
                g_scatter,
                s_gather,
                s_scatter,
            )
        )
        keys = ("m", "W") if mtype == "gaussian_mixture" else ("A", "b", "W")
        requirements.append(
            array_requirement(
                claims,
                gname,
                ArrayRecipe.HYBRID_GFE,
                tuple(members) + tuple(goals),
                tuple(f"{gname}:{key}" for key in (*keys, "goal_m", "goal_W")),
            )
        )

    return tuple(
        PlanDraft(plan, requirement)
        for plan, requirement in zip(hybrid_gfe_groups, requirements, strict=True)
    )

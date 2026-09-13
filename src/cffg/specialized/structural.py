"""Domain-specific structural equality and edge-delta meaning."""

import numpy as np

from cffg.compiled import EfGroupPlan, GaussGroupPlan, GroupPlan
from cffg.core.ir import ConstraintKind

from .contracts import SpecializedClaims, SpecializedPlanDrafts


def lower_equalities(
    claims: SpecializedClaims,
) -> dict[str, GroupPlan | GaussGroupPlan | EfGroupPlan]:
    topology = claims.topology
    buckets = {}
    for member, route in claims.equalities.items():
        buckets.setdefault((route.arena, route.key), []).append(member)
    plans = {}
    roles = ("a", "b", "c")
    for (arena, key), members in sorted(buckets.items()):
        gather = np.array(
            [[topology.opposite[m, r] for r in roles] for m in members], dtype=np.int64
        )
        scatter = np.array(
            [[topology.slot_of[m, r] for r in roles] for m in members], dtype=np.int64
        )
        if arena == "gaussian":
            kind, dims, _, _ = key
            name = f"gauss:{kind}:{'x'.join(map(str, dims))}"
            plans[name] = GaussGroupPlan(
                name, kind, roles, dims, tuple(members), gather, scatter
            )
        elif arena == "ef":
            kind, family = key
            name = f"ef:{kind}:{family}"
            plans[name] = EfGroupPlan(
                name, kind, family, roles, tuple(members), gather, scatter
            )
        else:
            kind, cards, mode = key
            name = f"{mode}:{kind}:{'x'.join(map(str, cards))}"
            plans[name] = GroupPlan(
                name,
                kind,
                roles,
                cards,
                tuple(members),
                gather,
                scatter,
                mode,
                np.stack([scatter, gather], axis=-1),
            )
    return plans


def lower_delta(claims: SpecializedClaims) -> tuple[np.ndarray, str | None]:
    topology = claims.topology
    names = sorted(
        c.target for c in topology.constraints if c.kind is ConstraintKind.DELTA
    )
    problem = None
    for name in names:
        edge = topology.edges[name]
        if name in claims.parameter_layouts:
            problem = (
                f"delta constraint on Dirichlet parameter edge {name!r}: parameter "
                "posteriors are continuous simplex distributions, not categorical states"
            )
        elif edge.dim:
            problem = (
                f"delta constraint on Gaussian edge {name!r}: the MAP of a Gaussian "
                "marginal is its mean — read gaussian_marginals instead (hybrid "
                "pointmass forms: cff-pksr)"
            )
        elif edge.ef:
            problem = (
                f"delta constraint on EF edge {name!r}: read the mode from "
                "ef_marginals instead (hybrid pointmass forms: cff-pksr)"
            )
        if problem:
            break
    pairs = np.array(
        [[topology.edges[n].slot0, topology.edges[n].slot1] for n in names],
        dtype=np.int64,
    )
    return pairs.reshape(-1, 2), problem


def semiring_problem(
    drafts: SpecializedPlanDrafts,
    has_mean_field: bool,
    gauss_present: bool,
    ef_present: bool,
) -> str | None:
    """Retain the final admission gate after all family array checks."""
    if drafts.semiring != "max":
        return None
    rejects = []
    if has_mean_field:
        rejects.append("mean-field (vmp) beads")
    if any(
        not claim.is_goal and claim.prior is not None for claim in drafts.parameters
    ):
        rejects.append("Dirichlet parameter-edge VMP groups")
    if drafts.gfe or drafts.transition_gfe or drafts.hybrid_gfe:
        rejects.append("P-substitution / GFE composites")
    if gauss_present or drafts.mixtures or drafts.recurrence or drafts.precision:
        rejects.append("Gaussian/hybrid groups")
    if ef_present:
        rejects.append("EF groups")
    return ", ".join(rejects) if rejects else None

"""Structural lowerers for mixed-domain, recurrence and precision factors."""

import numpy as np

from cffg.array_requirements import ArrayRecipe
from cffg.compiled import (
    MixGroupPlan,
    PrecGroupPlan,
    RecurrenceComponentPlan,
    RecurrencePlan,
)
from cffg.core.ir import FACTOR_TYPES
from cffg.recurrence_topology import classify_recurrence_components

from .contracts import OrdinaryMembership, PlanDraft, SpecializedClaims
from .requirements import array_requirement


def lower_mixtures(claims: SpecializedClaims) -> tuple[PlanDraft[MixGroupPlan], ...]:
    t = claims.topology
    slot_of, opposite = t.slot_of, t.opposite
    mix_buckets = {
        (claim.type, claim.dims, claim.selector_cardinality): claim.members
        for claim in claims.mixtures
    }
    requirements = []
    mix_groups: list[MixGroupPlan] = []
    for (mtype, dims, k_sel), members in sorted(mix_buckets.items()):
        gauss_roles = tuple(
            r for r in FACTOR_TYPES[mtype].edge_roles if r != "selector"
        )
        gname = f"mix:{mtype}:{'x'.join(map(str, dims))}:{k_sel}"
        g_gather = np.array(
            [[opposite[(m, r)] for r in gauss_roles] for m in members], dtype=np.int64
        )
        g_scatter = np.array(
            [[slot_of[(m, r)] for r in gauss_roles] for m in members], dtype=np.int64
        )
        s_gather = np.array(
            [opposite[(m, "selector")] for m in members], dtype=np.int64
        )
        s_scatter = np.array(
            [slot_of[(m, "selector")] for m in members], dtype=np.int64
        )
        mix_groups.append(
            MixGroupPlan(
                gname,
                mtype,
                dims,
                k_sel,
                tuple(members),
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
                ArrayRecipe.MIXTURE,
                members,
                tuple(f"{gname}:{key}" for key in keys),
            )
        )

    return tuple(
        PlanDraft(plan, requirement)
        for plan, requirement in zip(mix_groups, requirements, strict=True)
    )


def lower_recurrence(
    claims: SpecializedClaims, ordinary: OrdinaryMembership
) -> tuple[tuple[PlanDraft[RecurrencePlan], ...], tuple[RecurrenceComponentPlan, ...]]:
    t = claims.topology
    endpoints, slot_of, opposite = t.endpoints, t.slot_of, t.opposite
    recurrence_buckets, param_specs = claims.recurrence, claims.parameter_layouts
    requirements = []
    recurrence_members = {
        member for members in recurrence_buckets.values() for member in members
    }
    ordinary_bp = ordinary.bp_members
    recurrence_components = tuple(
        RecurrenceComponentPlan(*component)
        for component in classify_recurrence_components(
            t.variables,
            endpoints,
            ordinary_bp,
            recurrence_members,
            set(param_specs),
        )
    )
    component_of = {
        member: component_index
        for component_index, component in enumerate(recurrence_components)
        for member in component.members
    }
    recurrence_groups: list[RecurrencePlan] = []
    for (card, drive_dim), members in sorted(recurrence_buckets.items()):
        gname = f"recurrence:{card}:{drive_dim}"
        d_scatter = np.array(
            [[slot_of[(member, role)] for role in ("out", "in")] for member in members],
            dtype=np.int64,
        )
        d_gather_both = np.stack(
            [
                d_scatter,
                np.array(
                    [
                        [opposite[(member, role)] for role in ("out", "in")]
                        for member in members
                    ],
                    dtype=np.int64,
                ),
            ],
            axis=-1,
        )
        drive_scatter = np.array(
            [slot_of[(member, "drive")] for member in members],
            dtype=np.int64,
        )
        drive_gather_both = np.stack(
            [
                drive_scatter,
                np.array(
                    [opposite[(member, "drive")] for member in members],
                    dtype=np.int64,
                ),
            ],
            axis=-1,
        )
        recurrence_groups.append(
            RecurrencePlan(
                gname,
                (card, card),
                drive_dim,
                tuple(members),
                d_gather_both,
                d_scatter,
                drive_gather_both,
                drive_scatter,
                np.asarray(
                    [component_of[member] for member in members],
                    dtype=np.int64,
                ),
            )
        )
        requirements.append(
            array_requirement(
                claims,
                gname,
                ArrayRecipe.RECURRENCE,
                members,
                (f"{gname}:weights", f"{gname}:offsets"),
            )
        )

    return (
        tuple(
            PlanDraft(plan, requirement)
            for plan, requirement in zip(recurrence_groups, requirements, strict=True)
        ),
        recurrence_components,
    )


def lower_precision(claims: SpecializedClaims) -> tuple[PlanDraft[PrecGroupPlan], ...]:
    t = claims.topology
    slot_of, opposite = t.slot_of, t.opposite
    prec_members = claims.precision
    requirements = []
    prec_groups: list[PrecGroupPlan] = []
    if prec_members:
        members = sorted(prec_members)
        gname = "prec:gaussian_prec"
        x_scatter = np.array([slot_of[(m, "out")] for m in members], dtype=np.int64)
        x_gather_both = np.stack(
            [
                x_scatter,
                np.array([opposite[(m, "out")] for m in members], dtype=np.int64),
            ],
            axis=-1,
        )
        tau_scatter = np.array([slot_of[(m, "prec")] for m in members], dtype=np.int64)
        tau_gather_both = np.stack(
            [
                tau_scatter,
                np.array([opposite[(m, "prec")] for m in members], dtype=np.int64),
            ],
            axis=-1,
        )
        prec_groups.append(
            PrecGroupPlan(
                gname,
                tuple(members),
                x_gather_both,
                tau_gather_both,
                x_scatter,
                tau_scatter,
            )
        )
        requirements.append(
            array_requirement(
                claims,
                gname,
                ArrayRecipe.PRECISION,
                members,
                (f"{gname}:m",),
            )
        )

    return tuple(
        PlanDraft(plan, requirement)
        for plan, requirement in zip(prec_groups, requirements, strict=True)
    )

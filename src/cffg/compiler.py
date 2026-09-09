"""Template → immutable executable arena.

Topology records wiring; specialized and registered owners interpret factor
declarations. Resolution owns source loading, numeric validation, observations
and learning addresses. This facade retains public compiled-value exports and
compatibility helpers while final ledger-table assembly remains here.

Each Forney edge owns two directional message slots. Persistent arrays use
exact-sized segments; local factor batches use static shape signatures.
Parameters remain per-instance inputs, not baked compiled constants.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from cffg.compiled import (
    NEG,
    PARAMETER_TRANSFORMS,
    Arena,
    ConstraintPlan,
    DirichletGroupPlan,
    EdgeInfo,
    EfGroupPlan,
    GaussGroupPlan,
    GfePlan,
    GfeTmPlan,
    GoalParameterPlan,
    GroupPlan,
    HybridGfePlan,
    LearnPlan,
    LearnTarget,
    MemberAddress,
    MixGroupPlan,
    Mu3Plan,
    ParameterAddress,
    PrecGroupPlan,
    RecurrenceComponentPlan,
    RecurrencePlan,
)
from cffg.compiler_errors import CompileError
from cffg.core.ir import Template
from cffg.core.plates import expand_plates
from cffg.core.validate import validate
from cffg.registered import packing as _registered_packing
from cffg.registered.capabilities import snapshot_registry
from cffg.registered.lowering import lower_registered
from cffg.resolution import sources as _source_helpers
from cffg.resolution.resolve import resolve_compile_data
from cffg.specialized import (
    OrdinaryMembership,
    complete_specialized,
    discover_specialized,
)
from cffg.specialized import packing as _specialized_packing
from cffg.specialized.discovery import dirichlet_shape
from cffg.topology import build_topology

__all__ = [
    "NEG",
    "PARAMETER_TRANSFORMS",
    "Arena",
    "CompileError",
    "ConstraintPlan",
    "DirichletGroupPlan",
    "EdgeInfo",
    "EfGroupPlan",
    "GaussGroupPlan",
    "GfePlan",
    "GfeTmPlan",
    "GoalParameterPlan",
    "GroupPlan",
    "HybridGfePlan",
    "LearnPlan",
    "LearnTarget",
    "MemberAddress",
    "MixGroupPlan",
    "Mu3Plan",
    "ParameterAddress",
    "PrecGroupPlan",
    "RecurrenceComponentPlan",
    "RecurrencePlan",
    "compile_template",
]

_resolve_arrays = _source_helpers.resolve_arrays
_blob_shape = _source_helpers._blob_shape
_load_blob = _source_helpers._load_blob
_stack_params = _registered_packing._stack_params
_stack_gauss_params = _registered_packing._stack_gauss_params
_stack_mix_params = _specialized_packing._stack_mix_params
_stack_prec_means = _specialized_packing._stack_prec_means
_stack_recurrence_params = _specialized_packing._stack_recurrence_params
_log_of = _specialized_packing._log_of
_expect = _specialized_packing._expect
_stochastic = _specialized_packing._stochastic


def compile_template(
    t: Template,
    *,
    arrays: dict[str, np.ndarray] | None = None,
    root: str | Path | None = None,
    semiring: str = "sum",
) -> Arena:
    """Ground `t` into an Arena. Blob data comes from `arrays` (name -> array,
    wins) or is loaded from `root`/blob.path.

    `semiring` selects the inference algebra (cff-e8zk): "sum" is sum-product
    (the BFE ledger; -log Z on trees), "max" is min-sum / max-product (MAP
    decoding; Engine.objective, not Engine.bfe). MAP semantics are defined in v1
    for discrete BP groups and hard constraints only — "max" REJECTS vmp/mean-
    field beads, P-substitution/GFE composites, and Gaussian/EF/hybrid groups."""
    if semiring not in ("sum", "max"):
        raise CompileError(f"semiring must be 'sum' or 'max', got {semiring!r}")
    t = expand_plates(
        t
    )  # materialize plates (validates the spec first); no-op when plateless
    validate(t)
    topology = build_topology(t)
    registry = snapshot_registry()
    claims = discover_specialized(t, topology, semiring, registry)
    for errors in (claims.admission_problems, claims.layout_problems):
        if errors:
            raise CompileError(
                f"cannot compile {t.name!r}:\n  - " + "\n  - ".join(errors)
            )
    edges = dict(topology.edges)
    for name, layout in claims.parameter_layouts.items():
        edges[name] = replace(
            edges[name], param_shape=layout.shape, param_axis=layout.axis
        )
    registered = lower_registered(t, topology, claims, semiring, registry)
    drafts = complete_specialized(claims, OrdinaryMembership(registered.bp_members))
    data = resolve_compile_data(t, arrays, root, topology, registered, drafts)

    # Interior Gaussian edges, grouped by dim, for the Bethe entropy ledger.
    by_dim: dict[int, list[list[int]]] = {}
    for e in edges.values():
        if e.dim and e.interior and e.name not in drafts.excluded_edges:
            by_dim.setdefault(e.dim, []).append([e.slot0, e.slot1])
    gauss_interior = [
        (d, np.array(pairs, dtype=np.int64)) for d, pairs in sorted(by_dim.items())
    ]

    # Interior EF edges, grouped by family, likewise. Observation-family
    # edges are always exterior DATA ports (enforced above), so every
    # interior EF edge has a message family with a closed-form entropy.
    by_fam: dict[str, list[list[int]]] = {}
    for e in edges.values():
        if e.ef and e.interior:
            by_fam.setdefault(e.ef, []).append([e.slot0, e.slot1])
    ef_interior = [
        (fam, np.array(pairs, dtype=np.int64)) for fam, pairs in sorted(by_fam.items())
    ]

    return Arena(
        template_name=data.template_name,
        n_slots=topology.n_slots,
        message_starts=data.message_starts,
        edges=edges,
        groups=data.groups,
        data_slots=registered.data_slots,
        semiring=semiring,
        dirichlet_groups=tuple(draft.plan for draft in drafts.dirichlet),
        constraint_groups=data.constraint_groups,
        log_params=data.log_params,
        n_gslots=topology.n_gslots,
        g_eta_starts=data.g_eta_starts,
        g_lam_starts=data.g_lam_starts,
        gauss_groups=data.gauss_groups,
        gauss_data=data.gauss_data,
        n_obs_rows=data.n_obs_rows,
        g_obs_starts=data.g_obs_starts,
        mix_groups=tuple(draft.plan for draft in drafts.mixtures),
        recurrence_groups=tuple(draft.plan for draft in drafts.recurrence),
        recurrence_components=drafts.recurrence_components,
        prec_groups=tuple(draft.plan for draft in drafts.precision),
        gauss_interior=gauss_interior,
        n_efslots=topology.n_efslots,
        ef_groups=data.ef_groups,
        ef_data=data.ef_data,
        n_ef_obs=data.n_ef_obs,
        ef_interior=ef_interior,
        gfe_groups=tuple(draft.plan for draft in drafts.gfe),
        goal_parameter_plans=data.goal_parameters,
        mu3_groups=data.mu3,
        gfe_tm_groups=tuple(draft.plan for draft in drafts.transition_gfe),
        hybrid_gfe_groups=tuple(draft.plan for draft in drafts.hybrid_gfe),
        gfe_x_edges=drafts.excluded_edges,
        delta_pairs=drafts.delta_pairs,
        member_of=data.member_of,
        writable_members=data.writable_members,
        learn_plans=data.learn_plans,
    )


def _dirichlet_shape(t: Template, likelihood_name: str) -> tuple[tuple[int, ...], int]:
    """Compatibility entry for the specialized concentration layout owner."""
    try:
        return dirichlet_shape(t, likelihood_name)
    except ValueError as error:
        raise CompileError(str(error)) from error

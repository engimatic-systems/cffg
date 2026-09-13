"""Pure assembly of validated compiler stages into one immutable arena."""

from dataclasses import replace

import numpy as np

from cffg.assembly_checks import validate_assembly
from cffg.compiled import Arena
from cffg.registered.contracts import RegisteredPlanDrafts
from cffg.resolution.contracts import ResolvedCompileData
from cffg.specialized.contracts import SpecializedPlanDrafts
from cffg.topology import TopologyLayout


def assemble_arena(
    topology: TopologyLayout,
    registered: RegisteredPlanDrafts,
    drafts: SpecializedPlanDrafts,
    data: ResolvedCompileData,
    semiring: str,
) -> Arena:
    """Join stage values; reuse published parameters, prefixes and plans."""
    validate_assembly(topology, registered, drafts, data, semiring)
    edges = dict(topology.edges)
    for name, layout in drafts.parameter_layouts.items():
        edges[name] = replace(
            edges[name], param_shape=layout.shape, param_axis=layout.axis
        )

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

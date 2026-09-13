"""Cross-stage ownership checks over already interpreted compiler declarations."""

from collections import Counter

from cffg.compiler_errors import CompileError
from cffg.registered.contracts import (
    DiscreteDraft,
    EqualityReference,
    RegisteredPlanDrafts,
)
from cffg.resolution.contracts import ResolvedCompileData
from cffg.specialized.contracts import SpecializedPlanDrafts
from cffg.topology import TopologyLayout


def _require(condition, message):
    if not condition:
        raise CompileError(f"arena assembly: {message}")


def _resolved_groups(registered, drafts, data):
    equalities = {plan.members: plan for plan in drafts.equalities.values()}
    _require(
        len(equalities) == len(drafts.equalities),
        "multiple equality plans declare the same members",
    )
    ordinary, structural = [], []
    for kind, declarations, plans in (
        ("discrete", registered.discrete, data.groups),
        ("Gaussian", registered.gaussian, data.gauss_groups),
        ("EF", registered.ef, data.ef_groups),
    ):
        _require(len(declarations) == len(plans), f"{kind} group count disagrees")
        for draft, plan in zip(declarations, plans, strict=True):
            if isinstance(draft, EqualityReference):
                expected = equalities.get(draft.members)
                _require(expected is not None, f"missing equality for {draft.members}")
                structural.append(plan)
            else:
                expected = draft.plan if isinstance(draft, DiscreteDraft) else draft
                ordinary.append(plan)
            for field in ("name", "type", "roles", "members"):
                _require(
                    getattr(plan, field) == getattr(expected, field),
                    f"{kind} plan {plan.name!r}: resolved {field} disagrees",
                )
            for field in ("gather", "scatter"):
                _require(
                    getattr(plan, field) is getattr(expected, field),
                    f"{kind} plan {plan.name!r}: resolved {field} was replaced",
                )
    _require(
        {plan.members for plan in structural} == set(equalities),
        "not every declared equality has an execution plan",
    )
    _require(
        len(drafts.constraints) == len(data.constraint_groups),
        "constraint group count disagrees",
    )
    for draft, plan in zip(drafts.constraints, data.constraint_groups, strict=True):
        _require(
            (plan.name, plan.type, plan.cards, plan.members)
            == (draft.name, draft.type, draft.cards, (draft.name,)),
            f"constraint {draft.name!r}: resolved declaration disagrees",
        )
        _require(
            plan.gather is draft.gather and plan.scatter is draft.scatter,
            f"constraint {draft.name!r}: resolved addresses were replaced",
        )
    return ordinary, structural


def _execution_owners(topology, drafts, data, ordinary, equalities):
    owners = {}
    for index, owner in enumerate(drafts.ownership):
        _require(owner.executed, f"owner {owner.plan!r} has no executed members")
        for member in (*owner.executed, *owner.absorbed):
            _require(member in topology.factors, f"unknown owned factor {member!r}")
            _require(
                member not in owners, f"factor {member!r} has multiple execution owners"
            )
            owners[member] = index
    plans = [
        draft.plan
        for collection in (
            drafts.dirichlet,
            drafts.gfe,
            drafts.transition_gfe,
            drafts.hybrid_gfe,
            drafts.mixtures,
            drafts.recurrence,
            drafts.precision,
        )
        for draft in collection
    ]
    plans.extend((*equalities, *data.constraint_groups))
    used = set()
    for plan in plans:
        _require(plan.members, f"plan {plan.name!r} has no executed members")
        index = owners.get(plan.members[0])
        _require(
            index is not None, f"factor {plan.members[0]!r} has no specialized owner"
        )
        owner = drafts.ownership[index]
        _require(
            (owner.plan, owner.executed) == (plan.name, plan.members),
            f"plan {plan.name!r}: execution members disagree with its owner",
        )
        _require(
            index not in used, f"owner {owner.plan!r} has multiple execution plans"
        )
        used.add(index)
    _require(
        len(used) == len(drafts.ownership), "an ownership record has no execution plan"
    )
    all_members = set(owners)
    for plan in ordinary:
        for member in plan.members:
            _require(member in topology.factors, f"unknown executed factor {member!r}")
            _require(
                member not in all_members,
                f"factor {member!r} has multiple execution owners",
            )
            all_members.add(member)
    missing = set(topology.factors) - all_members
    _require(not missing, f"factors have no execution owner: {sorted(missing)}")
    return owners


def _objective_declarations(topology, drafts):
    excluded = Counter(
        edge for owner in drafts.ownership for edge in owner.excluded_edges
    )
    _require(
        set(excluded) == set(drafts.excluded_edges)
        and all(n == 1 for n in excluded.values()),
        "composite entropy exclusions disagree with ownership",
    )
    _require(
        set(excluded) <= set(topology.edges),
        "an entropy exclusion names an unknown edge",
    )
    parameters = Counter(
        edge for owner in drafts.ownership for edge in owner.parameter_edges
    )
    _require(
        set(parameters) == set(drafts.parameter_layouts)
        and all(n == 1 for n in parameters.values()),
        "parameter-edge tags disagree with ownership",
    )
    _require(
        set(parameters) <= set(topology.edges), "a parameter tag names an unknown edge"
    )
    constraints = {}
    for index, bead in enumerate(topology.constraints):
        constraints.setdefault(bead.target, []).append(
            (index, (bead.kind.value, bead.target))
        )
    for owner in drafts.ownership:
        expected = tuple(
            bead
            for _, bead in sorted(
                item
                for member in (*owner.executed, *owner.absorbed)
                for item in constraints.get(member, ())
            )
        )
        _require(
            owner.constraints == expected,
            f"owner {owner.plan!r}: constraint declarations disagree",
        )


def _parameter_owners(drafts, data, owners):
    for name in ("mu3", "goal_parameters"):
        declared, resolved = getattr(drafts, name), getattr(data, name)
        _require(
            len(declared) == len(resolved)
            and all(a is b for a, b in zip(declared, resolved, strict=True)),
            f"resolved {name} owners disagree with specialized plans",
        )
    gfe = {draft.plan.name: draft.plan for draft in drafts.gfe}
    _require(len(gfe) == len(drafts.gfe), "multiple GFE plans have the same name")
    users = {}

    def define(name, members):
        _require(name not in users, f"duplicate parameter owner {name!r}")
        _require(members, f"parameter owner {name!r} has no contributors")
        _require(
            all(member in owners for member in members),
            f"parameter owner {name!r} has unknown contributors",
        )
        users[name] = {owners[member] for member in members}

    for draft in drafts.dirichlet:
        define(draft.plan.name, draft.plan.members)
    for plan in data.mu3:
        group = gfe.get(plan.gfe_name)
        _require(group is not None, f"mu3 owner {plan.name!r} has an unknown GFE group")
        _require(
            plan.rows.shape == (len(plan.members),) and plan.rows.dtype.kind in "iu",
            f"mu3 owner {plan.name!r} has invalid contributor rows",
        )
        rows = tuple(plan.rows.tolist())
        _require(
            len(set(rows)) == len(rows)
            and all(0 <= row < len(group.members) for row in rows),
            f"mu3 owner {plan.name!r} has invalid contributor rows",
        )
        _require(
            tuple(group.members[row] for row in rows) == plan.members,
            f"mu3 owner {plan.name!r}: contributor rows disagree",
        )
        define(plan.name, plan.members)
    seen = set()
    for plan in data.goal_parameters:
        members, edges, prior_slots, count_slots = [], [], [], []
        for name, rows in plan.group_rows:
            group = gfe.get(name)
            _require(
                group is not None,
                f"goal owner {plan.prior_blob!r} has an unknown GFE group",
            )
            for row in rows:
                _require(
                    0 <= row < len(group.goal_prior_blobs),
                    f"goal owner {plan.prior_blob!r} has an invalid contributor row",
                )
                _require(
                    (name, row) not in seen,
                    f"goal contributor {(name, row)!r} is owned twice",
                )
                seen.add((name, row))
                _require(
                    group.goal_prior_blobs[row] == plan.prior_blob
                    and group.cards[0] == plan.cardinality,
                    f"goal owner {plan.prior_blob!r}: contributor declaration disagrees",
                )
                members.append(group.members[row])
                edges.append(group.goal_param_edges[row])
                prior_slots.append(int(group.goal_prior_scatter[row]))
                count_slots.append(int(group.goal_count_scatter[row]))
        _require(
            tuple(edges) == plan.parameter_edges
            and prior_slots == plan.prior_scatter.tolist()
            and count_slots == plan.count_scatter.tolist(),
            f"goal owner {plan.prior_blob!r}: parameter carriers disagree",
        )
        define(f"goal:{plan.prior_blob}", members)
    expected = {
        (group.name, row)
        for group in gfe.values()
        for row in range(len(group.goal_prior_blobs))
    }
    _require(seen == expected, "a learned-goal contribution has no parameter owner")
    references = {}
    for index, owner in enumerate(drafts.ownership):
        _require(
            len(set(owner.parameter_owners)) == len(owner.parameter_owners),
            f"owner {owner.plan!r} repeats a parameter-owner reference",
        )
        for name in owner.parameter_owners:
            references.setdefault(name, set()).add(index)
    mismatches = sorted(
        name
        for name in references.keys() | users.keys()
        if references.get(name) != users.get(name)
    )
    _require(not mismatches, f"parameter-owner references disagree for {mismatches}")


def validate_assembly(
    topology: TopologyLayout,
    registered: RegisteredPlanDrafts,
    drafts: SpecializedPlanDrafts,
    data: ResolvedCompileData,
    semiring: str,
) -> None:
    """Check joins and ownership, not family mathematics or numeric parameters."""
    _require(semiring == drafts.semiring, "semiring disagrees with specialized plans")
    for name, count in (
        ("message_starts", topology.n_slots),
        ("g_eta_starts", topology.n_gslots),
        ("g_lam_starts", topology.n_gslots),
        ("g_obs_starts", data.n_obs_rows),
    ):
        _require(
            getattr(data, name).shape == (count + 1,),
            f"resolved {name} length disagrees",
        )
    ordinary, equalities = _resolved_groups(registered, drafts, data)
    owners = _execution_owners(topology, drafts, data, ordinary, equalities)
    _objective_declarations(topology, drafts)
    _parameter_owners(drafts, data, owners)

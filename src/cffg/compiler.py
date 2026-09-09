"""Template → executable arena.

Grounds a validated template into shape-segmented message arenas
(PGMax-inspired). Each FFG edge owns two logical slots, one per side, holding
the message flowing *into* the edge from that side. Persistent values are
packed exactly into flat arrays; prefix-offset tables map logical slots to
their cardinality/dimension-sized segments. Factors are grouped by (type,
shape signature), so rule calls still use dense static local shapes without a
model-wide maximum padding every unrelated edge.

Parameters are *inputs* to the compiled step, not baked constants: instances
sharing a structure hash can be vmapped with different parameters (the colony
contract, docs/ARCHITECTURE.md §5b).

Port policy at compile time:
- DATA ports: allowed; the edge's outer slot is a runtime clamp (uniform when
  unbound = missing observation).
- GLUE ports: allowed; an unbonded glue edge is an FFG half-edge, which is
  implicit marginalization (Loeliger 2004) — the outer slot stays uniform.
- PRIOR / GOAL ports: rejected; they declare that a factor is *missing*, so
  compiling would silently change the model. Compose the prior (M2) or bind
  the goal machinery (M3) first.

Module map:
- ``compiled`` owns immutable plan dataclasses and ``Arena``; this facade
  retains their compatibility exports.
- ``topology`` records declared wiring; ``specialized`` owns named semantic
  claims and drafts. ``registered`` lowers remaining same-arena structure and
  supplies ordinary BP membership for specialized completion.
- ``compile_template`` preserves validation order, packs declared arrays, and
  assembles the arena from these stage products.
- ``_resolve_learnables`` binds declarations to statistic sources and writes.
- ``_resolve_arrays``, ``_blob_shape``, and ``_stack_*`` load and validate
  parameter data.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from cffg import kernels
from cffg.array_requirements import ArrayRequirement
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
from cffg.core.ir import Template
from cffg.core.plates import expand_plates
from cffg.core.validate import validate
from cffg.registered.capabilities import snapshot_registry
from cffg.registered.contracts import EqualityReference
from cffg.registered.lowering import lower_registered
from cffg.specialized import (
    OrdinaryMembership,
    complete_specialized,
    discover_specialized,
)
from cffg.specialized.constraints import ConstraintValidationError, resolve_constraint
from cffg.specialized.discovery import dirichlet_shape, specialized_initializer_shape
from cffg.specialized.structural import semiring_problem
from cffg.topology import build_topology, prefix_offsets

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

class CompileError(ValueError):
    pass


def _requirement_arrays(requirement: ArrayRequirement, resolved):
    """Restrict a compatibility packer to its declared source arrays."""
    return {source.blob: resolved[source.blob] for source in requirement.inputs}


def _publish_required_parameters(parameters, requirement: ArrayRequirement, packed):
    values = {f"{requirement.owner}:{key}": value for key, value in packed.items()}
    expected = {destination.name for destination in requirement.destinations}
    if set(values) != expected:
        raise CompileError(
            f"plan {requirement.owner!r}: packed keys {sorted(values)} "
            f"do not match declared destinations {sorted(expected)}"
        )
    parameters.update(values)




def _set_member_address(
    member_of: dict[str, MemberAddress],
    template: Template,
    member: str,
    index: int,
    runtime: dict[str, tuple[str, str]],
    declarations=None,
) -> None:
    """Register the runtime row for every blob-backed role of ``member``."""
    factor = template.factors[member]
    if not factor.param_blobs:
        return
    missing = sorted(set(factor.param_blobs) - set(runtime))
    if missing:
        raise CompileError(
            f"factor {member!r}: no runtime parameter address for roles {missing}"
        )
    parameters = tuple(
        ParameterAddress(
            role=role,
            blob=blob,
            key=runtime[role][0],
            index=index,
            transform=runtime[role][1],
            probability_axis=declarations[role].probability_axis if declarations else None,
            permutation=declarations[role].permutation if declarations else (),
        )
        for role, blob in sorted(factor.param_blobs.items())
    )
    member_of[member] = MemberAddress(factor.type, parameters)


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
    t = expand_plates(t)  # materialize plates (validates the spec first); no-op when plateless
    validate(t)
    problems: list[str] = []
    topology = build_topology(t)

    registry = snapshot_registry()
    claims = discover_specialized(t, topology, semiring, registry)
    for errors in (claims.admission_problems, claims.layout_problems):
        if errors:
            raise CompileError(f"cannot compile {t.name!r}:\n  - " + "\n  - ".join(errors))
    param_specs = {
        variable: (layout.shape, layout.axis, layout.likelihood)
        for variable, layout in claims.parameter_layouts.items()
    }

    edges = dict(topology.edges)
    for name, (shape, axis, _) in param_specs.items():
        edges[name] = replace(edges[name], param_shape=shape, param_axis=axis)
    registered = lower_registered(t, topology, claims, semiring, registry)
    problems.extend(registered.conditioning_problems)
    data_slots = registered.data_slots
    drafts = complete_specialized(claims, OrdinaryMembership(registered.bp_members))
    equality_of = {
        member: plan for plan in drafts.equalities.values() for member in plan.members
    }
    resolved = _resolve_arrays(t, arrays, root)
    member_of: dict[str, MemberAddress] = {}
    writable_members: set[str] = set()
    log_params: dict[str, np.ndarray] = {}
    consumed: set[str] = set()

    # Diagnostic success is separate from semantic ownership: invalid prior
    # arrays still leave an orphan-prior diagnostic at this compatibility gate.
    concentrations = {}
    gfe_goal_dirichlet = {}
    for claim in claims.parameters:
        shape, prior_name = claim.layout.shape, claim.prior
        if claim.problem:
            problems.append(claim.problem)
            continue
        prior = t.factors[prior_name]
        alpha0 = resolved[prior.param_blobs["counts"]]
        if alpha0.shape != shape:
            problems.append(
                f"Dirichlet prior {prior_name!r}: concentration shape {alpha0.shape} "
                f"!= likelihood table shape {shape}"
            )
            continue
        if not np.all(np.isfinite(alpha0)) or not np.all(alpha0 > 0):
            problems.append(
                f"Dirichlet prior {prior_name!r}: concentrations must be finite and > 0"
            )
            continue
        likelihood_name = claim.layout.likelihood
        concentrations[likelihood_name] = alpha0
        consumed.update((likelihood_name, prior_name))
        if claim.is_goal:
            gfe_goal_dirichlet[likelihood_name] = (prior_name, claim.variable, alpha0)

    orphan_priors = sorted(
        f.name
        for f in t.factors.values()
        if f.type == "dirichlet" and f.name not in consumed
    )
    if orphan_priors:
        problems.append(
            "dirichlet factors are parameter priors and must face a supported "
            f"mean-field likelihood; unpaired: {orphan_priors}"
        )
    if problems:
        raise CompileError(f"cannot compile {t.name!r}:\n  - " + "\n  - ".join(problems))

    dirichlet_groups = tuple(draft.plan for draft in drafts.dirichlet)
    for draft in drafts.dirichlet:
        log_params[draft.arrays.destinations[0].name] = np.stack(
            [concentrations[member] for member in draft.plan.members]
        )

    problems.extend(claims.composite_problems)
    if drafts.ownership_problem:
        problems.append(drafts.ownership_problem)
    if problems:
        raise CompileError(f"cannot compile {t.name!r}:\n  - " + "\n  - ".join(problems))
    gfe_x_edges = claims.excluded_edges
    gfe_groups = tuple(draft.plan for draft in drafts.gfe)
    composite_keys: set[str] = set()
    mu3_groups = drafts.mu3
    for draft in drafts.gfe:
        group = draft.plan
        gname, (ko, ki) = group.name, group.cards
        learned, goal_learned = group.learned, group.goal_learned
        pairs = tuple(zip(group.members, drafts.gfe_goals[gname], strict=True))
        sources = {
            (source.factor, source.role): resolved[source.blob]
            for source in draft.arrays.inputs
        }
        la, lc = [], []
        for obs_name, goal_name in pairs:
            a = sources[obs_name, "matrix"]
            _expect(obs_name, a, (ko, ki))
            if learned:
                if not np.all(a > 0):
                    raise CompileError(f"factor {obs_name!r}: Dirichlet counts must be > 0")
                la.append(a)  # raw counts α, not probabilities
            else:
                _stochastic(obs_name, a)
                la.append(_log_of(a))
            if goal_learned:
                c_alpha0 = gfe_goal_dirichlet[goal_name][2]
                _expect(goal_name, c_alpha0, (ko,))
                lc.append(c_alpha0)
            else:
                c = sources[goal_name, "statistic"]
                _expect(goal_name, c, (ko,))
                _stochastic(goal_name, c[:, None])
                lc.append(_log_of(c))
        a_key, c_key = (destination.name for destination in draft.arrays.destinations)
        log_params[a_key] = np.stack(la)
        log_params[c_key] = np.stack(lc)
        composite_keys.update((a_key, c_key))
        for i, (obs_name, goal_name) in enumerate(pairs):
            _set_member_address(
                member_of,
                t,
                obs_name,
                i,
                {
                    "matrix": (
                        a_key,
                        "identity" if learned else "probability_to_log",
                    )
                },
            )
            _set_member_address(
                member_of,
                t,
                goal_name,
                i,
                {
                    "statistic": (
                        c_key,
                        "identity" if goal_learned else "probability_to_log",
                    )
                },
            )
    gfe_tm_groups = tuple(draft.plan for draft in drafts.transition_gfe)
    for draft in drafts.transition_gfe:
        group = draft.plan
        gname, (ko, ki, ku), members, learned = group.name, group.cards, group.members, group.learned
        sources = {source.factor: resolved[source.blob] for source in draft.arrays.inputs}
        stacked = []
        for m in members:
            arr = sources[m]
            _expect(m, arr, (ku, ko, ki))
            if learned:
                if not np.all(arr > 0):
                    raise CompileError(f"factor {m!r}: Dirichlet counts must be > 0")
                stacked.append(arr)  # raw counts α [U, K', K]
            else:
                for u in range(ku):
                    _stochastic(f"{m}[u={u}]", arr[u])
                stacked.append(_log_of(arr))
        tm_key = draft.arrays.destinations[0].name
        log_params[tm_key] = np.stack(stacked)
        composite_keys.add(tm_key)
        for i, m in enumerate(members):
            _set_member_address(
                member_of,
                t,
                m,
                i,
                {
                    "matrices": (
                        tm_key,
                        "identity" if learned else "probability_to_log",
                    )
                },
            )
    hybrid_gfe_groups = tuple(draft.plan for draft in drafts.hybrid_gfe)
    for draft in drafts.hybrid_gfe:
        group = draft.plan
        gname, mtype, dims, k_sel = group.name, group.type, group.dims, group.k_sel
        members, goals = group.members, group.goal_members
        sources = _requirement_arrays(draft.arrays, resolved)
        packed = _stack_mix_params(t, mtype, dims, k_sel, members, sources)
        goal_params = _stack_gauss_params(
            t, "gaussian", (dims[0],), goals, sources
        )
        packed.update(goal_m=goal_params["m"], goal_W=goal_params["W"])
        _publish_required_parameters(log_params, draft.arrays, packed)

    if registered.grouping_problem:
        raise CompileError(registered.grouping_problem)

    gauss_groups: list[GaussGroupPlan] = []
    gauss_group_names: set[str] = set()
    gauss_data: dict[str, int] = {}
    n_obs_rows = 0
    for draft in registered.gaussian:
        if isinstance(draft, EqualityReference):
            plan = equality_of[draft.members[0]]
            if plan.name in gauss_group_names:
                raise CompileError(f"Gaussian group name collision: {plan.name!r}")
            gauss_group_names.add(plan.name)
            gauss_groups.append(plan)
            continue
        gname, gtype, roles, dims = draft.name, draft.type, draft.roles, draft.dims
        members, kern = list(draft.members), draft.kernel
        if gname in gauss_group_names:
            raise CompileError(f"Gaussian group name collision: {gname!r}")
        gauss_group_names.add(gname)
        obs_rows, observed_role = None, None
        if draft.observation is not None:
            observed_role = draft.observation.role_index
            rows = []
            for source in draft.observation.sources:
                if source.port is None:
                    raise KeyError(source.variable)
                gauss_data[source.port] = (n_obs_rows, source.dim)
                rows.append(n_obs_rows)
                n_obs_rows += 1
            obs_rows = np.array(rows, dtype=np.int64)
        if draft.path_problem:
            raise ValueError(draft.path_problem)
        paths = dict(draft.parameter_paths)
        gauss_groups.append(GaussGroupPlan(
            gname, gtype, roles, dims, tuple(members), draft.gather, draft.scatter,
            obs_rows, kern, draft.parameter_paths, observed_role,
        ))
        if kern is not None:
            try:
                from cffg.kernel_runtime import validate_gaussian_shapes

                packed = kernels.packed_parameters(kern, t, members, resolved)
                validate_gaussian_shapes(gauss_groups[-1], packed)
            except (ValueError, TypeError, KeyError, AttributeError) as err:
                raise CompileError(f"kernel {kern.impl_id!r}: {err}") from err
            log_params.update({paths[key]: value for key, value in packed.items()})
            specs = kern.binding.parameters
            if all(spec.key is not None for spec in specs.values()):
                runtime = {role: (paths[spec.key], spec.transform) for role, spec in specs.items()}
                for index, member in enumerate(members):
                    _set_member_address(member_of, t, member, index, runtime, specs)
                    if member in member_of:
                        writable_members.add(member)

    mix_groups = tuple(draft.plan for draft in drafts.mixtures)
    for draft in drafts.mixtures:
        group = draft.plan
        gname, mtype, dims, k_sel, members = group.name, group.type, group.dims, group.k_sel, group.members
        packed = _stack_mix_params(
            t, mtype, dims, k_sel, members, _requirement_arrays(draft.arrays, resolved),
        )
        _publish_required_parameters(log_params, draft.arrays, packed)
        runtime = (
            {
                "mean": (f"{gname}:m", "identity"),
                "cov": (f"{gname}:W", "covariance_to_precision"),
            }
            if mtype == "gaussian_mixture"
            else {
                "matrix": (f"{gname}:A", "identity"),
                "offset": (f"{gname}:b", "identity"),
                "cov": (f"{gname}:W", "covariance_to_precision"),
            }
        )
        for index, member in enumerate(members):
            _set_member_address(member_of, t, member, index, runtime)
            writable_members.add(member)

    recurrence_components = drafts.recurrence_components
    recurrence_groups = tuple(draft.plan for draft in drafts.recurrence)
    for draft in drafts.recurrence:
        group = draft.plan
        gname, card, drive_dim, members = group.name, group.cards[0], group.dim, group.members
        packed = _stack_recurrence_params(
            t, card, drive_dim, members, _requirement_arrays(draft.arrays, resolved),
        )
        _publish_required_parameters(log_params, draft.arrays, packed)
        runtime = {
            "weights": (f"{gname}:weights", "identity"),
            "offsets": (f"{gname}:offsets", "identity"),
        }
        for index, member in enumerate(members):
            _set_member_address(member_of, t, member, index, runtime)
            writable_members.add(member)

    prec_groups = tuple(draft.plan for draft in drafts.precision)
    for draft in drafts.precision:
        group = draft.plan
        gname, members = group.name, group.members
        packed = {"m": _stack_prec_means(t, members, _requirement_arrays(draft.arrays, resolved))}
        _publish_required_parameters(log_params, draft.arrays, packed)
        for index, member in enumerate(members):
            _set_member_address(
                member_of,
                t,
                member,
                index,
                {"mean": (f"{gname}:m", "identity")},
            )
            writable_members.add(member)

    ef_groups: list[EfGroupPlan] = []
    ef_data: dict[str, int] = {}
    n_ef_obs = 0
    for draft in registered.ef:
        if isinstance(draft, EqualityReference):
            ef_groups.append(equality_of[draft.members[0]])
            continue
        gname, ftype, family, roles = draft.name, draft.type, draft.family, draft.roles
        members, kern = list(draft.members), draft.kernel
        obs_rows = None
        if draft.observation is not None:
            rows = []
            for source in draft.observation.sources:
                if source.port is None:
                    raise KeyError(source.variable)
                ef_data[source.port] = (n_ef_obs, source.family)
                rows.append(n_ef_obs)
                n_ef_obs += 1
            obs_rows = np.array(rows, dtype=np.int64)
        if kern is not None and kern.pack_params is not None:
            try:
                packed = kern.pack_params(t, members, resolved)
            except ValueError as err:
                raise CompileError(str(err)) from err
            if set(packed) != set(kern.param_keys):
                raise CompileError(
                    f"kernel {kern.impl_id!r}: pack_params returned keys "
                    f"{sorted(packed)}, declared param_keys {sorted(kern.param_keys)}"
                )
            for key, arr in packed.items():
                log_params[f"{gname}:{key}"] = arr
        ef_groups.append(
            EfGroupPlan(gname, ftype, family, roles, tuple(members), draft.gather, draft.scatter, obs_rows)
        )
        # member_of stays discrete-only: per-member EF prior updates are the
        # conjugate-learning ticket's call (cff-x0sv), as with Gaussian.

    groups: list[GroupPlan] = []
    for draft in registered.discrete:
        if isinstance(draft, EqualityReference):
            groups.append(equality_of[draft.members[0]])
            continue
        if draft.problem:
            raise CompileError(draft.problem)
        if draft.path_problem:
            raise ValueError(draft.path_problem)
        plan = draft.plan
        gname, members, kern = plan.name, list(plan.members), plan.kernel
        paths = dict(plan.parameter_paths)
        groups.append(plan)
        if kern is not None:
            try:
                packed = kernels.packed_parameters(kern, t, members, resolved)
                from cffg.kernel_runtime import validate_discrete_shapes

                validate_discrete_shapes(groups[-1], packed, semiring)
            except (ValueError, TypeError, KeyError, AttributeError) as err:
                raise CompileError(f"kernel {kern.impl_id!r}: {err}") from err
            log_params.update({paths[key]: value for key, value in packed.items()})
            specs = kern.binding.parameters
            direct = all(spec.key is not None for spec in specs.values())
            runtime = {
                role: (paths[spec.key], spec.transform)
                for role, spec in specs.items() if spec.key is not None
            }
            for index, member in enumerate(members):
                if direct:
                    _set_member_address(member_of, t, member, index, runtime, specs)
                    writable_members.add(member)

    constraint_groups = []
    for draft in drafts.constraints:
        try:
            constraint_groups.append(resolve_constraint(draft, resolved))
        except ConstraintValidationError as error:
            raise CompileError(str(error)) from error

    rejects = semiring_problem(
        claims, any(group.mode == "vmp" for group in groups), bool(gauss_groups), bool(ef_groups),
    )
    if rejects:
        raise CompileError(
            f"semiring='max' (MAP) is defined in v1 for discrete BP groups and hard "
            f"constraints only; template {t.name!r} has: " + rejects
        )
    if drafts.delta_problem:
        raise CompileError(drafts.delta_problem)
    delta_pairs = drafts.delta_pairs

    # Interior Gaussian edges, grouped by dim, for the Bethe entropy ledger.
    by_dim: dict[int, list[list[int]]] = {}
    for e in edges.values():
        if e.dim and e.interior and e.name not in gfe_x_edges:
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

    obs_lengths = np.zeros(n_obs_rows, dtype=np.int64)
    for row, dim in gauss_data.values():
        obs_lengths[row] = dim

    if drafts.goal_parameter_problem:
        raise ValueError(drafts.goal_parameter_problem)

    return Arena(
        template_name=t.name,
        n_slots=topology.n_slots,
        message_starts=prefix_offsets(topology.message_lengths),
        edges=edges,
        groups=groups,
        data_slots=data_slots,
        semiring=semiring,
        dirichlet_groups=dirichlet_groups,
        constraint_groups=constraint_groups,
        log_params=log_params,
        n_gslots=topology.n_gslots,
        g_eta_starts=prefix_offsets(topology.g_eta_lengths),
        g_lam_starts=prefix_offsets(topology.g_lam_lengths),
        gauss_groups=gauss_groups,
        gauss_data=gauss_data,
        n_obs_rows=n_obs_rows,
        g_obs_starts=prefix_offsets(obs_lengths),
        mix_groups=mix_groups,
        recurrence_groups=recurrence_groups,
        recurrence_components=recurrence_components,
        prec_groups=prec_groups,
        gauss_interior=gauss_interior,
        n_efslots=topology.n_efslots,
        ef_groups=ef_groups,
        ef_data=ef_data,
        n_ef_obs=n_ef_obs,
        ef_interior=ef_interior,
        gfe_groups=gfe_groups,
        goal_parameter_plans=drafts.goal_parameters,
        mu3_groups=mu3_groups,
        gfe_tm_groups=gfe_tm_groups,
        hybrid_gfe_groups=hybrid_gfe_groups,
        gfe_x_edges=frozenset(gfe_x_edges),
        delta_pairs=delta_pairs,
        member_of=member_of,
        writable_members=frozenset(writable_members),
        learn_plans=_resolve_learnables(t, resolved, groups, member_of, composite_keys),
    )


def _resolve_learnables(
    t: Template,
    resolved: dict[str, np.ndarray],
    groups: list[GroupPlan],
    member_of: dict[str, MemberAddress],
    composite_keys: set[str],
) -> list[LearnPlan]:
    """Bind each declared learnable to its stat sources and write targets
    (cff-x0sv).

    Every factor reading one of the learnable's blobs becomes a target, so a
    posterior update reaches every instance that shares the parameter — the
    tying the IR already expresses by blob name. Sources are the subset that
    live in a plain discrete group, because those are the nodes whose beliefs
    Engine.node_beliefs reports; GFE composites are write-only here."""
    from cffg.learn import FAMILIES

    plans: list[LearnPlan] = []
    ftype_of_group = {g.name: g.type for g in groups}
    for name in sorted(t.learnables):
        spec = t.learnables[name]
        if spec.family not in FAMILIES:
            raise CompileError(
                f"learnable {name!r}: unknown conjugate family {spec.family!r} "
                f"(known: {sorted(FAMILIES)})"
            )
        blobs = set(spec.blobs)
        readers = sorted(
            f.name
            for f in t.factors.values()
            if blobs & set(f.param_blobs.values())
        )
        if not readers:
            raise CompileError(f"learnable {name!r}: no factor reads blobs {sorted(blobs)}")
        ftypes = {t.factors[r].type for r in readers}
        if len(ftypes) > 1:
            raise CompileError(
                f"learnable {name!r}: blobs {sorted(blobs)} are read by mixed factor "
                f"types {sorted(ftypes)}; one posterior needs one sufficient statistic"
            )
        ftype = ftypes.pop()
        family = FAMILIES[spec.family]
        # Only the blobs this template actually reads: one learnable spans a
        # probability blob and a counts blob (build_rollout's A/Ac, Bs/Bc), and
        # a perception graph resolves the first while a planning graph with
        # parameter uncertainty resolves the second.
        shapes = {b: tuple(np.shape(resolved[b])) for b in sorted(blobs) if b in resolved}
        # Which parameter ROLE each blob fills. A multi-parameter family
        # writes its outputs by role, so a gaussian_linear's matrix, offset
        # and cov blobs each receive their own part of one MNIW rollover.
        roles = {
            b: role
            for r in readers
            for role, b in t.factors[r].param_blobs.items()
            if b in blobs
        }
        try:
            family.check(name, ftype, shapes, roles)
        except ValueError as err:
            raise CompileError(str(err)) from None

        targets: list[LearnTarget] = []
        sources: list[str] = []
        for r in readers:
            if r not in member_of:
                raise CompileError(
                    f"learnable {name!r}: factor {r!r} ({t.factors[r].type}) has no "
                    "addressable parameter slot"
                )
            member = member_of[r]
            read = sorted(blobs & set(t.factors[r].param_blobs.values()))
            for blob in read:
                role = roles[blob]
                parameter = member.parameter(role)
                if parameter.blob != blob:
                    raise CompileError(
                        f"learnable {name!r}: factor {r!r} role {role!r} reads "
                        f"{parameter.blob!r}, not {blob!r}"
                    )
                if parameter.key in composite_keys:
                    form = (
                        "counts"
                        if parameter.key.endswith(":counts")
                        else "expected"
                    )
                    targets.append(
                        LearnTarget(
                            parameter.key,
                            parameter.index,
                            form,
                            "as_is",
                            blob,
                            role,
                            parameter.transform,
                        )
                    )
                    continue
                layout = (
                    "tm"
                    if ftype_of_group.get(parameter.key) == "transition_mixture"
                    else "as_is"
                )
                targets.append(
                    LearnTarget(
                        parameter.key,
                        parameter.index,
                        "expected",
                        layout,
                        blob,
                        role,
                        parameter.transform,
                    )
                )
            if all(p.key not in composite_keys for p in member.parameters) and (
                spec.sources is None or r in spec.sources
            ):
                sources.append(r)
        # A graph with no sources is normal, not an error: the planning
        # template reads A only through GFE composites, and learns nothing.
        # Statistics come from the perception template, parameters flow back.
        plans.append(
            LearnPlan(name, spec.family, ftype, shapes, tuple(sources), tuple(targets))
        )
    return plans


def _dirichlet_shape(t: Template, likelihood_name: str) -> tuple[tuple[int, ...], int]:
    """Compatibility entry for the specialized concentration layout owner."""
    try:
        return dirichlet_shape(t, likelihood_name)
    except ValueError as error:
        raise CompileError(str(error)) from error


def _resolve_arrays(
    t: Template, arrays: dict[str, np.ndarray] | None, root: str | Path | None
) -> dict[str, np.ndarray]:
    """Resolution order per blob: explicit arrays > existing file > init spec.
    Files beat init specs so a continual-learning checkpoint written to the
    blob path overrides the warm start on reload (cff-wx28)."""
    needed = {b for f in t.factors.values() for b in f.param_blobs.values()}
    covariance_blobs = {
        blob
        for factor in t.factors.values()
        if (kernel := kernels.kernel_for(factor.type)) is not None
        for role, blob in factor.param_blobs.items()
        if (spec := kernel.binding.parameters.get(role)) is not None
        and spec.transform == "covariance_to_precision"
    }
    out: dict[str, np.ndarray] = {}
    for name in sorted(needed):
        path = Path(root) / t.blobs[name].path if root is not None else None
        if arrays is not None and name in arrays:
            value = np.asarray(arrays[name])
        elif path is not None and path.exists():
            value = _load_blob(path, name)
        elif t.blobs[name].init:
            from cffg import inits

            shape = _blob_shape(t, name)
            if shape is None:
                raise CompileError(f"blob {name!r}: cannot infer shape for init spec")
            try:
                value = np.asarray(inits.from_spec(t.blobs[name].init, shape))
            except ValueError as e:
                raise CompileError(f"blob {name!r}: init spec failed: {e}") from None
        else:
            raise CompileError(
                f"blob {name!r}: no array override, no file"
                + (f" at {path}" if path is not None else " (no root directory)")
                + ", and no init spec"
            )
        if name in covariance_blobs:
            from cffg.params import _validate_covariance

            try:
                # Validate roundoff in the source dtype before host packing widens it.
                _validate_covariance(value, f"blob {name!r}: covariance")
            except ValueError as err:
                raise CompileError(str(err)) from err
        out[name] = np.asarray(value, dtype=np.float64)
    return out


def _blob_shape(t: Template, blob_name: str) -> tuple[int, ...] | None:
    """Expected array shape for a blob, inferred from the first factor role
    that references it (the compiler validates full agreement later)."""
    declared = set()
    for f in t.factors.values():
        kern = kernels.kernel_for(f.type)
        if kern is None or not kern.binding.parameters:
            continue
        sizes = {r: t.variables[v].dim or t.variables[v].cardinality for r, v in f.edges.items()}
        for role, blob in f.param_blobs.items():
            if blob == blob_name and role in kern.binding.parameters:
                declared.add(tuple(kern.binding.parameters[role].shape(sizes)))
    if len(declared) > 1:
        raise CompileError(f"blob {blob_name!r}: conflicting registered shapes {sorted(declared)}")
    if declared:
        return declared.pop()
    for f in t.factors.values():
        for role, b in f.param_blobs.items():
            if b != blob_name:
                continue
            if f.type in ("gamma", "beta"):
                return ()
            shape = specialized_initializer_shape(t, f, role)
            if shape is not None:
                return shape
    return None


def _load_blob(path: Path, name: str) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path))
    if path.suffix == ".safetensors":
        from safetensors.numpy import load_file

        tensors = load_file(path)
        key = name if name in tensors else next(iter(tensors)) if len(tensors) == 1 else None
        if key is None:
            raise CompileError(f"blob {name!r}: {path} holds {sorted(tensors)}, need {name!r}")
        return np.asarray(tensors[key])
    raise CompileError(f"blob {name!r}: unsupported format {path.suffix!r}")


def _stack_params(
    t: Template,
    ftype: str,
    cards: tuple[int, ...],
    members: list[str],
    resolved: dict[str, np.ndarray],
) -> np.ndarray | None:
    """Stack per-member parameter arrays into [M, ...] log-space, with shape
    and column-stochasticity checks ("true FE" needs normalized conditionals)."""
    if ftype == "equality":
        return None
    kern = kernels.kernel_for(ftype)
    if kern is None or kern.arena != "discrete":
        raise CompileError(f"factor type {ftype!r}: no discrete binding")
    packed = kernels.packed_parameters(kern, t, members, resolved)
    return np.asarray(kern.binding.potential(packed, cards))


def _stack_gauss_params(
    t: Template,
    ftype: str,
    dims: tuple[int, ...],
    members: list[str],
    resolved: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Per-member Gaussian parameters, stacked [M, ...]. Covariances are
    validated SPD and stored as precisions W = V⁻¹ (the form every rule
    consumes); the compile-time Cholesky is also the SPD check."""
    if ftype == "equality":
        return {}
    kern = kernels.kernel_for(ftype)
    if kern is None or kern.arena != "gaussian":
        raise CompileError(f"factor type {ftype!r}: no Gaussian binding")
    try:
        return kernels.packed_parameters(kern, t, members, resolved)
    except ValueError as err:
        raise CompileError(str(err)) from err


def _stack_mix_params(
    t: Template,
    ftype: str,
    dims: tuple[int, ...],
    k_sel: int,
    members: list[str],
    resolved: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Per-member hybrid parameters, stacked [M, K, ...] over the K regimes
    (cff-pksr). Blobs are per-regime stacks: mean [K, d]/cov [K, d, d] for
    gaussian_mixture, matrix [K, do, di]/offset [K, do]/cov [K, do, do] for
    gaussian_mixture_linear. Each cov[k] is validated symmetric + SPD (the
    Cholesky is the check) and stored as a precision W = V⁻¹, the form every
    mixg rule consumes."""
    out: dict[str, list[np.ndarray]] = {}
    for m in members:
        f = t.factors[m]
        blobs = {role: resolved[f.param_blobs[role]] for role in sorted(f.param_blobs)}
        cov = blobs["cov"]
        if ftype == "gaussian_mixture":
            (d,) = dims
            _expect(m, blobs["mean"], (k_sel, d))
            _expect(m, cov, (k_sel, d, d))
            out.setdefault("m", []).append(blobs["mean"])
        else:  # gaussian_mixture_linear
            do, di = dims
            _expect(m, blobs["matrix"], (k_sel, do, di))
            _expect(m, blobs["offset"], (k_sel, do))
            _expect(m, cov, (k_sel, do, do))
            out.setdefault("A", []).append(blobs["matrix"])
            out.setdefault("b", []).append(blobs["offset"])
        w_regimes = []
        for k in range(k_sel):
            ck = cov[k]
            if not np.allclose(ck, ck.T, atol=1e-12):
                raise CompileError(f"factor {m!r}: covariance not symmetric")
            try:
                np.linalg.cholesky(ck)
            except np.linalg.LinAlgError:
                raise CompileError(f"factor {m!r}: covariance not positive definite") from None
            w_regimes.append(np.linalg.inv(ck))
        out.setdefault("W", []).append(np.stack(w_regimes))
    return {k: np.stack(v) for k, v in out.items()}


def _stack_prec_means(
    t: Template, members: list[str], resolved: dict[str, np.ndarray]
) -> np.ndarray:
    """Stack per-member gaussian_prec means [M, 1]. The mean is the scalar m in
    N(x; m, τ⁻¹); the out edge is dim-1, so any size-1 blob is accepted and
    reshaped to a length-1 vector (the form gauss.prec_to_x consumes)."""
    means = []
    for m in members:
        arr = resolved[t.factors[m].param_blobs["mean"]]
        if arr.size != 1:
            raise CompileError(
                f"factor {m!r}: parameter 'mean' must be a scalar, got shape {arr.shape}"
            )
        means.append(arr.reshape(1))
    return np.stack(means)


def _stack_recurrence_params(
    t: Template,
    card: int,
    drive_dim: int,
    members: list[str],
    resolved: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Stack source-specific stick weights and offsets for one plan."""
    weights = []
    offsets = []
    for member in members:
        factor = t.factors[member]
        member_weights = resolved[factor.param_blobs["weights"]]
        member_offsets = resolved[factor.param_blobs["offsets"]]
        _expect(member, member_weights, (card, card - 1, drive_dim))
        _expect(member, member_offsets, (card, card - 1))
        if not (
            np.all(np.isfinite(member_weights))
            and np.all(np.isfinite(member_offsets))
        ):
            raise CompileError(
                f"factor {member!r}: recurrence weights and offsets must be finite"
            )
        weights.append(member_weights)
        offsets.append(member_offsets)
    return {"weights": np.stack(weights), "offsets": np.stack(offsets)}


def _log_of(arr: np.ndarray) -> np.ndarray:
    """Elementwise log with exact zeros mapped to NEG, the engine's finite
    log-zero (semantically zero: exp(NEG) == 0, and NEG - NEG == 0).

    The floor under the log branch is the dtype's smallest normal, so log(0)
    is never evaluated even for the entries the where-mask discards — a
    literal 1e-300 floor underflows to exactly 0 under float32 and both warns
    and returns -inf (cff-ey4p). Positive normal values are unchanged in any
    float dtype; positive subnormals clamp to log(tiny), which is finite.
    """
    dtype = arr.dtype if np.issubdtype(arr.dtype, np.floating) else np.float64
    return np.where(arr > 0.0, np.log(np.maximum(arr, np.finfo(dtype).tiny)), NEG)


def _expect(name: str, arr: np.ndarray, shape: tuple[int, ...]) -> None:
    if arr.shape != shape:
        raise CompileError(f"factor {name!r}: parameter shape {arr.shape} != expected {shape}")


def _stochastic(name: str, arr: np.ndarray) -> None:
    sums = arr.sum(axis=0)
    if not np.allclose(sums, 1.0, atol=1e-6):
        raise CompileError(
            f"factor {name!r}: conditional not column-stochastic (col sums {sums}); "
            "true-FE semantics require normalized factors"
        )

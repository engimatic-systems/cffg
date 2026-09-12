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
- Plan dataclasses and ``Arena`` define compiled structure and addressing.
- ``compile_template`` validates, classifies edges, groups factors, and packs
  all runtime plans.
- ``_resolve_learnables`` binds declarations to statistic sources and writes.
- ``_resolve_arrays``, ``_blob_shape``, and ``_stack_*`` load and validate
  parameter data.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from cffg import ef, kernels
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
from cffg.core.ir import FACTOR_TYPES, ConstraintKind, PortRole, Template
from cffg.core.plates import expand_plates
from cffg.core.validate import validate
from cffg.goal_parameters import build_goal_parameter_plans
from cffg.recurrence_topology import classify_recurrence_components
from cffg.topology import build_topology, prefix_offsets

__all__ = [
    "NEG",
    "PARAMETER_TRANSFORMS",
    "Arena",
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
    "CompileError",
    "compile_template",
]

class CompileError(ValueError):
    pass




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
    degrees = topology.degrees
    ports_by_variable = topology.ports_by_variable

    mean_field = {
        c.target for c in t.constraints if c.kind is ConstraintKind.MEAN_FIELD
    }
    psub = {
        c.target: c.detail_dict()
        for c in t.constraints
        if c.kind is ConstraintKind.P_SUBSTITUTION
    }
    for constraint in t.constraints:
        if constraint.kind is ConstraintKind.MOMENT_MATCHING:
            problems.append(
                f"factor {constraint.target!r}: legacy moment-matching syntax "
                "does not identify an incident edge endpoint or projection"
            )

    for f in t.factors.values():
        if f.param_edges:
            if f.type not in ("categorical", "matrix_cat", "transition_mixture"):
                problems.append(
                    f"factor {f.name!r}: Dirichlet parameter edges are supported only "
                    "for categorical, matrix_cat and transition_mixture likelihoods"
                )
            if f.name not in mean_field:
                problems.append(
                    f"factor {f.name!r}: latent parameter edge is not Dirichlet-closed "
                    "under exact BP; declare a mean_field constraint for VMP"
                )
            if f.name in psub:
                problems.append(
                    f"factor {f.name!r}: a P-substituted parameter edge needs the "
                    "separate functional mu3 contract (cff-almk/cff-iwgl), not a "
                    "Dirichlet natural message"
                )
        if f.name in mean_field and f.type == "equality":
            problems.append(
                f"factor {f.name!r}: mean-field bead on an equality node is ill-defined "
                "(geometric expectation of a hard constraint)"
            )
        if f.name in mean_field and FACTOR_TYPES[f.type].family == "gaussian":
            problems.append(
                f"factor {f.name!r}: mean-field bead on a Gaussian node is Gaussian VMP "
                "(ticket cff-7c5o)"
            )
        if f.name in psub and FACTOR_TYPES[f.type].family == "gaussian":
            problems.append(
                f"factor {f.name!r}: P-substitution on Gaussian nodes is open research "
                "(ticket cff-dfpi)"
            )
        kern = kernels.kernel_for(f.type)
        if kern is not None and not (
            f.name in psub and f.type in {"categorical", "matrix_cat", "transition_mixture"}
        ):
            requested = set()
            if f.name in mean_field:
                requested.add(ConstraintKind.MEAN_FIELD.value)
            if f.name in psub:
                requested.add(ConstraintKind.P_SUBSTITUTION.value)
            bad = sorted(requested - set(kern.constraint_modes))
            if bad:
                problems.append(
                    f"factor {f.name!r}: kernel {kern.impl_id!r} does not accept "
                    f"{bad} beads (conjugate BP is already exact on EF subgraphs)"
                )
        elif FACTOR_TYPES[f.type].family == "ef" and f.name in (mean_field | set(psub)):
            problems.append(
                f"factor {f.name!r}: beads on EF nodes are undefined in M-scope — "
                "conjugate BP is already exact on EF subgraphs (cff-8s6a)"
            )
        if FACTOR_TYPES[f.type].family == "hybrid" and f.name in mean_field:
            problems.append(
                f"factor {f.name!r}: hybrid beads are undefined for mean-field "
                "constraints; "
                "collapse rules are already assumed-density (cff-pksr)"
            )
        if f.type == "recurrence_link" and f.name in psub:
            problems.append(
                f"factor {f.name!r}: P-substitution on recurrence_link is "
                "unsupported; its PG/JJ coordinate defines a VFE bound"
            )
        if f.type in kernels.conditioned_out_types():
            out_var = f.edges["out"]
            out_ports = [
                p
                for p in ports_by_variable.get(out_var, ())
                if p.role is PortRole.DATA
            ]
            if degrees.get(out_var, 0) == 2 or not out_ports:
                problems.append(
                    f"factor {f.name!r}: {f.type} is a conditioned likelihood — its 'out' "
                    "edge must be a DATA port (unobserved predictives leave the family)"
                )
    problems.extend(topology.port_problems)
    if problems:
        raise CompileError(f"cannot compile {t.name!r}:\n  - " + "\n  - ".join(problems))

    # Parameter-edge shapes are fixed by the likelihood's structural roles.
    # The carrier VariableSpec stays schema-1 discrete and declares the
    # flattened size; EdgeInfo.param_shape retags it for the derived arena.
    param_specs: dict[str, tuple[tuple[int, ...], int, str]] = {}
    for f in t.factors.values():
        for var in f.param_edges.values():
            shape, axis = _dirichlet_shape(t, f.name)
            if var in param_specs:
                problems.append(
                    f"parameter edge {var!r}: more than one likelihood endpoint; "
                    "insert a supported parameter-tying structure (not available in D6a)"
                )
            else:
                param_specs[var] = (shape, axis, f.name)
            v = t.variables[var]
            size = int(np.prod(shape))
            if v.family != "discrete" or v.cardinality != size:
                problems.append(
                    f"parameter edge {var!r}: carrier cardinality must equal the "
                    f"flattened Dirichlet table size {size} for shape {shape}, got "
                    f"{v.family} cardinality {v.cardinality}"
                )
    if problems:
        raise CompileError(f"cannot compile {t.name!r}:\n  - " + "\n  - ".join(problems))

    endpoints = topology.endpoints
    edges = dict(topology.edges)
    for name, (shape, axis, _) in param_specs.items():
        edges[name] = replace(edges[name], param_shape=shape, param_axis=axis)
    slot_of, opposite = topology.slot_of, topology.opposite
    data_slots: dict[str, int] = {}
    gauss_data_ports: dict[str, str] = {}
    ef_data_ports: dict[str, str] = {}
    for candidate in topology.data_candidates:
        vname, port_name = candidate.variable, candidate.port
        variable = topology.variables[vname]
        fname, role = candidate.endpoint
        ftype_here = t.factors[fname].type
        if variable.dim:
            kern = kernels.kernel_for(ftype_here)
            conditioning = kern.binding.conditioning if kern is not None and kern.arena == "gaussian" else None
            if conditioning is not None and role == conditioning.role:
                gauss_data_ports[vname] = port_name
            elif ftype_here in ("gaussian_mixture", "gaussian_mixture_linear"):
                problems.append(
                    f"port {port_name!r}: observed mixture outputs are not "
                    f"supported yet (Gaussian DATA port on {fname!r}/{role!r})"
                )
            elif ftype_here == "gaussian_prec":
                problems.append(
                    f"port {port_name!r}: observed gaussian_prec outputs: bind "
                    "observations via a gaussian_linear emission or an equality branch"
                )
            else:
                problems.append(
                    f"port {port_name!r}: Gaussian DATA port on {fname!r}/{role!r} "
                    "requires a registered conditioning role"
                )
        elif variable.ef:
            if ftype_here in kernels.conditioned_out_types() and role == "out":
                ef_data_ports[vname] = port_name
            else:
                problems.append(
                    f"port {port_name!r}: EF DATA ports must clamp a conditioned "
                    f"observation kernel's 'out' edge "
                    f"({sorted(kernels.conditioned_out_types())}), not "
                    f"{fname!r}/{role!r}"
                )
        else:
            data_slots[port_name] = candidate.external_slot

    resolved = _resolve_arrays(t, arrays, root)
    member_of: dict[str, MemberAddress] = {}
    writable_members: set[str] = set()
    log_params: dict[str, np.ndarray] = {}
    consumed: set[str] = set()

    # A categorical goal facing a P-substituted observation node is absorbed
    # into that composite. If its statistic is a parameter edge, its
    # Dirichlet prior/update must therefore be owned by GfePlan rather than by
    # an ordinary standalone DirichletGroupPlan.
    gfe_goal_names: set[str] = set()
    for obs_name in psub:
        obs = t.factors[obs_name]
        if obs.type != "matrix_cat" or "out" not in obs.edges:
            continue
        facing = [
            factor_name
            for factor_name, _ in endpoints[obs.edges["out"]]
            if factor_name != obs_name
        ]
        if len(facing) == 1 and t.factors[facing[0]].type == "categorical":
            gfe_goal_names.add(facing[0])

    # In-graph conjugate parameter paths.  A schema-1 FFG parameter edge has
    # exactly two endpoints: one supported likelihood param role and one
    # dirichlet prior's ``out`` role.  The prior blob is the serialized source
    # of truth; the packed messages are derived runtime state.
    dir_buckets: dict[
        tuple[str, tuple[int, ...], tuple[int, ...], int],
        list[tuple[str, str, str, np.ndarray]],
    ] = {}
    gfe_goal_dirichlet: dict[str, tuple[str, str, np.ndarray]] = {}
    for var, (shape, axis, likelihood_name) in sorted(param_specs.items()):
        ends = endpoints[var]
        other = [(fname, role) for fname, role in ends if fname != likelihood_name]
        if len(other) != 1:
            problems.append(
                f"parameter edge {var!r}: requires exactly one facing Dirichlet prior, "
                f"got endpoints {list(ends)}"
            )
            continue
        prior_name, prior_role = other[0]
        prior = t.factors[prior_name]
        if prior.type != "dirichlet" or prior_role != "out":
            problems.append(
                f"parameter edge {var!r}: {likelihood_name!r} must face a dirichlet "
                f"prior on role 'out', got {prior_name!r}/{prior.type}:{prior_role}"
            )
            continue
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
        likelihood = t.factors[likelihood_name]
        if likelihood_name in gfe_goal_names:
            gfe_goal_dirichlet[likelihood_name] = (
                prior_name,
                var,
                alpha0,
            )
            consumed.update((likelihood_name, prior_name))
            continue
        roles = FACTOR_TYPES[likelihood.type].edge_roles
        cards = tuple(t.variables[likelihood.edges[r]].cardinality for r in roles)
        key = (likelihood.type, cards, shape, axis)
        dir_buckets.setdefault(key, []).append(
            (likelihood_name, prior_name, var, alpha0)
        )
        consumed.update((likelihood_name, prior_name))

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

    dirichlet_groups: list[DirichletGroupPlan] = []
    for (ftype, cards, shape, axis), entries in sorted(dir_buckets.items()):
        roles = FACTOR_TYPES[ftype].edge_roles
        gname = f"dirvmp:{ftype}:{'x'.join(map(str, cards))}"
        members = tuple(e[0] for e in entries)
        priors = tuple(e[1] for e in entries)
        gather_both = np.array(
            [
                [
                    [slot_of[(likelihood, r)], opposite[(likelihood, r)]]
                    for r in roles
                ]
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
        log_params[f"{gname}:alpha0"] = np.stack([e[3] for e in entries])

    # GFE composites: P-substituted obs factor + facing goal factor across x.
    gfe_x_edges: set[str] = set()
    gfe_buckets: dict[tuple, list[tuple[str, str]]] = {}  # cards/mode -> (obs, goal)
    tm_buckets: dict[tuple, list[str]] = {}
    hybrid_gfe_buckets: dict[tuple, list[tuple[str, str]]] = {}
    for fname, detail in sorted(psub.items()):
        f = t.factors[fname]
        if f.type == "transition_mixture" and detail.get("substituted_role") == "out":
            cards3 = tuple(
                t.variables[f.edges[r]].cardinality for r in ("out", "in", "selector")
            )
            learned = detail.get("parameter_uncertainty") == "dirichlet"
            tm_buckets.setdefault((cards3, learned), []).append(fname)
            consumed.add(fname)
            continue
        if f.type in ("gaussian_mixture", "gaussian_mixture_linear"):
            if detail.get("substituted_role") != "out":
                problems.append(
                    f"p_substitution on {fname!r}: hybrid GFE supports "
                    "substituted_role 'out' only"
                )
                continue
            unsupported = sorted(set(detail) - {"substituted_role"})
            if unsupported:
                problems.append(
                    f"p_substitution on {fname!r}: unsupported hybrid GFE detail "
                    f"keys {unsupported}"
                )
                continue
            x_var = f.edges["out"]
            facing = [fn for fn, _ in endpoints[x_var] if fn != fname]
            goal = t.factors[facing[0]] if len(facing) == 1 else None
            if (
                goal is None
                or goal.type != "gaussian"
                or goal.edges.get("out") != x_var
            ):
                problems.append(
                    f"p_substitution on {fname!r}: edge {x_var!r} must face "
                    "exactly one fixed Gaussian goal factor"
                )
                continue
            roles = FACTOR_TYPES[f.type].edge_roles
            dims = tuple(
                t.variables[f.edges[role]].dim
                for role in roles
                if role != "selector"
            )
            if f.type == "gaussian_mixture_linear":
                state_var = f.edges["in"]
                state_facing = [
                    fn for fn, _ in endpoints[state_var] if fn != fname
                ]
                if len(state_facing) != 1:
                    problems.append(
                        f"p_substitution on {fname!r}: hybrid GFE requires an "
                        "interior Gaussian state edge with a proper facing model"
                    )
                    continue
            k_sel = t.variables[f.edges["selector"]].cardinality
            hybrid_gfe_buckets.setdefault(
                (f.type, dims, k_sel), []
            ).append((fname, goal.name))
            consumed.update((fname, goal.name))
            gfe_x_edges.add(x_var)
            continue
        if f.type != "matrix_cat" or detail.get("substituted_role") != "out":
            problems.append(
                f"p_substitution on {fname!r}: supported on matrix_cat, "
                "transition_mixture, gaussian_mixture, and "
                "gaussian_mixture_linear nodes with substituted_role 'out' only"
            )
            continue
        x_var = f.edges["out"]
        facing = [fn for fn, _ in endpoints[x_var] if fn != fname]
        goal = t.factors[facing[0]] if facing else None
        goal_fixed = (
            goal is not None
            and goal.type == "categorical"
            and "statistic" in goal.param_blobs
        )
        goal_learned = (
            goal is not None
            and goal.type == "categorical"
            and "statistic" in goal.param_edges
        )
        if not goal_fixed and not goal_learned:
            problems.append(
                f"p_substitution on {fname!r}: edge {x_var!r} must face a categorical "
                "goal factor with a fixed statistic blob or Dirichlet parameter edge"
            )
            continue
        declared_goal = detail.get("goal_uncertainty")
        if declared_goal not in (None, "dirichlet"):
            problems.append(
                f"p_substitution on {fname!r}: unknown goal_uncertainty "
                f"{declared_goal!r} (known: 'dirichlet')"
            )
            continue
        if goal_learned != (declared_goal == "dirichlet"):
            problems.append(
                f"p_substitution on {fname!r}: a learned goal parameter edge and "
                "goal_uncertainty='dirichlet' must be declared together"
            )
            continue
        if goal_learned and goal.name not in gfe_goal_dirichlet:
            problems.append(
                f"p_substitution on {fname!r}: learned goal {goal.name!r} must "
                "face one proper Dirichlet prior"
            )
            continue
        ko, ki = (t.variables[f.edges[r]].cardinality for r in ("out", "in"))
        learned = detail.get("parameter_uncertainty") == "dirichlet"
        parameter_message = detail.get("parameter_message")
        if parameter_message not in (None, "mu3_is_v1"):
            problems.append(
                f"p_substitution on {fname!r}: unknown parameter_message "
                f"{parameter_message!r} (known: 'mu3_is_v1')"
            )
            continue
        mu3 = parameter_message == "mu3_is_v1"
        if mu3 and not learned:
            problems.append(
                f"p_substitution on {fname!r}: mu3_is_v1 requires "
                "parameter_uncertainty='dirichlet' and a proper count-array proposal"
            )
            continue
        if goal_learned and learned:
            problems.append(
                f"p_substitution on {fname!r}: simultaneous learned A and learned "
                "goal c is not yet a registered composite; land each coordinate separately"
            )
            continue
        gfe_buckets.setdefault(
            (ko, ki, learned, mu3, goal_learned), []
        ).append((fname, goal.name))
        consumed.update((fname, goal.name))
        gfe_x_edges.add(x_var)
    if problems:
        raise CompileError(f"cannot compile {t.name!r}:\n  - " + "\n  - ".join(problems))

    gfe_groups: list[GfePlan] = []
    # Params keys owned by a GFE composite: write targets for a learnable, but
    # never stat sources (the composite owns its own update path).
    composite_keys: set[str] = set()
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
            tuple(gfe_goal_dirichlet[g][0] for _, g in pairs)
            if goal_learned
            else ()
        )
        goal_prior_blobs = (
            tuple(
                t.factors[prior].param_blobs["counts"]
                for prior in goal_priors
            )
            if goal_learned
            else ()
        )
        goal_tie_groups = (
            tuple(
                tuple(
                    i
                    for i, value in enumerate(goal_prior_blobs)
                    if value == blob
                )
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
                gather_z=np.array([opposite[(o, "in")] for o, _ in pairs], dtype=np.int64),
                scatter_z=np.array([slot_of[(o, "in")] for o, _ in pairs], dtype=np.int64),
                scatter_x=np.array([slot_of[(o, "out")] for o, _ in pairs], dtype=np.int64),
                learned=learned,
                mu3=mu3,
                param_blobs=param_blobs,
                goal_learned=goal_learned,
                goal_members=(
                    tuple(goal for _, goal in pairs)
                    if goal_learned
                    else ()
                ),
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
        la, lc = [], []
        for obs_name, goal_name in pairs:
            a = resolved[next(iter(t.factors[obs_name].param_blobs.values()))]
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
                c = resolved[t.factors[goal_name].param_blobs["statistic"]]
                _expect(goal_name, c, (ko,))
                _stochastic(goal_name, c[:, None])
                lc.append(_log_of(c))
        a_key = f"{gname}:counts" if learned else f"{gname}:A"
        c_key = f"{gname}:c_alpha0" if goal_learned else f"{gname}:c"
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
        stacked = []
        for m in members:
            arr = resolved[t.factors[m].param_blobs["matrices"]]
            _expect(m, arr, (ku, ko, ki))
            if learned:
                if not np.all(arr > 0):
                    raise CompileError(f"factor {m!r}: Dirichlet counts must be > 0")
                stacked.append(arr)  # raw counts α [U, K', K]
            else:
                for u in range(ku):
                    _stochastic(f"{m}[u={u}]", arr[u])
                stacked.append(_log_of(arr))
        tm_key = f"{gname}:counts" if learned else f"{gname}:B"
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
        gfe_tm_groups.append(
            GfeTmPlan(gname, (ko, ki, ku), tuple(members), gather, scatter, learned)
        )

    hybrid_gfe_groups: list[HybridGfePlan] = []
    for (mtype, dims, k_sel), pairs in sorted(hybrid_gfe_buckets.items()):
        members = [member for member, _ in pairs]
        goals = [goal for _, goal in pairs]
        gauss_roles = tuple(
            role
            for role in FACTOR_TYPES[mtype].edge_roles
            if role != "selector"
        )
        gname = f"gfeH:{mtype}:{'x'.join(map(str, dims))}:{k_sel}"
        g_gather = np.array(
            [
                [opposite[(member, role)] for role in gauss_roles]
                for member in members
            ],
            dtype=np.int64,
        )
        g_scatter = np.array(
            [
                [slot_of[(member, role)] for role in gauss_roles]
                for member in members
            ],
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
        for key, value in _stack_mix_params(
            t, mtype, dims, k_sel, members, resolved
        ).items():
            log_params[f"{gname}:{key}"] = value
        goal_params = _stack_gauss_params(
            t, "gaussian", (dims[0],), goals, resolved
        )
        log_params[f"{gname}:goal_m"] = goal_params["m"]
        log_params[f"{gname}:goal_W"] = goal_params["W"]

    # Group remaining factors by (type, cardinality signature, mode); the
    # Gaussian family (and equality nodes on Gaussian edges) go to their own
    # arena's buckets, keyed by (type, dim signature, observed?); the EF
    # family (and equality nodes on EF edges) to theirs, keyed by (type,
    # message family).
    buckets: dict[tuple, list[str]] = {}
    gauss_buckets: dict[tuple, list[str]] = {}
    ef_buckets: dict[tuple[str, str], list[str]] = {}
    mix_buckets: dict[tuple, list[str]] = {}
    recurrence_buckets: dict[tuple[int, int], list[str]] = {}
    prec_members: list[str] = []
    constraint_names: list[str] = []
    for fname in sorted(t.factors):
        if fname in consumed:
            continue
        f = t.factors[fname]
        ft = FACTOR_TYPES[f.type]
        if ft.family == "constraint":
            constraint_names.append(fname)
            continue
        roles = ft.edge_roles
        gaussian = ft.family == "gaussian" or (
            ft.family == "any" and t.variables[f.edges[roles[0]]].dim > 0
        )
        if gaussian:
            dims = tuple(t.variables[f.edges[r]].dim for r in roles)
            kern = kernels.kernel_for(f.type)
            if f.type != "equality" and (kern is None or kern.arena != "gaussian"):
                raise CompileError(f"factor type {f.type!r} has no Gaussian kernel binding")
            conditioning = None if kern is None else kern.binding.conditioning
            observed = conditioning is not None and f.edges[conditioning.role] in gauss_data_ports
            gtype = (conditioning.plan_type or f"{f.type}:observed:{conditioning.role}") if observed else f.type
            gauss_buckets.setdefault((gtype, dims, f.type, observed), []).append(fname)
            continue
        if f.type == "gaussian_prec":  # mixed Gaussian/EF node: its own arena pair
            d_out = t.variables[f.edges["out"]].dim
            if d_out != 1:
                raise CompileError(
                    f"factor {fname!r}: gaussian_prec is scalar (dim-1 out edge); "
                    "vector precision-mean priors are cff-7c5o follow-up work"
                )
            prec_members.append(fname)
            continue
        if f.type == "recurrence_link":
            k_out = t.variables[f.edges["out"]].cardinality
            k_in = t.variables[f.edges["in"]].cardinality
            drive_dim = t.variables[f.edges["drive"]].dim
            if k_out != k_in or k_out < 2:
                raise CompileError(
                    f"factor {fname!r}: recurrence_link requires equal out/in "
                    f"cardinality K >= 2, got {(k_out, k_in)}"
                )
            recurrence_buckets.setdefault((k_out, drive_dim), []).append(fname)
            continue
        if ft.family == "hybrid":  # Gaussian-mixture nodes: their own arena pair
            dims = tuple(t.variables[f.edges[r]].dim for r in roles if r != "selector")
            k_sel = t.variables[f.edges["selector"]].cardinality
            mix_buckets.setdefault((f.type, dims, k_sel), []).append(fname)
            continue
        ef_family = ft.family == "ef" or (
            ft.family == "any" and bool(t.variables[f.edges[roles[0]]].ef)
        )
        if ef_family:
            kern = kernels.kernel_for(f.type)
            if kern is not None:  # executable registration owns the family
                family = kern.family
            elif f.type == "equality":  # structural: routed by its edges' family
                family = t.variables[f.edges[roles[0]]].ef
                if family not in ef.FAMILIES:
                    raise CompileError(
                        f"factor {fname!r}: no messages flow on observation-family "
                        f"({family!r}) edges — they are conditioned DATA ports only"
                    )
            else:
                raise CompileError(
                    f"factor {fname!r}: EF type {f.type!r} has no registered "
                    "FactorKernel — register one before compiling (cff-ri8m)"
                )
            ef_buckets.setdefault((f.type, family), []).append(fname)
            continue
        cards = tuple(t.variables[f.edges[r]].cardinality for r in roles)
        mode = "vmp" if fname in mean_field else "bp"
        buckets.setdefault((f.type, cards, mode), []).append(fname)

    gauss_groups: list[GaussGroupPlan] = []
    gauss_group_names: set[str] = set()
    gauss_data: dict[str, int] = {}
    n_obs_rows = 0
    for (gtype, dims, base, observed), members in sorted(gauss_buckets.items()):
        kern = kernels.kernel_for(base)
        roles = FACTOR_TYPES[base].edge_roles
        gname = f"gauss:{gtype}:{'x'.join(map(str, dims))}"
        if gname in gauss_group_names:
            raise CompileError(f"Gaussian group name collision: {gname!r}")
        gauss_group_names.add(gname)
        gather = np.array([[opposite[(m, r)] for r in roles] for m in members], dtype=np.int64)
        scatter = np.array([[slot_of[(m, r)] for r in roles] for m in members], dtype=np.int64)
        obs_rows = None
        observed_role = None
        if observed:
            role = kern.binding.conditioning.role
            observed_role = roles.index(role)
            rows = []
            for m in members:
                port = gauss_data_ports[t.factors[m].edges[role]]
                gauss_data[port] = (n_obs_rows, dims[observed_role])
                rows.append(n_obs_rows)
                n_obs_rows += 1
            obs_rows = np.array(rows, dtype=np.int64)
        paths = {} if kern is None else kernels.parameter_paths(kern, gname)
        gauss_groups.append(GaussGroupPlan(
            gname, gtype, roles, dims, tuple(members), gather, scatter, obs_rows,
            kern, tuple(paths.items()), observed_role,
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

    mix_groups: list[MixGroupPlan] = []
    for (mtype, dims, k_sel), members in sorted(mix_buckets.items()):
        gauss_roles = tuple(r for r in FACTOR_TYPES[mtype].edge_roles if r != "selector")
        gname = f"mix:{mtype}:{'x'.join(map(str, dims))}:{k_sel}"
        g_gather = np.array(
            [[opposite[(m, r)] for r in gauss_roles] for m in members], dtype=np.int64
        )
        g_scatter = np.array(
            [[slot_of[(m, r)] for r in gauss_roles] for m in members], dtype=np.int64
        )
        s_gather = np.array([opposite[(m, "selector")] for m in members], dtype=np.int64)
        s_scatter = np.array([slot_of[(m, "selector")] for m in members], dtype=np.int64)
        mix_groups.append(
            MixGroupPlan(
                gname, mtype, dims, k_sel, tuple(members),
                g_gather, g_scatter, s_gather, s_scatter,
            )
        )
        for key, arr in _stack_mix_params(t, mtype, dims, k_sel, members, resolved).items():
            log_params[f"{gname}:{key}"] = arr
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

    recurrence_members = {
        member for members in recurrence_buckets.values() for member in members
    }
    ordinary_bp = {
        member
        for (_, _, mode), members in buckets.items()
        if mode == "bp"
        for member in members
    }
    recurrence_components = tuple(
        RecurrenceComponentPlan(*component)
        for component in classify_recurrence_components(
            t,
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
            [
                [slot_of[(member, role)] for role in ("out", "in")]
                for member in members
            ],
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
        for key, value in _stack_recurrence_params(
            t, card, drive_dim, members, resolved
        ).items():
            log_params[f"{gname}:{key}"] = value
        runtime = {
            "weights": (f"{gname}:weights", "identity"),
            "offsets": (f"{gname}:offsets", "identity"),
        }
        for index, member in enumerate(members):
            _set_member_address(member_of, t, member, index, runtime)
            writable_members.add(member)

    prec_groups: list[PrecGroupPlan] = []
    if prec_members:
        members = sorted(prec_members)
        gname = "prec:gaussian_prec"
        x_scatter = np.array([slot_of[(m, "out")] for m in members], dtype=np.int64)
        x_gather_both = np.stack(
            [x_scatter, np.array([opposite[(m, "out")] for m in members], dtype=np.int64)],
            axis=-1,
        )
        tau_scatter = np.array([slot_of[(m, "prec")] for m in members], dtype=np.int64)
        tau_gather_both = np.stack(
            [tau_scatter, np.array([opposite[(m, "prec")] for m in members], dtype=np.int64)],
            axis=-1,
        )
        prec_groups.append(
            PrecGroupPlan(
                gname, tuple(members), x_gather_both, tau_gather_both, x_scatter, tau_scatter
            )
        )
        log_params[f"{gname}:m"] = _stack_prec_means(t, members, resolved)
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
    for (ftype, family), members in sorted(ef_buckets.items()):
        kern = kernels.kernel_for(ftype)
        roles = ("a", "b", "c") if kern is None else kern.message_roles
        gname = f"ef:{ftype}:{family}"
        gather = np.array([[opposite[(m, r)] for r in roles] for m in members], dtype=np.int64)
        scatter = np.array([[slot_of[(m, r)] for r in roles] for m in members], dtype=np.int64)
        obs_rows = None
        if kern is not None and kern.observed_out is not None:
            rows = []
            for m in members:
                out_var = t.factors[m].edges["out"]
                ef_data[ef_data_ports[out_var]] = (n_ef_obs, t.variables[out_var].ef)
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
            EfGroupPlan(gname, ftype, family, roles, tuple(members), gather, scatter, obs_rows)
        )
        # member_of stays discrete-only: per-member EF prior updates are the
        # conjugate-learning ticket's call (cff-x0sv), as with Gaussian.

    groups: list[GroupPlan] = []
    for (ftype, cards, mode), members in sorted(buckets.items()):
        roles = FACTOR_TYPES[ftype].edge_roles
        gname = f"{mode}:{ftype}:{'x'.join(map(str, cards))}"
        gather = np.array(
            [[opposite[(m, r)] for r in roles] for m in members], dtype=np.int64
        )
        scatter = np.array(
            [[slot_of[(m, r)] for r in roles] for m in members], dtype=np.int64
        )
        both = np.stack([scatter, gather], axis=-1)  # own & opposite = the edge pair
        kern = kernels.kernel_for(ftype)
        if ftype != "equality" and (kern is None or kern.arena != "discrete"):
            raise CompileError(f"factor type {ftype!r} has no discrete kernel binding")
        if kern is not None and semiring not in kern.binding.semirings:
            raise CompileError(f"kernel {kern.impl_id!r} does not support {semiring!r}")
        paths = {} if kern is None else kernels.parameter_paths(kern, gname)
        groups.append(GroupPlan(
            gname, ftype, roles, cards, tuple(members), gather, scatter, mode, both,
            kern, tuple(paths.items()),
        ))
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

    constraint_groups: list[ConstraintPlan] = []
    for fname in constraint_names:
        f = t.factors[fname]
        n = len(f.edges)
        roles = tuple(f"x{i}" for i in range(n))
        cards = tuple(t.variables[f.edges[r]].cardinality for r in roles)
        gather = np.array([opposite[(fname, r)] for r in roles], dtype=np.int64)
        scatter = np.array([slot_of[(fname, r)] for r in roles], dtype=np.int64)
        k = 0
        table_arr = None
        if f.type == "alldifferent":
            K = cards[0]
            if any(c != K for c in cards):
                raise CompileError(
                    f"factor {fname!r}: alldifferent needs one shared cardinality, got {cards}"
                )
            if K < n:
                raise CompileError(
                    f"factor {fname!r}: alldifferent over {n} edges needs cardinality "
                    f"K >= {n}, got K={K} (pigeonhole: unsatisfiable)"
                )
            if n > 10 or K > 10:
                raise CompileError(
                    f"factor {fname!r}: exact alldifferent DP is capped at n, K <= 10 "
                    f"(got n={n}, K={K}); larger n is the permanent-approximation regime "
                    "(Huang & Jebara, arXiv:0908.1769)"
                )
        elif f.type in ("exactly_k", "at_most_k"):
            if any(c != 2 for c in cards):
                raise CompileError(
                    f"factor {fname!r}: {f.type} operates on binary edges (cardinality 2), "
                    f"got {cards}"
                )
            karr = resolved[f.param_blobs["k"]]
            if karr.size != 1 or int(karr.reshape(())) != karr.reshape(()):
                raise CompileError(f"factor {fname!r}: parameter 'k' must be a scalar integer")
            k = int(karr.reshape(()))
            if not 0 <= k <= n:
                raise CompileError(
                    f"factor {fname!r}: k={k} out of range [0, {n}] for {n} binary edges"
                )
        else:  # table
            table_arr = np.asarray(resolved[f.param_blobs["table"]])
            if table_arr.ndim != 2 or table_arr.shape[1] != n:
                raise CompileError(
                    f"factor {fname!r}: table must be [T, {n}] (T tuples over {n} edges), "
                    f"got shape {table_arr.shape}"
                )
            if table_arr.shape[0] >= 100000:
                raise CompileError(
                    f"factor {fname!r}: table has {table_arr.shape[0]} tuples (>= 1e5); "
                    "large tables leave the explicit-enumeration regime"
                )
            if not np.issubdtype(table_arr.dtype, np.integer) and not np.allclose(
                table_arr, np.round(table_arr)
            ):
                raise CompileError(f"factor {fname!r}: table entries must be integers")
            table_arr = np.round(table_arr).astype(np.int64)
            for j in range(n):
                if table_arr[:, j].min() < 0 or table_arr[:, j].max() >= cards[j]:
                    raise CompileError(
                        f"factor {fname!r}: table column {j} has values outside [0, {cards[j]})"
                    )
        constraint_groups.append(
            ConstraintPlan(
                name=fname,
                type=f.type,
                cards=cards,
                members=(fname,),
                gather=gather,
                scatter=scatter,
                k=k,
                table=table_arr,
            )
        )

    if semiring == "max":
        rejects = []
        if any(g.mode == "vmp" for g in groups):
            rejects.append("mean-field (vmp) beads")
        if dirichlet_groups:
            rejects.append("Dirichlet parameter-edge VMP groups")
        if gfe_groups or gfe_tm_groups or hybrid_gfe_groups:
            rejects.append("P-substitution / GFE composites")
        if gauss_groups or mix_groups or recurrence_groups or prec_groups:
            rejects.append("Gaussian/hybrid groups")
        if ef_groups:
            rejects.append("EF groups")
        if rejects:
            raise CompileError(
                f"semiring='max' (MAP) is defined in v1 for discrete BP groups and hard "
                f"constraints only; template {t.name!r} has: " + ", ".join(rejects)
            )

    delta_edges = sorted(
        c.target for c in t.constraints if c.kind is ConstraintKind.DELTA
    )
    for v in delta_edges:
        if edges[v].param_shape:
            raise CompileError(
                f"delta constraint on Dirichlet parameter edge {v!r}: parameter "
                "posteriors are continuous simplex distributions, not categorical states"
            )
        if edges[v].dim:
            raise CompileError(
                f"delta constraint on Gaussian edge {v!r}: the MAP of a Gaussian "
                "marginal is its mean — read gaussian_marginals instead (hybrid "
                "pointmass forms: cff-pksr)"
            )
        if edges[v].ef:
            raise CompileError(
                f"delta constraint on EF edge {v!r}: read the mode from "
                "ef_marginals instead (hybrid pointmass forms: cff-pksr)"
            )
    delta_pairs = np.array(
        [[edges[v].slot0, edges[v].slot1] for v in delta_edges], dtype=np.int64
    ).reshape(-1, 2)

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
        goal_parameter_plans=build_goal_parameter_plans(gfe_groups),
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
        try:
            semantic_identity = family.canonical_identity(spec.semantic_identity)
        except ValueError as err:
            raise CompileError(f"learnable {name!r}: {err}") from None
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
            LearnPlan(
                name, spec.family, semantic_identity, ftype, shapes,
                tuple(sources), tuple(targets)
            )
        )
    return plans


def _dirichlet_shape(t: Template, likelihood_name: str) -> tuple[tuple[int, ...], int]:
    """Canonical concentration layout and output axis for a D6a likelihood."""
    f = t.factors[likelihood_name]
    card = lambda role: t.variables[f.edges[role]].cardinality  # noqa: E731
    if f.type == "categorical":
        return (card("out"),), 0
    if f.type == "matrix_cat":
        return (card("out"), card("in")), 0
    if f.type == "transition_mixture":
        return (card("selector"), card("out"), card("in")), 1
    raise CompileError(
        f"factor {likelihood_name!r}: no Dirichlet VMP layout for {f.type!r}"
    )


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
            card = lambda r: t.variables[f.edges[r]].cardinality  # noqa: E731
            dim = lambda r: t.variables[f.edges[r]].dim  # noqa: E731
            if f.type == "dirichlet":
                parameter_var = f.edges["out"]
                for likelihood in t.factors.values():
                    if parameter_var in likelihood.param_edges.values():
                        return _dirichlet_shape(t, likelihood.name)[0]
                return (card("out"),)
            if f.type == "gaussian_mixture":
                d, k = dim("out"), card("selector")
                return {"mean": (k, d), "cov": (k, d, d)}[role]
            if f.type == "gaussian_mixture_linear":
                do, di, k = dim("out"), dim("in"), card("selector")
                return {"matrix": (k, do, di), "offset": (k, do), "cov": (k, do, do)}[role]
            if f.type == "recurrence_link":
                k, d = card("out"), dim("drive")
                return {
                    "weights": (k, k - 1, d),
                    "offsets": (k, k - 1),
                }[role]
            if f.type in ("gamma", "beta"):
                return ()  # scalar prior parameters
            if f.type == "gaussian_prec":
                return (dim("out"),)  # scalar mean (dim-1 out edge)
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

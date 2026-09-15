"""Declaration-only admission and ownership for named specialized families.

DATA ports request runtime clamps under the family's conditioning contract.
Unbonded GLUE ports are half-edges with implicit marginalization. PRIOR and
GOAL ports must be composed or bound before compilation; otherwise a factor
is missing from the declared model.
"""

import numpy as np

from cffg import ef
from cffg.core.ir import FACTOR_TYPES, ConstraintKind, Template
from cffg.core.moment_matching import moment_matching_endpoint
from cffg.registered.capabilities import (
    conditioned_output_problems,
    constraint_problems,
    snapshot_registry,
)
from cffg.registered.contracts import RegistrySnapshot
from cffg.topology import TopologyLayout

from .contracts import (
    EqualityRoute,
    GfeClaim,
    MixtureClaim,
    ParameterClaim,
    ParameterLayout,
    SpecializedClaims,
    TransitionGfeClaim,
)


def _admission(t, constraints, registry):
    problems = []
    topology = t
    mean_field = {c.target for c in constraints if c.kind is ConstraintKind.MEAN_FIELD}
    psub = {
        c.target: c.detail_dict()
        for c in constraints
        if c.kind is ConstraintKind.P_SUBSTITUTION
    }
    for constraint in constraints:
        if constraint.kind is ConstraintKind.MOMENT_MATCHING:
            try:
                endpoint = moment_matching_endpoint(constraint, t)
            except ValueError as exc:
                problems.append(str(exc))
                continue
            kernel = registry.bindings.get(t.factors[constraint.target].type)
            projections = getattr(getattr(kernel, "binding", None), "moment_projections", ())
            if endpoint.projection.value not in projections:
                problems.append(
                    f"factor {constraint.target!r}: no registered moment-matching rule "
                    f"for projection {endpoint.projection.value!r} at role {endpoint.role!r}"
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
        problems.extend(
            constraint_problems(
                f,
                registry,
                mean_field=f.name in mean_field,
                psub=f.name in psub,
                delegated_psub=f.name in psub
                and f.type in {"categorical", "matrix_cat", "transition_mixture"},
            )
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
        problems.extend(conditioned_output_problems(f, topology, registry))
    problems.extend(topology.port_problems)

    return mean_field, psub, tuple(problems)


def _parameter_layouts(t):
    problems = []
    param_specs: dict[str, tuple[tuple[int, ...], int, str]] = {}
    for f in t.factors.values():
        for var in f.param_edges.values():
            shape, axis = dirichlet_shape(t, f.name)
            if var in param_specs:
                problems.append(
                    f"parameter edge {var!r}: more than one likelihood endpoint; "
                    "insert a supported parameter-tying structure (not available in D6a)"
                )
            else:
                param_specs[var] = ParameterLayout(shape, axis, f.name)
            v = t.variables[var]
            size = int(np.prod(shape))
            if v.family != "discrete" or v.cardinality != size:
                problems.append(
                    f"parameter edge {var!r}: carrier cardinality must equal the "
                    f"flattened Dirichlet table size {size} for shape {shape}, got "
                    f"{v.family} cardinality {v.cardinality}"
                )

    return param_specs, tuple(problems)


def dirichlet_shape(
    t: TopologyLayout, likelihood_name: str
) -> tuple[tuple[int, ...], int]:
    """Canonical concentration layout and output axis for a D6a likelihood."""
    f = t.factors[likelihood_name]
    card = lambda role: t.variables[f.edges[role]].cardinality
    if f.type == "categorical":
        return (card("out"),), 0
    if f.type == "matrix_cat":
        return (card("out"), card("in")), 0
    if f.type == "transition_mixture":
        return (card("selector"), card("out"), card("in")), 1
    raise ValueError(
        f"factor {likelihood_name!r}: no Dirichlet VMP layout for {f.type!r}"
    )


def _parameter_claims(t, layouts, psub):
    endpoints = t.endpoints
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

    claims = []
    for variable, layout in sorted(layouts.items()):
        ends = endpoints[variable]
        other = [(name, role) for name, role in ends if name != layout.likelihood]
        prior_name = None
        problem = None
        if len(other) != 1:
            problem = (
                f"parameter edge {variable!r}: requires exactly one facing Dirichlet prior, "
                f"got endpoints {list(ends)}"
            )
        else:
            name, role = other[0]
            prior = t.factors[name]
            if prior.type != "dirichlet" or role != "out":
                problem = (
                    f"parameter edge {variable!r}: {layout.likelihood!r} must face a dirichlet "
                    f"prior on role 'out', got {name!r}/{prior.type}:{role}"
                )
            else:
                prior_name = name
        claims.append(
            ParameterClaim(
                variable,
                layout,
                prior_name,
                layout.likelihood in gfe_goal_names,
                problem,
            )
        )
    return tuple(claims)


def _composites(t, psub, parameters):
    endpoints = t.endpoints
    problems = []
    consumed = set()
    gfe_goal_dirichlet = {
        claim.layout.likelihood: claim
        for claim in parameters
        if claim.is_goal and claim.prior is not None
    }
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
                t.variables[f.edges[role]].dim for role in roles if role != "selector"
            )
            if f.type == "gaussian_mixture_linear":
                state_var = f.edges["in"]
                state_facing = [fn for fn, _ in endpoints[state_var] if fn != fname]
                if len(state_facing) != 1:
                    problems.append(
                        f"p_substitution on {fname!r}: hybrid GFE requires an "
                        "interior Gaussian state edge with a proper facing model"
                    )
                    continue
            k_sel = t.variables[f.edges["selector"]].cardinality
            hybrid_gfe_buckets.setdefault((f.type, dims, k_sel), []).append(
                (fname, goal.name)
            )
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
        gfe_buckets.setdefault((ko, ki, learned, mu3, goal_learned), []).append(
            (fname, goal.name)
        )
        consumed.update((fname, goal.name))
        gfe_x_edges.add(x_var)

    gfe = tuple(
        GfeClaim((ko, ki), learned, mu3, goal_learned, tuple(pairs))
        for (ko, ki, learned, mu3, goal_learned), pairs in sorted(gfe_buckets.items())
    )
    transition = tuple(
        TransitionGfeClaim(cards, learned, tuple(members))
        for (cards, learned), members in sorted(tm_buckets.items())
    )
    hybrid = tuple(
        MixtureClaim(
            kind, dims, selector, tuple(m for m, _ in pairs), tuple(g for _, g in pairs)
        )
        for (kind, dims, selector), pairs in sorted(hybrid_gfe_buckets.items())
    )
    return gfe, transition, hybrid, consumed, gfe_x_edges, tuple(problems)


def _remaining(t, consumed):
    mixtures, recurrence, precision, constraints, equalities, problems = (
        {},
        {},
        [],
        [],
        {},
        {},
    )
    for name in sorted(t.factors):
        if name in consumed:
            continue
        factor = t.factors[name]
        kind = FACTOR_TYPES[factor.type]
        if kind.family == "constraint":
            constraints.append(name)
        elif factor.type == "equality":
            variable = t.variables[factor.edges[kind.edge_roles[0]]]
            if variable.dim:
                dims = tuple(
                    t.variables[factor.edges[role]].dim for role in kind.edge_roles
                )
                equalities[name] = EqualityRoute(
                    "gaussian", ("equality", dims, "equality", False)
                )
            elif variable.ef:
                if variable.ef not in ef.FAMILIES:
                    problems[name] = (
                        f"factor {name!r}: no messages flow on observation-family "
                        f"({variable.ef!r}) edges — they are conditioned DATA ports only"
                    )
                equalities[name] = EqualityRoute("ef", ("equality", variable.ef))
            else:
                cards = tuple(
                    t.variables[factor.edges[role]].cardinality
                    for role in kind.edge_roles
                )
                equalities[name] = EqualityRoute("discrete", ("equality", cards, "bp"))
        elif factor.type == "gaussian_prec":
            if t.variables[factor.edges["out"]].dim != 1:
                problems[name] = (
                    f"factor {name!r}: gaussian_prec is scalar (dim-1 out edge); "
                    "vector precision-mean priors are cff-7c5o follow-up work"
                )
            else:
                precision.append(name)
        elif factor.type == "recurrence_link":
            ko = t.variables[factor.edges["out"]].cardinality
            ki = t.variables[factor.edges["in"]].cardinality
            dim = t.variables[factor.edges["drive"]].dim
            if ko != ki or ko < 2:
                problems[name] = (
                    f"factor {name!r}: recurrence_link requires equal out/in "
                    f"cardinality K >= 2, got {(ko, ki)}"
                )
            else:
                recurrence.setdefault((ko, dim), []).append(name)
        elif kind.family == "hybrid":
            dims = tuple(
                t.variables[factor.edges[role]].dim
                for role in kind.edge_roles
                if role != "selector"
            )
            selector = t.variables[factor.edges["selector"]].cardinality
            mixtures.setdefault((factor.type, dims, selector), []).append(name)
    mix = tuple(
        MixtureClaim(kind, dims, selector, tuple(members))
        for (kind, dims, selector), members in sorted(mixtures.items())
    )
    return mix, recurrence, tuple(precision), tuple(constraints), equalities, problems


def specialized_initializer_shape(t, factor, role, layouts=None, *, composite=False):
    """Family declarations without executing an ordinary kernel's callbacks.

    Built-in shapes belong to their composite when used there. The compatibility
    loader retains registered declarations first and its narrower fallback.
    """
    cards = {
        name: t.variables[variable].cardinality
        for name, variable in factor.edges.items()
    }
    dims = {name: t.variables[variable].dim for name, variable in factor.edges.items()}
    declared = {}
    if composite and factor.type == "categorical":
        declared = {"statistic": (cards["out"],)}
    elif composite and factor.type == "matrix_cat":
        declared = {"matrix": (cards["out"], cards["in"])}
    elif composite and factor.type == "transition_mixture":
        declared = {"matrices": (cards["selector"], cards["out"], cards["in"])}
    elif composite and factor.type == "gaussian":
        d = dims["out"]
        declared = {"mean": (d,), "cov": (d, d)}
    elif factor.type == "dirichlet":
        variable = factor.edges["out"]
        layout = layouts.get(variable) if layouts is not None else None
        shape = layout.shape if layout else None
        if layouts is None:
            for likelihood in t.factors.values():
                if variable in likelihood.param_edges.values():
                    shape = dirichlet_shape(t, likelihood.name)[0]
                    break
        declared = {"counts": shape or (cards["out"],)}
    elif factor.type == "gaussian_mixture":
        d, k = dims["out"], cards["selector"]
        declared = {"mean": (k, d), "cov": (k, d, d)}
    elif factor.type == "gaussian_mixture_linear":
        do, di, k = dims["out"], dims["in"], cards["selector"]
        declared = {"matrix": (k, do, di), "offset": (k, do), "cov": (k, do, do)}
    elif factor.type == "recurrence_link":
        k, d = cards["out"], dims["drive"]
        declared = {"weights": (k, k - 1, d), "offsets": (k, k - 1)}
    elif factor.type == "gaussian_prec":
        declared = {"mean": (dims["out"],)}
    return declared.get(role)


def _initializer_shapes(t, layouts, consumed):
    shapes = {}
    for factor in t.factors.values():
        for role in factor.param_blobs:
            shape = specialized_initializer_shape(
                t, factor, role, layouts, composite=factor.name in consumed
            )
            if shape is not None:
                shapes[factor.name, role] = shape
    return shapes


def initializer_fallbacks(t, layouts=None):
    """Historical fallback shapes, distinct from canonical composite shapes."""
    return {
        (factor.name, role): shape
        for _, factor in sorted(t.factors.items())
        for role in sorted(factor.param_blobs)
        if (shape := specialized_initializer_shape(t, factor, role, layouts))
        is not None
    }


def discover_specialized(
    template: Template,
    topology: TopologyLayout,
    semiring: str,
    registry: RegistrySnapshot | None = None,
) -> SpecializedClaims:
    """Interpret a validated immutable declaration snapshot, without blob access.

    The template argument makes the stage boundary explicit; no mutable template
    reference escapes. Diagnostics are data until the compiler's existing gates.
    """
    registry = snapshot_registry() if registry is None else registry
    mean_field, psub, admission = _admission(topology, template.constraints, registry)
    layouts, layout_problems = ({}, ()) if admission else _parameter_layouts(topology)
    parameters = (
        ()
        if admission or layout_problems
        else _parameter_claims(topology, layouts, psub)
    )
    if admission or layout_problems:
        gfe, transition, hybrid, consumed, excluded, composite_problems = (
            (),
            (),
            (),
            set(),
            set(),
            (),
        )
        mix, recurrence, precision, constraints, equalities, grouping_problems = (
            (),
            {},
            (),
            (),
            {},
            {},
        )
        shapes = {}
    else:
        gfe, transition, hybrid, consumed, excluded, composite_problems = _composites(
            topology, psub, parameters
        )
        consumed.update(
            name
            for claim in parameters
            if claim.prior is not None
            for name in (claim.layout.likelihood, claim.prior)
        )
        mix, recurrence, precision, constraints, equalities, grouping_problems = (
            _remaining(topology, consumed)
        )
        shapes = _initializer_shapes(topology, layouts, consumed)
    conditioning = {}
    for candidate in topology.data_candidates:
        name, role = candidate.endpoint
        kind = topology.factors[name].type
        if not topology.variables[candidate.variable].dim:
            continue
        if kind in ("gaussian_mixture", "gaussian_mixture_linear"):
            conditioning[candidate.port] = (
                f"port {candidate.port!r}: observed mixture outputs are not "
                f"supported yet (Gaussian DATA port on {name!r}/{role!r})"
            )
        elif kind == "gaussian_prec":
            conditioning[candidate.port] = (
                f"port {candidate.port!r}: observed gaussian_prec outputs: bind "
                "observations via a gaussian_linear emission or an equality branch"
            )
    return SpecializedClaims(
        topology=topology,
        registry=registry,
        semiring=semiring,
        mean_field=mean_field,
        parameter_layouts=layouts,
        parameters=parameters,
        gfe=gfe,
        transition_gfe=transition,
        hybrid_gfe=hybrid,
        mixtures=mix,
        recurrence=recurrence,
        precision=precision,
        constraints=constraints,
        equalities=equalities,
        consumed=consumed,
        excluded_edges=excluded,
        admission_problems=admission,
        layout_problems=layout_problems,
        composite_problems=composite_problems,
        grouping_problems=grouping_problems,
        conditioning_problems=conditioning,
        initializer_shapes=shapes,
    )

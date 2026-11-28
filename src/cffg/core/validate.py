"""Template validation: the port invariant and FFG well-formedness.

The invariant that makes composition, unrolling, and "no dangling edges" one
rule (docs/ARCHITECTURE.md §1): every variable has degree 1 or 2; a degree-1
variable is exposed through exactly one port; a degree-2 variable through none.
Validation collects *all* problems before raising, so edit-action failures
report the complete damage, not the first symptom.
"""

from __future__ import annotations

from cffg.core.constraint_details import p_substitution_conflicts
from cffg.core.moment_matching import moment_matching_problems
from cffg.core.functions import FunctionSpec
from cffg.core.plate_grid import PlateGrid
from cffg.core import gaussian_agent
from cffg.core.ir import (
    EDGE_CONSTRAINTS,
    EF_EDGE_FAMILIES,
    FACTOR_TYPES,
    HYBRID_EDGE_FAMILIES,
    Broadcast,
    BoundaryFactorization,
    Chain,
    ConstraintKind,
    Expose,
    PlateSpec,
    PortRole,
    Template,
)
from cffg.ef import FAMILIES as EF_FAMILIES


class ModelError(ValueError):
    """Raised on invalid templates; `problems` lists every violation found."""

    def __init__(self, template_name: str, problems: list[str]):
        self.problems = problems
        bullet_list = "\n  - ".join(problems)
        super().__init__(f"invalid template {template_name!r}:\n  - {bullet_list}")


def validate(t: Template) -> None:
    problems = _model_problems(t, frozenset({id(t)}), 1)
    if problems:
        raise ModelError(t.name, problems)


def _model_problems(t: Template, ancestors: frozenset[int], copies: int) -> list[str]:
    return [
        *_check_factors(t),
        *_check_degrees_and_ports(t),
        *_check_constraints(t),
        *_check_plates(t, ancestors, copies),
        *_check_learnables(t),
        *_check_agent(t),
    ]


def _check_factors(t: Template) -> list[str]:
    problems = []
    for f in t.factors.values():
        ft = FACTOR_TYPES.get(f.type)
        if ft is None:
            problems.append(f"factor {f.name!r}: unknown type {f.type!r}")
            continue
        if ft.requires_function and not isinstance(f.function, FunctionSpec):
            problems.append(f"factor {f.name!r}: requires a portable mean FunctionSpec")
        if not ft.requires_function and f.function is not None:
            problems.append(f"factor {f.name!r}: this type does not accept a mean function")
        if ft.variadic:
            n = len(f.edges)
            want = {f"x{i}" for i in range(n)}
            if n < 2 or set(f.edges) != want:
                problems.append(
                    f"factor {f.name!r}: variadic type {f.type!r} needs contiguous edge "
                    f"roles x0..x{{n-1}} (n >= 2), got {sorted(f.edges)}"
                )
        elif set(f.edges) != set(ft.edge_roles):
            problems.append(
                f"factor {f.name!r}: edge roles {sorted(f.edges)} != required {sorted(ft.edge_roles)}"
            )
        for role in ft.param_roles:
            sources = (role in f.param_edges) + (role in f.param_blobs)
            if sources != 1:
                problems.append(
                    f"factor {f.name!r}: param role {role!r} must be bound by exactly "
                    f"one of edge/blob, got {sources}"
                )
        for role in set(f.param_edges) | set(f.param_blobs):
            if role not in ft.param_roles:
                problems.append(f"factor {f.name!r}: unknown param role {role!r}")
        for role, var in f.all_edges().items():
            if var not in t.variables:
                problems.append(f"factor {f.name!r}: role {role!r} references unknown variable {var!r}")
        for role, blob in f.param_blobs.items():
            if blob not in t.blobs:
                problems.append(f"factor {f.name!r}: role {role!r} references unknown blob {blob!r}")
        if ft.symmetric_domain:
            domains = {
                t.variables[v].domain_key
                for v in f.edges.values()
                if v in t.variables
            }
            if len(domains) > 1:
                problems.append(
                    f"factor {f.name!r}: type {f.type!r} requires one shared domain, got {sorted(domains)}"
                )
        if ft.family in ("discrete", "gaussian", "ef"):
            wrong = {
                v: t.variables[v].family
                for v in f.edges.values()
                if v in t.variables and t.variables[v].family != ft.family
            }
            if wrong:
                problems.append(
                    f"factor {f.name!r}: type {f.type!r} needs {ft.family} edges, got "
                    + ", ".join(f"{v!r} ({fam})" for v, fam in sorted(wrong.items()))
                )
        if ft.family == "constraint":
            wrong = {
                v: t.variables[v].family
                for v in f.edges.values()
                if v in t.variables and t.variables[v].family != "discrete"
            }
            if wrong:
                problems.append(
                    f"factor {f.name!r}: constraint type {f.type!r} needs discrete edges, got "
                    + ", ".join(f"{v!r} ({fam})" for v, fam in sorted(wrong.items()))
                )
        for role, want in EF_EDGE_FAMILIES.get(f.type, {}).items():
            var = f.edges.get(role)
            if var in t.variables and t.variables[var].ef not in ("", want):
                problems.append(
                    f"factor {f.name!r}: role {role!r} needs an ef={want!r} edge, "
                    f"got {var!r} (ef={t.variables[var].ef!r})"
                )
        for role, want in HYBRID_EDGE_FAMILIES.get(f.type, {}).items():
            var = f.edges.get(role)
            if var in t.variables and t.variables[var].family != want:
                problems.append(
                    f"factor {f.name!r}: role {role!r} needs a {want} edge, "
                    f"got {var!r} ({t.variables[var].family})"
                )
    return problems


def _check_degrees_and_ports(t: Template) -> list[str]:
    problems = []
    degrees = {name: 0 for name in t.variables}
    for factor in t.factors.values():
        for variable in factor.all_edges().values():
            if variable in degrees:
                degrees[variable] += 1
    ports_by_variable: dict[str, list] = {name: [] for name in t.variables}
    for p in t.ports.values():
        if p.variable not in t.variables:
            problems.append(f"port {p.name!r}: unknown variable {p.variable!r}")
        else:
            ports_by_variable[p.variable].append(p)

    # Deferred import: cffg.kernels imports cffg.core.ir, so a module-level
    # import here would cycle through the cffg.core package init.
    from cffg.kernels import observation_families

    obs_families = observation_families()  # built-ins + registered kernels
    for v in t.variables.values():
        if v.ef and v.ef not in EF_FAMILIES and v.ef not in obs_families:
            problems.append(
                f"variable {v.name!r}: unknown EF family {v.ef!r}; message families "
                f"are {sorted(EF_FAMILIES)}, observation families {sorted(obs_families)}"
            )
        deg = degrees[v.name]
        ports = ports_by_variable[v.name]
        n_ports = len(ports)
        if deg == 0:
            problems.append(f"variable {v.name!r}: unused (degree 0)")
        elif deg > 2:
            problems.append(
                f"variable {v.name!r}: degree {deg} > 2; FFG edges join at most two factors "
                "(insert an equality factor to branch)"
            )
        elif deg == 1 and n_ports != 1:
            problems.append(
                f"variable {v.name!r}: dangling (degree 1) so it must have exactly one port, has {n_ports}"
            )
        elif deg == 2 and n_ports != 0:
            problems.append(
                f"variable {v.name!r}: fully connected (degree 2) but exposed by "
                f"{[p.name for p in ports]}"
            )
    return problems


def _copy_port(spec: PlateSpec, grid: PlateGrid, name: str) -> tuple[PortRole | None, str | None]:
    # Broadcast hubs have no copy coordinates at their own level.
    if any(isinstance(bond, Broadcast) and bond.hub == name for bond in spec.bonds):
        return PortRole.GLUE, None
    fields = name.split("/", len(grid.dims))
    if len(fields) != len(grid.dims) + 1:
        return None, None
    try:
        coordinate = tuple(int(field) for field in fields[:-1])
    except ValueError:
        return None, None
    if tuple(map(str, coordinate)) != tuple(fields[:-1]):
        return None, "copy coordinates must use their canonical integer names"
    try:
        grid.ordinal(coordinate)
    except ValueError as error:
        return None, str(error)

    port_name = fields[-1]
    role, reason = _fragment_port(spec.fragment, port_name)
    if role is None:
        return None, reason
    for bond in spec.bonds:
        if isinstance(bond, Broadcast) and bond.port == port_name:
            return None, "copy port is consumed by a Broadcast"
        if isinstance(bond, Chain) and not grid.chain_problems(bond.axis, bond.offset):
            position = coordinate[bond.axis]
            if ((bond.port_a == port_name and position + bond.offset < grid.dims[bond.axis])
                    or (bond.port_b == port_name and position >= bond.offset)):
                return None, "copy port is consumed by a Chain"
    return role, None


def _fragment_port(fragment: Template, name: str) -> tuple[PortRole | None, str | None]:
    port = fragment.ports.get(name)
    if port is not None:
        return port.role, None

    explanation = None
    for plate, spec in fragment.plates.items():
        prefix = plate + "/"
        if not name.startswith(prefix):
            continue
        try:
            grid = PlateGrid(spec.dims)
        except (TypeError, ValueError):
            continue  # The declaration itself reports the invalid grid.
        role, reason = _copy_port(spec, grid, name[len(prefix):])
        if role is not None:
            return role, None
        if reason is not None:
            explanation = reason
    return None, explanation


def _check_plates(t: Template, ancestors: frozenset[int], copies: int) -> list[str]:
    """Validate plate dimensions, fragment declarations and surviving ports."""
    problems: list[str] = []
    copy_limit = (1 << 63) - 1
    for name, spec in sorted(t.plates.items()):
        try:
            grid = PlateGrid(spec.dims)
        except (TypeError, ValueError) as err:
            problems.append(f"plate {name!r}: {err}")
            grid = None
        combined = copies if grid is None else copies * grid.count
        if combined > copy_limit >= copies:
            problems.append(
                f"plate {name!r}: dims {spec.dims} with {copies} enclosing copies "
                "exceed the signed 64-bit copy count"
            )

        frag = spec.fragment
        if id(frag) in ancestors:
            problems.append(f"plate {name!r}: cyclic plate fragment reference to {frag.name!r}")
        else:
            problems.extend(
                f"plate {name!r} fragment: {problem}"
                for problem in _model_problems(frag, ancestors | {id(frag)}, min(combined, copy_limit + 1))
            )
        if frag.agent is not None:
            problems.append(
                f"plate {name!r}: fragment declares an AgentSpec; the runtime "
                "interface belongs on the complete host Template"
            )
        for blob in spec.per_copy_blobs:
            if blob not in frag.blobs:
                problems.append(f"plate {name!r}: per_copy_blob {blob!r} is not a fragment blob")
        for rule in spec.bonds:
            if isinstance(rule, Chain):
                if grid is not None:
                    problems.extend(f"plate {name!r}: {reason}"
                                    for reason in grid.chain_problems(rule.axis, rule.offset))
                for role, port_name in (("port_a", rule.port_a), ("port_b", rule.port_b)):
                    port_role, reason = _fragment_port(frag, port_name)
                    if port_role is None:
                        message = f"plate {name!r}: chain {role} {port_name!r} is not a fragment port"
                        problems.append(f"{message}: {reason}" if reason else message)
                    elif port_role is not PortRole.GLUE:
                        problems.append(
                            f"plate {name!r}: chain {role} {port_name!r} must be a glue port "
                            f"(has role {port_role.value})"
                        )
            elif isinstance(rule, Broadcast):
                port_role, reason = _fragment_port(frag, rule.port)
                if port_role is None:
                    message = f"plate {name!r}: broadcast port {rule.port!r} is not a fragment port"
                    problems.append(f"{message}: {reason}" if reason else message)
            elif not isinstance(rule, Expose):  # pragma: no cover - closed vocabulary
                problems.append(f"plate {name!r}: unknown wiring rule {rule!r}")
    return problems


def _check_learnables(t: Template) -> list[str]:
    """Structural checks on declared learnables (cff-x0sv). The conjugate
    family and per-factor-type sufficient statistic are checked at compile
    time, where the factor types and resolved arrays are known; core stays
    JAX-free and checks names, semantic identities, and disjointness."""
    problems: list[str] = []
    owner: dict[str, str] = {}
    from cffg.core.learning_identity import (
        LEARNING_IDENTITIES, check_identity, check_scoped_name,
    )

    for name, spec in sorted(t.learnables.items()):
        try:
            if not spec.semantic_identity and spec.family not in LEARNING_IDENTITIES:
                raise ValueError(f"unknown conjugate family {spec.family!r} needs an explicit semantic identity")
            check_identity(name, spec.semantic_identity)
            expected = LEARNING_IDENTITIES.get(spec.family)
            if expected is not None and spec.semantic_identity != expected:
                raise ValueError(f"unknown or incompatible semantic identity {spec.semantic_identity!r}")
            for scoped in (name, *spec.blobs, *(spec.sources or ())):
                check_scoped_name(scoped)
            if spec.sources is not None and len(set(spec.sources)) != len(spec.sources):
                raise ValueError("duplicate statistic sources")
        except ValueError as error:
            problems.append(f"learnable {name!r}: {error}")
        if spec.name != name:
            problems.append(f"learnable {name!r}: spec name is {spec.name!r}")
        if not spec.blobs:
            problems.append(f"learnable {name!r}: declares no blobs")
        for b in spec.blobs:
            if b not in t.blobs:
                problems.append(f"learnable {name!r}: unknown blob {b!r}")
            elif b in owner:
                problems.append(
                    f"learnable {name!r}: blob {b!r} is already learned by "
                    f"{owner[b]!r} — one posterior per blob"
                )
            else:
                owner[b] = name
        for f in spec.sources or ():
            if f not in t.factors:
                problems.append(f"learnable {name!r}: unknown source factor {f!r}")
            elif not (set(spec.blobs) & set(t.factors[f].param_blobs.values())):
                problems.append(
                    f"learnable {name!r}: source {f!r} does not read any of its blobs"
                )
    return problems


def _check_agent(t: Template) -> list[str]:
    """Validate the serialized model/runtime bindings (cff-nhxf).

    The runtime is intentionally name-agnostic, so every role error must be
    caught here while the message can still use the model's own names.
    """
    if t.agent is None:
        return []

    spec = t.agent
    if isinstance(spec, (gaussian_agent.GaussianPlanSpec, gaussian_agent.GaussianFilterSpec)):
        return gaussian_agent.validate_window(t, spec)
    problems: list[str] = []
    action = t.variables.get(spec.action)
    if action is None:
        problems.append(f"agent action readout: unknown variable {spec.action!r}")
    elif action.family != "discrete":
        problems.append(
            f"agent action readout {spec.action!r}: expected discrete variable, "
            f"got {action.family}"
        )

    if (spec.action_port is None) != (spec.observation_port is None):
        problems.append(
            "agent perception bindings: action_port and observation_port must "
            "be declared together"
        )
    for role, name in (
        ("action", spec.action_port),
        ("observation", spec.observation_port),
    ):
        if name is None:
            continue
        port = t.ports.get(name)
        if port is None:
            problems.append(f"agent {role} DATA port: unknown port {name!r}")
        elif port.role is not PortRole.DATA:
            problems.append(
                f"agent {role} port {name!r}: expected role 'data', "
                f"got {port.role.value!r}"
            )
    if (
        action is not None
        and spec.action_port in t.ports
        and t.ports[spec.action_port].variable in t.variables
    ):
        port_var = t.variables[t.ports[spec.action_port].variable]
        if port_var.domain_key != action.domain_key:
            problems.append(
                f"agent action port {spec.action_port!r}: domain "
                f"{port_var.domain_key} != action readout {spec.action!r} "
                f"domain {action.domain_key}"
            )
    if (
        spec.action_port is not None
        and spec.action_port == spec.observation_port
    ):
        problems.append("agent action and observation DATA ports must be different")

    if not spec.beliefs:
        problems.append("agent declares no boundary beliefs")
        return problems

    mean_field = {
        c.target for c in t.constraints if c.kind is ConstraintKind.MEAN_FIELD
    }
    seen_names: set[str] = set()
    seen_blocks: dict[str, str] = {}
    for belief in spec.beliefs:
        label = f"agent belief {belief.name!r}"
        if not belief.name:
            problems.append("agent belief name must not be empty")
        elif belief.name in seen_names:
            problems.append(f"agent belief name {belief.name!r} is duplicated")
        seen_names.add(belief.name)

        if not belief.blocks:
            problems.append(f"{label}: declares no state blocks")
        for block in belief.blocks:
            if not block:
                problems.append(f"{label}: state block names must not be empty")
            elif block in seen_blocks:
                problems.append(
                    f"{label}: state block {block!r} is already carried by "
                    f"belief {seen_blocks[block]!r}"
                )
            else:
                seen_blocks[block] = belief.name

        prior = t.factors.get(belief.prior)
        prior_var = None
        prior_blob = None
        if prior is None:
            problems.append(f"{label}: unknown prior factor {belief.prior!r}")
        elif prior.type != "categorical":
            problems.append(
                f"{label}: prior target {belief.prior!r} must be a categorical "
                f"factor, got {prior.type!r}"
            )
        else:
            prior_name = prior.edges.get("out")
            prior_var = t.variables.get(prior_name or "")
            prior_blob = prior.param_blobs.get("statistic")
            if prior_blob is None:
                problems.append(
                    f"{label}: prior target {belief.prior!r} must read a "
                    "categorical statistic blob"
                )

        posterior = t.variables.get(belief.posterior)
        if posterior is None:
            problems.append(
                f"{label}: unknown filter posterior variable {belief.posterior!r}"
            )
        elif posterior.family != "discrete":
            problems.append(
                f"{label}: filter posterior {belief.posterior!r} must be "
                f"discrete, got {posterior.family}"
            )
        if (
            prior_var is not None
            and posterior is not None
            and prior_var.domain_key != posterior.domain_key
        ):
            problems.append(
                f"{label}: prior domain {prior_var.domain_key} != filter "
                f"posterior {belief.posterior!r} domain {posterior.domain_key}"
            )

        if belief.factorization is BoundaryFactorization.JOINT:
            if belief.constraints:
                problems.append(
                    f"{label}: joint boundary must not name mean-field "
                    f"constraints {list(belief.constraints)}"
                )
        elif belief.factorization is BoundaryFactorization.MEAN_FIELD:
            if len(belief.blocks) != 1:
                problems.append(
                    f"{label}: mean-field boundary must carry exactly one state "
                    f"block, got {belief.blocks}"
                )
            if not belief.constraints:
                problems.append(
                    f"{label}: mean-field boundary must name at least one "
                    "Template mean_field constraint"
                )
            for target in belief.constraints:
                if target not in mean_field:
                    problems.append(
                        f"{label}: factorization target {target!r} is not an "
                        "explicit Template mean_field constraint"
                    )
                elif belief.posterior in t.variables and belief.posterior not in (
                    t.factors[target].all_edges().values()
                ):
                    problems.append(
                        f"{label}: mean_field factor {target!r} does not touch "
                        f"filter posterior {belief.posterior!r}"
                    )
        else:
            problems.append(
                f"{label}: unknown factorization {belief.factorization!r}"
            )

        if belief.episode_learnable is not None:
            learnable = t.learnables.get(belief.episode_learnable)
            if learnable is None:
                problems.append(
                    f"{label}: unknown episode learnable "
                    f"{belief.episode_learnable!r}"
                )
            elif prior_blob is not None and prior_blob not in learnable.blobs:
                problems.append(
                    f"{label}: episode learnable {belief.episode_learnable!r} "
                    f"does not own prior blob {prior_blob!r}"
                )
    return problems


def _check_constraints(t: Template) -> list[str]:
    problems = []
    for c in t.constraints:
        if c.kind in EDGE_CONSTRAINTS:
            if c.target not in t.variables:
                problems.append(f"{c.kind.value} constraint: unknown variable {c.target!r}")
            continue
        f = t.factors.get(c.target)
        if f is None:
            problems.append(f"{c.kind.value} constraint: unknown factor {c.target!r}")
            continue
        if c.kind is ConstraintKind.P_SUBSTITUTION:
            role = c.detail_dict().get("substituted_role")
            if role is None:
                problems.append(f"p_substitution on {c.target!r}: missing detail 'substituted_role'")
            elif role not in f.all_edges():
                problems.append(
                    f"p_substitution on {c.target!r}: role {role!r} is not one of its edges"
                )
    problems.extend(p_substitution_conflicts(t.constraints))
    problems.extend(moment_matching_problems(t))
    return problems

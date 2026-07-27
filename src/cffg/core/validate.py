"""Template validation: the port invariant and FFG well-formedness.

The invariant that makes composition, unrolling, and "no dangling edges" one
rule (docs/ARCHITECTURE.md §1): every variable has degree 1 or 2; a degree-1
variable is exposed through exactly one port; a degree-2 variable through none.
Validation collects *all* problems before raising, so edit-action failures
report the complete damage, not the first symptom.
"""

from __future__ import annotations

from cffg.core.ir import (
    EDGE_CONSTRAINTS,
    EF_EDGE_FAMILIES,
    FACTOR_TYPES,
    HYBRID_EDGE_FAMILIES,
    Broadcast,
    Chain,
    ConstraintKind,
    Expose,
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
    problems: list[str] = []

    problems += _check_factors(t)
    problems += _check_degrees_and_ports(t)
    problems += _check_constraints(t)
    problems += _check_plates(t)
    problems += _check_learnables(t)

    if problems:
        raise ModelError(t.name, problems)


def _check_factors(t: Template) -> list[str]:
    problems = []
    for f in t.factors.values():
        ft = FACTOR_TYPES.get(f.type)
        if ft is None:
            problems.append(f"factor {f.name!r}: unknown type {f.type!r}")
            continue
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
    for p in t.ports.values():
        if p.variable not in t.variables:
            problems.append(f"port {p.name!r}: unknown variable {p.variable!r}")

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
        deg = t.degree(v.name)
        n_ports = len(t.ports_of(v.name))
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
                f"{[p.name for p in t.ports_of(v.name)]}"
            )
    return problems


def _check_plates(t: Template) -> list[str]:
    """Validate every plate SPEC (cff-cpxj): the fragment on its own, the
    dims arity (1-D/2-D in phase 1), and each wiring rule's references."""
    problems: list[str] = []
    for name, spec in sorted(t.plates.items()):
        if not 1 <= len(spec.dims) <= 2:
            problems.append(
                f"plate {name!r}: dims {spec.dims} must be 1-D or 2-D in phase 1"
            )
        if any(d < 1 for d in spec.dims):
            problems.append(f"plate {name!r}: dims {spec.dims} must all be >= 1")
        frag = spec.fragment
        try:
            validate(frag)
        except ModelError as err:
            problems += [f"plate {name!r} fragment: {p}" for p in err.problems]
        if frag.plates:
            problems.append(
                f"plate {name!r}: fragment declares nested plates "
                f"{sorted(frag.plates)}; nested plate grounding is not "
                "supported in phase 1"
            )
        for b in spec.per_copy_blobs:
            if b not in frag.blobs:
                problems.append(f"plate {name!r}: per_copy_blob {b!r} is not a fragment blob")
        for rule in spec.bonds:
            if isinstance(rule, Chain):
                if not 0 <= rule.axis < len(spec.dims):
                    problems.append(
                        f"plate {name!r}: chain axis {rule.axis} out of range for dims {spec.dims}"
                    )
                if rule.offset < 1:
                    problems.append(f"plate {name!r}: chain offset {rule.offset} must be >= 1")
                for role, pname in (("port_a", rule.port_a), ("port_b", rule.port_b)):
                    port = frag.ports.get(pname)
                    if port is None:
                        problems.append(
                            f"plate {name!r}: chain {role} {pname!r} is not a fragment port"
                        )
                    elif port.role is not PortRole.GLUE:
                        problems.append(
                            f"plate {name!r}: chain {role} {pname!r} must be a glue port "
                            f"(has role {port.role.value})"
                        )
            elif isinstance(rule, Broadcast):
                if rule.port not in frag.ports:
                    problems.append(
                        f"plate {name!r}: broadcast port {rule.port!r} is not a fragment port"
                    )
            elif isinstance(rule, Expose):
                pass
            else:  # pragma: no cover - closed vocabulary
                problems.append(f"plate {name!r}: unknown wiring rule {rule!r}")
    return problems


def _check_learnables(t: Template) -> list[str]:
    """Structural checks on declared learnables (cff-x0sv). The conjugate
    family and per-factor-type sufficient statistic are checked at compile
    time, where the factor types and resolved arrays are known; core stays
    JAX-free and checks only names and disjointness."""
    problems: list[str] = []
    owner: dict[str, str] = {}
    for name, spec in sorted(t.learnables.items()):
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
    return problems

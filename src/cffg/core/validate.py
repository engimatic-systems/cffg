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
    ConstraintKind,
    Template,
)
from cffg.ef import FAMILIES as EF_FAMILIES, OBS_FAMILIES as EF_OBS_FAMILIES


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

    if problems:
        raise ModelError(t.name, problems)


def _check_factors(t: Template) -> list[str]:
    problems = []
    for f in t.factors.values():
        ft = FACTOR_TYPES.get(f.type)
        if ft is None:
            problems.append(f"factor {f.name!r}: unknown type {f.type!r}")
            continue
        if set(f.edges) != set(ft.edge_roles):
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

    for v in t.variables.values():
        if v.ef and v.ef not in EF_FAMILIES and v.ef not in EF_OBS_FAMILIES:
            problems.append(
                f"variable {v.name!r}: unknown EF family {v.ef!r}; message families "
                f"are {sorted(EF_FAMILIES)}, observation families {sorted(EF_OBS_FAMILIES)}"
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

"""Template composition and unrolling.

Composition bonds ports: two dangling edges (degree 1 each) unify into one
fully-connected edge (degree 2), and both ports disappear. Everything else is
namespacing. Unrolling a temporal slice is repeated self-composition along
glue ports — there is no separate unrolling machinery to maintain.

Name scoping uses `/` (e.g. "t3/z_out"), which is reserved and rejected in
user-authored names elsewhere only by convention; compose enforces prefix
uniqueness instead.
"""

from __future__ import annotations

import dataclasses

from cffg.core.ir import (
    AgentSpec,
    BeliefBinding,
    BlobRef,
    ConstraintSpec,
    FactorSpec,
    LearnableSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.core.validate import validate

SCOPE_SEP = "/"


class ComposeError(ValueError):
    pass


def _require_flat(template: Template) -> None:
    if template.plates:
        names = sorted({
            name for plate in template.plates.values() for name in plate.fragment.learnables
        })
        raise ComposeError(
            f"compact plates cannot be composed directly (learnables {names}); "
            "call expand_plates before scoping or merging"
        )


def scoped(template: Template, prefix: str) -> Template:
    """A copy of `template` with every name under `prefix/`. Blob names are
    scoped too (their *paths* are not — two scopes may share a parameter file)."""
    _require_flat(template)
    p = prefix + SCOPE_SEP

    def s(name: str) -> str:
        return p + name

    return Template(
        name=s(template.name),
        variables={s(v.name): dataclasses.replace(v, name=s(v.name)) for v in template.variables.values()},
        factors={
            s(f.name): FactorSpec(
                s(f.name),
                f.type,
                edges={r: s(v) for r, v in f.edges.items()},
                param_edges={r: s(v) for r, v in f.param_edges.items()},
                param_blobs={r: s(b) for r, b in f.param_blobs.items()},
                function=f.function,
            )
            for f in template.factors.values()
        },
        constraints=[dataclasses.replace(c, target=s(c.target)) for c in template.constraints],
        ports={s(pt.name): PortSpec(s(pt.name), s(pt.variable), pt.role) for pt in template.ports.values()},
        blobs={s(b.name): BlobRef(s(b.name), b.path, b.init) for b in template.blobs.values()},
        # Learnables scope with their blobs: two scoped copies learn separate
        # posteriors, matching the blob-name tying rule (cff-x0sv).
        learnables={
            s(ln.name): LearnableSpec(
                s(ln.name),
                ln.family,
                tuple(s(b) for b in ln.blobs),
                None if ln.sources is None else tuple(s(f) for f in ln.sources),
                ln.semantic_identity,
            )
            for ln in template.learnables.values()
        },
        agent=(
            AgentSpec(
                action=s(template.agent.action),
                action_port=(
                    s(template.agent.action_port)
                    if template.agent.action_port is not None
                    else None
                ),
                observation_port=(
                    s(template.agent.observation_port)
                    if template.agent.observation_port is not None
                    else None
                ),
                beliefs=tuple(
                    BeliefBinding(
                        name=s(belief.name),
                        blocks=tuple(s(block) for block in belief.blocks),
                        prior=s(belief.prior),
                        posterior=s(belief.posterior),
                        factorization=belief.factorization,
                        constraints=tuple(s(c) for c in belief.constraints),
                        episode_learnable=(
                            s(belief.episode_learnable)
                            if belief.episode_learnable is not None
                            else None
                        ),
                    )
                    for belief in template.agent.beliefs
                ),
            )
            if template.agent is not None
            else None
        ),
        schema_version=template.schema_version,
    )


def merge(a: Template, b: Template, name: str) -> Template:
    """Disjoint union; caller is responsible for prior scoping."""
    _require_flat(a)
    _require_flat(b)
    for kind in ("variables", "factors", "ports", "blobs", "learnables"):
        clash = set(getattr(a, kind)) & set(getattr(b, kind))
        if clash:
            raise ComposeError(f"{kind} name clash: {sorted(clash)} (scope the operands)")
    if a.agent is not None and b.agent is not None:
        raise ComposeError(
            "cannot merge two AgentSpec declarations; compose one complete "
            "agent model with fragments that do not declare a runtime interface"
        )
    return Template(
        name=name,
        variables={**a.variables, **b.variables},
        factors={**a.factors, **b.factors},
        constraints=[*a.constraints, *b.constraints],
        ports={**a.ports, **b.ports},
        blobs={**a.blobs, **b.blobs},
        learnables={**a.learnables, **b.learnables},
        agent=a.agent if a.agent is not None else b.agent,
    )


def bond(t: Template, port_a: str, port_b: str) -> Template:
    """Unify the edges behind two ports of the same template, in place-ish:
    returns the same object mutated (used inside compose/unroll pipelines)."""
    for name in (port_a, port_b):
        if name not in t.ports:
            raise ComposeError(f"no port {name!r}")
    if port_a == port_b:
        raise ComposeError(f"cannot bond port {port_a!r} to itself")
    pa, pb = t.ports[port_a], t.ports[port_b]
    va, vb = t.variables[pa.variable], t.variables[pb.variable]
    if va.domain_key != vb.domain_key:
        raise ComposeError(
            f"bond {port_a!r}~{port_b!r}: domain {va.domain_key} != {vb.domain_key}"
        )
    if va.name == vb.name:
        raise ComposeError(f"bond {port_a!r}~{port_b!r}: ports share variable {va.name!r}")

    # Keep a's variable; rewire every reference to b's variable onto it.
    for fname, f in t.factors.items():
        t.factors[fname] = FactorSpec(
            f.name,
            f.type,
            edges={r: va.name if v == vb.name else v for r, v in f.edges.items()},
            param_edges={r: va.name if v == vb.name else v for r, v in f.param_edges.items()},
            param_blobs=dict(f.param_blobs),
            function=f.function,
        )
    t.constraints = [
        dataclasses.replace(c, target=va.name) if c.target == vb.name else c
        for c in t.constraints
    ]
    del t.variables[vb.name]
    del t.ports[port_a]
    del t.ports[port_b]
    return t


def compose(
    a: Template,
    b: Template,
    bonds: dict[str, str],
    *,
    name: str,
    scope_a: str = "a",
    scope_b: str = "b",
) -> Template:
    """Scope both operands, merge, bond `a`-ports to `b`-ports, validate."""
    sa, sb = scoped(a, scope_a), scoped(b, scope_b)
    out = merge(sa, sb, name)
    for pa, pb in bonds.items():
        bond(out, scope_a + SCOPE_SEP + pa, scope_b + SCOPE_SEP + pb)
    validate(out)
    return out


def unroll(slice_t: Template, n: int, chain: dict[str, str], *, name: str | None = None) -> Template:
    """Chain `n` scoped copies of a temporal slice, bonding chain[tail]=head
    pairs between consecutive copies: t0, t1, ... t{n-1}.

    The result keeps t0's upstream glue ports and t{n-1}'s downstream ones
    (plus every non-chained port of every copy) — those are the rollout's own
    boundary, to be terminated by priors, goals, data, or further gluing.
    """
    if n < 1:
        raise ComposeError("unroll needs n >= 1")
    for tail, head in chain.items():
        for pname in (tail, head):
            if pname not in slice_t.ports:
                raise ComposeError(f"chain references unknown port {pname!r}")
            if slice_t.ports[pname].role is not PortRole.GLUE:
                raise ComposeError(f"chain port {pname!r} must have role glue")

    out = scoped(slice_t, "t0")
    for i in range(1, n):
        nxt = scoped(slice_t, f"t{i}")
        out = merge(out, nxt, name or f"{slice_t.name}[unrolled x{n}]")
        for tail, head in chain.items():
            bond(out, f"t{i - 1}{SCOPE_SEP}{tail}", f"t{i}{SCOPE_SEP}{head}")
    out.name = name or f"{slice_t.name}[unrolled x{n}]"
    validate(out)
    return out

"""Plate expansion: materialize a replicated fragment into a flat template.

`expand_plates` is a pure function run at the top of the compiler's grounding
phase (cff-cpxj phase 1): every plate's fragment is copied `dims` times with
scoped names, parameters tied by name (per-copy blobs excepted), and the copies
wired together by the plate's bonds. The wiring vocabulary:

- ``Chain``     — bond consecutive copies along an axis (a grid is one per axis);
                  boundary copies keep their unbonded chain ports exposed.
- ``Broadcast`` — tie every copy's port to one shared hub edge; because FFG
                  edges have max degree 2, this synthesizes an equality tree
                  (a chain of ternary equality factors).
- ``Expose``    — leftover ports stay exposed as ``plate/<i>/port`` (the
                  default in phase 1; the rule is the explicit marker).

Phase 1 is deliberately naive: it materializes Python objects. Phase 2 targets
index-table expansion into the group tensors without ever building the copies
(the arena was designed for it).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from cffg.core.compose import SCOPE_SEP, ComposeError, bond
from cffg.core.ir import (
    BlobRef,
    Broadcast,
    Chain,
    Expose,
    FactorSpec,
    LearnableSpec,
    PlateSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.core.validate import validate


# Portable bundles use this marker in a per-copy BlobRef path.  Expansion is
# the point where a compact plate spec acquires concrete indices, so it is also
# the only layer that can turn one declared path into one path per copy.
PLATE_INDEX_TOKEN = "{index}"


def expand_plates(t: Template) -> Template:
    """Materialize every plate in `t`, returning a flat, plate-free Template.

    The spec is validated first (fragments validate on their own; bonds
    reference real fragment ports; chain endpoints are GLUE ports). Templates
    with no plates are returned unchanged, so compiling a plateless template is
    byte-for-byte the old path.
    """
    if not t.plates:
        return t

    validate(t)  # spec validation (includes _check_plates)

    out = Template(
        name=t.name,
        variables=dict(t.variables),
        factors=dict(t.factors),
        constraints=list(t.constraints),
        ports=dict(t.ports),
        blobs=dict(t.blobs),
        learnables=dict(t.learnables),
        agent=t.agent,
        schema_version=t.schema_version,
    )
    for pname in sorted(t.plates):
        _expand_one(out, pname, t.plates[pname])
    return out


def _indices(dims: tuple[int, ...]) -> list[tuple[int, ...]]:
    return [tuple(int(i) for i in idx) for idx in np.ndindex(*dims)]


def _prefix(plate: str, idx: tuple[int, ...]) -> str:
    return plate + SCOPE_SEP + SCOPE_SEP.join(str(i) for i in idx)


def _scope_copy(
    frag: Template, prefix: str, index: tuple[int, ...], per_copy: frozenset[str]
) -> tuple[dict, dict, list, dict, dict]:
    """Scope one fragment copy under `prefix/`. Tied blob names stay unscoped
    (copies reference the same parent blob entry); per-copy blob names are
    scoped and given their own indexed parent entry."""
    p = prefix + SCOPE_SEP

    def s(name: str) -> str:
        return p + name

    def sb(bname: str) -> str:
        return s(bname) if bname in per_copy else bname

    variables = {
        s(v.name): dataclasses.replace(v, name=s(v.name)) for v in frag.variables.values()
    }
    factors = {
        s(f.name): FactorSpec(
            s(f.name),
            f.type,
            edges={r: s(v) for r, v in f.edges.items()},
            param_edges={r: s(v) for r, v in f.param_edges.items()},
            param_blobs={r: sb(b) for r, b in f.param_blobs.items()},
        )
        for f in frag.factors.values()
    }
    constraints = [dataclasses.replace(c, target=s(c.target)) for c in frag.constraints]
    ports = {
        s(pt.name): PortSpec(s(pt.name), s(pt.variable), pt.role) for pt in frag.ports.values()
    }
    per_copy_blobs = {
        s(b.name): BlobRef(
            s(b.name),
            b.path.replace(PLATE_INDEX_TOKEN, SCOPE_SEP.join(map(str, index))),
            b.init,
        )
        for b in frag.blobs.values()
        if b.name in per_copy
    }
    per_copy_learnables = {
        s(ln.name): LearnableSpec(
            s(ln.name),
            ln.family,
            tuple(sb(b) for b in ln.blobs),
            None if ln.sources is None else tuple(s(f) for f in ln.sources),
        )
        for ln in frag.learnables.values()
        if set(ln.blobs) <= per_copy
    }
    return variables, factors, constraints, ports, per_copy_blobs, per_copy_learnables


def _expand_one(out: Template, plate: str, spec: PlateSpec) -> None:
    frag = spec.fragment
    per_copy = frozenset(spec.per_copy_blobs)
    idxs = _indices(spec.dims)

    # Tied blobs: one shared parent entry per fragment blob (name unscoped).
    for b in frag.blobs.values():
        if b.name not in per_copy:
            out.blobs[b.name] = BlobRef(b.name, b.path, b.init)

    # A learnable over tied blobs stays ONE posterior for the whole plate:
    # every copy's factor instance reads the same blob, so every copy's
    # sufficient statistics accumulate into it. That is the transfer property
    # plates exist for (cff-x0sv). Per-copy blobs scope, so their learnables
    # become one posterior per copy, emitted alongside the copy below.
    tied_learnables = [
        ln for ln in frag.learnables.values() if not (set(ln.blobs) & per_copy)
    ]
    for ln in tied_learnables:
        if ln.sources:
            raise ComposeError(
                f"plate {plate!r}: tied learnable {ln.name!r} restricts sources to "
                f"{sorted(ln.sources)}, which names one copy's factor and not the "
                "others; a shared posterior accumulates over every copy"
            )
        out.learnables[ln.name] = ln
    split = [
        ln.name
        for ln in frag.learnables.values()
        if (set(ln.blobs) & per_copy) and not set(ln.blobs) <= per_copy
    ]
    if split:
        raise ComposeError(
            f"plate {plate!r}: learnables {sorted(split)} mix tied and per-copy "
            "blobs; a posterior is either shared across copies or private to one"
        )

    # Materialize every copy (disjoint union into `out`; scoping makes names
    # disjoint, so a clash means the plate name collides with existing content).
    for idx in idxs:
        prefix = _prefix(plate, idx)
        variables, factors, constraints, ports, pblobs, plearn = _scope_copy(
            frag, prefix, idx, per_copy
        )
        for kind, new in (
            ("variables", variables),
            ("factors", factors),
            ("ports", ports),
            ("blobs", pblobs),
            ("learnables", plearn),
        ):
            clash = set(getattr(out, kind)) & set(new)
            if clash:
                raise ComposeError(f"plate {plate!r}: {kind} name clash {sorted(clash)}")
            getattr(out, kind).update(new)
        out.constraints.extend(constraints)

    # Apply wiring rules.
    for rule in spec.bonds:
        if isinstance(rule, Chain):
            _apply_chain(out, plate, spec, rule)
        elif isinstance(rule, Broadcast):
            _apply_broadcast(out, plate, spec, rule)
        elif isinstance(rule, Expose):
            pass  # leftover ports are already exposed by scoping (phase 1)
        else:  # pragma: no cover - closed vocabulary
            raise TypeError(f"unknown wiring rule {rule!r}")


def _apply_chain(out: Template, plate: str, spec: PlateSpec, rule: Chain) -> None:
    """Bond copy i's port_a to copy (i+offset)'s port_b along `rule.axis`."""
    dims = spec.dims
    axis = rule.axis
    for idx in _indices(dims):
        nxt = list(idx)
        nxt[axis] += rule.offset
        if nxt[axis] >= dims[axis]:
            continue  # boundary copy: its chain port stays exposed
        pa = _prefix(plate, idx) + SCOPE_SEP + rule.port_a
        pb = _prefix(plate, tuple(nxt)) + SCOPE_SEP + rule.port_b
        bond(out, pa, pb)


def _apply_broadcast(out: Template, plate: str, spec: PlateSpec, rule: Broadcast) -> None:
    """Tie every copy's `rule.port` to one shared hub edge via an equality
    tree. N copies + hub = N+1 edges, connected by N-1 ternary equality
    factors (N==1: hub IS the single branch; N==2: one equality)."""
    idxs = _indices(spec.dims)
    # Branch variable + port name for each copy, in row-major order.
    branch_vars: list[str] = []
    for idx in idxs:
        port_name = _prefix(plate, idx) + SCOPE_SEP + rule.port
        port = out.ports[port_name]
        branch_vars.append(port.variable)
        del out.ports[port_name]  # branch becomes interior (degree 2)

    proto = out.variables[branch_vars[0]]
    hub = f"{plate}{SCOPE_SEP}{rule.hub}"
    hub_port = f"{plate}{SCOPE_SEP}{rule.hub}"

    n = len(branch_vars)
    if n == 1:
        # No equality needed: the single branch IS the hub edge. Rename it.
        _rename_variable(out, branch_vars[0], hub, proto)
    else:
        out.variables[hub] = dataclasses.replace(proto, name=hub)
        base = f"{plate}{SCOPE_SEP}_bcast{SCOPE_SEP}{rule.hub}"
        prev = hub  # the "a" edge feeding the next equality
        for k in range(n - 1):
            b_edge = branch_vars[k]
            last = k == n - 2
            if last:
                c_edge = branch_vars[k + 1]
            else:
                c_edge = f"{base}{SCOPE_SEP}link{k}"
                out.variables[c_edge] = dataclasses.replace(proto, name=c_edge)
            eq = f"{base}{SCOPE_SEP}eq{k}"
            out.factors[eq] = FactorSpec(
                eq, "equality", edges={"a": prev, "b": b_edge, "c": c_edge}
            )
            prev = c_edge

    out.ports[hub_port] = PortSpec(hub_port, hub, PortRole.GLUE)


def _rename_variable(out: Template, old: str, new: str, proto: VariableSpec) -> None:
    out.variables.pop(old, None)
    out.variables[new] = dataclasses.replace(proto, name=new)
    for fname, f in out.factors.items():
        out.factors[fname] = FactorSpec(
            f.name,
            f.type,
            edges={r: new if v == old else v for r, v in f.edges.items()},
            param_edges={r: new if v == old else v for r, v in f.param_edges.items()},
            param_blobs=dict(f.param_blobs),
        )
    out.constraints = [
        dataclasses.replace(c, target=new) if c.target == old else c for c in out.constraints
    ]


def plate_observations(
    plate_name: str,
    array_like,
    port_name: str,
    *,
    dims: tuple[int, ...] | None = None,
) -> dict:
    """Fan a whole tensor out to a plate's exposed DATA ports in one call.

    Maps element (i,) -> ``plate/<i>/port`` (1-D) or (i, j) -> ``plate/<i>/<j>/
    port`` (2-D), so a full image binds through the existing `Engine.bind_data`.
    `dims` overrides the grid shape (e.g. when each cell is itself a vector);
    by default the array's shape is the grid. Scalar cells are returned as
    python ints (discrete clamps); vector cells pass through as arrays."""
    arr = np.asarray(array_like)
    grid = arr.shape if dims is None else tuple(dims)
    out: dict = {}
    for idx in np.ndindex(*grid):
        key = plate_name + SCOPE_SEP + SCOPE_SEP.join(str(i) for i in idx) + SCOPE_SEP + port_name
        val = arr[idx]
        out[key] = int(val) if np.ndim(val) == 0 else np.asarray(val)
    return out

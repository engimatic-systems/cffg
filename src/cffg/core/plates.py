"""Plate expansion: materialize a replicated fragment into a flat template.

`expand_plates` returns an editable flat model. Every plate's fragment is copied
once per grid cell with
scoped names, parameters tied by name (per-copy blobs excepted), and the copies
wired together by the plate's bonds. The wiring vocabulary:

- ``Chain``     — bond consecutive copies along an axis (a grid is one per axis);
                  boundary copies keep their unbonded chain ports exposed.
- ``Broadcast`` — tie every copy's port to one shared hub edge; because FFG
                  edges have max degree 2, this synthesizes an equality tree
                  (a balanced tree of ternary equality factors).
- ``Expose``    — leftover ports stay exposed as ``plate/<i>/port`` (the
                  default in phase 1; the rule is the explicit marker).

The compiler constructs slot addresses directly for compact plates. This
public expansion also provides a reference for checking those addresses.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from cffg.core.compose import SCOPE_SEP, ComposeError, bond
from cffg.core.equality_tree import plan_equality_tree
from cffg.core.plate_grid import PlateGrid
from cffg.core.ir import (
    EDGE_CONSTRAINTS,
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

    validate(t)
    return _expand(t, 0)


def _prepared_plates(template, prepare, depth):
    """Bind child fragments before copying them into this lexical scope.

    An owner introduced by a child plate belongs to the enclosing copy.
    A blob also declared directly by the enclosing fragment follows that
    fragment's own tying rule.
    """
    for name, spec in sorted(template.plates.items()):
        fragment = spec.fragment
        private = frozenset(spec.per_copy_blobs)
        if fragment.plates:
            child = prepare(fragment, depth + 1)
            private = private.union(child.blobs.keys() - fragment.blobs.keys())
            fragment = child
        yield name, spec, fragment, private


def _owner_template(template, plates):
    out = dataclasses.replace(template, blobs=dict(template.blobs), learnables=dict(template.learnables))
    for name, spec, fragment, private in plates:
        _install_shared(out, name, spec, fragment, private)
    return out


def _expand(template, depth):
    plates = list(_prepared_plates(template, _expand, depth))
    out = dataclasses.replace(
        _owner_template(template, plates),
        variables=dict(template.variables),
        factors=dict(template.factors),
        constraints=list(template.constraints),
        ports=dict(template.ports),
        plates={},
    )
    for name, spec, fragment, private in plates:
        _expand_one(out, name, spec, fragment, private, depth)
    return out


def _with_shared_owners(template, depth=0):
    """Collect externally tied owners during composition without allocating copies."""
    return _owner_template(template, _prepared_plates(template, _with_shared_owners, depth))


def _prefix(plate: str, idx: tuple[int, ...]) -> str:
    return plate + SCOPE_SEP + SCOPE_SEP.join(str(i) for i in idx)


def _scoped(prefix: str, name: str) -> str:
    return prefix + SCOPE_SEP + name


def _scoped_blob_refs(fragment, prefix, index, per_copy, depth):
    coordinate = SCOPE_SEP.join(map(str, index))
    # A plain marker denotes the whole numeric coordinate. Binding proceeds
    # inside out, so leave its enclosing part for the next level.
    full_index = f"{PLATE_INDEX_TOKEN}/{coordinate}" if depth else coordinate
    level_token = "{index:" + str(depth) + "}"
    return {
        _scoped(prefix, b.name): BlobRef(
            _scoped(prefix, b.name),
            b.path.replace(level_token, coordinate).replace(PLATE_INDEX_TOKEN, full_index),
            b.init,
        )
        for b in fragment.blobs.values()
        if b.name in per_copy
    }


def _numeric_sources(template, depth):
    blobs = dict(template.blobs)
    for name, spec, fragment, private in _prepared_plates(template, _numeric_sources, depth):
        _install_shared_blobs(blobs, name, spec, fragment, private)
        if private:
            for index in PlateGrid(spec.dims):
                blobs.update(_scoped_blob_refs(fragment, _prefix(name, index), index, private, depth))
    return dataclasses.replace(template, blobs=blobs)


def expanded_blob_refs(template):
    """Resolve current paths and initializers on a structurally validated cache hit."""
    return _numeric_sources(template, 0).blobs


def _scope_copy(spec, frag, prefix, index, per_copy, depth):
    """Scope copy-local declarations under the index and tied blobs under their owner."""
    def s(name: str) -> str:
        return _scoped(prefix, name)

    def sb(bname: str) -> str:
        return s(bname) if bname in per_copy else spec.shared_name(bname)

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
            function=f.function,
        )
        for f in frag.factors.values()
    }
    constraints = [dataclasses.replace(c, target=s(c.target)) for c in frag.constraints]
    ports = {
        s(pt.name): PortSpec(s(pt.name), s(pt.variable), pt.role) for pt in frag.ports.values()
    }
    per_copy_blobs = _scoped_blob_refs(frag, prefix, index, per_copy, depth)
    per_copy_learnables = _scoped_learnables(frag, prefix, per_copy)
    return variables, factors, constraints, ports, per_copy_blobs, per_copy_learnables


def _scoped_learnables(fragment, prefix, per_copy):
    return {
        _scoped(prefix, ln.name): LearnableSpec(
            _scoped(prefix, ln.name),
            ln.family,
            tuple(_scoped(prefix, b) for b in ln.blobs),
            None if ln.sources is None else tuple(_scoped(prefix, f) for f in ln.sources),
            ln.semantic_identity,
        )
        for ln in fragment.learnables.values()
        if set(ln.blobs) <= per_copy
    }


def _install_shared_blobs(blobs, plate, spec, frag, per_copy):
    for b in frag.blobs.values():
        if b.name not in per_copy:
            name = spec.shared_name(b.name)
            ref = dataclasses.replace(b, name=name)
            if name in blobs and blobs[name] != ref:
                raise ComposeError(f"plate {plate!r}: conflicting declarations for tied blob {name!r}")
            blobs[name] = ref


def _install_shared(out, plate, spec, frag, per_copy):
    _install_shared_blobs(out.blobs, plate, spec, frag, per_copy)

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
        name = spec.shared_name(ln.name)
        learnable = dataclasses.replace(
            ln, name=name, blobs=tuple(spec.shared_name(blob) for blob in ln.blobs)
        )
        if name in out.learnables and out.learnables[name] != learnable:
            raise ComposeError(f"plate {plate!r}: conflicting declarations for tied learnable {name!r}")
        out.learnables[name] = learnable
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


def _expand_one(out, plate, spec, fragment, per_copy, depth):
    for idx in PlateGrid(spec.dims):
        prefix = _prefix(plate, idx)
        variables, factors, constraints, ports, pblobs, plearn = _scope_copy(
            spec, fragment, prefix, idx, per_copy, depth
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
    for idx, nxt in PlateGrid(spec.dims).chain(rule.axis, rule.offset):
        pa = _prefix(plate, idx) + SCOPE_SEP + rule.port_a
        pb = _prefix(plate, nxt) + SCOPE_SEP + rule.port_b
        bond(out, pa, pb)


def _apply_broadcast(out: Template, plate: str, spec: PlateSpec, rule: Broadcast) -> None:
    """Tie every copy's `rule.port` to one shared hub edge via an equality
    tree. N copies + hub = N+1 edges, connected by N-1 ternary equality
    factors (N==1: hub IS the single branch; N==2: one equality)."""
    hub = f"{plate}/{rule.hub}"
    ports = [f"{_prefix(plate, i)}/{rule.port}" for i in PlateGrid(spec.dims)]
    branches = [out.ports[name].variable for name in ports]
    if hub in out.variables or hub in out.ports:
        raise ComposeError(f"plate {plate!r}: generated Broadcast hub {hub!r} already exists")
    tree = plan_equality_tree(hub, branches, f"{plate}/_bcast/{rule.hub}")
    prototype = out.variables[branches[0]]
    tree.install(out.variables, out.factors, prototype)
    for name in ports:
        del out.ports[name]
    if len(branches) == 1:
        _rename_variable(out, branches[0], hub, prototype)
    else:
        out.variables[hub] = dataclasses.replace(prototype, name=hub)
    out.ports[hub] = PortSpec(hub, hub, PortRole.GLUE)


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
            function=f.function,
        )
    out.constraints = [
        dataclasses.replace(c, target=new) if c.kind in EDGE_CONSTRAINTS and c.target == old else c for c in out.constraints
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

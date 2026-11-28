"""Static Forney incidence, source declarations and directed passes for interval forests."""

from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

import jax
import numpy as np

from cffg._immutable import ImmutableRecord
from cffg.compiler_errors import CompileError
from cffg.core.ir import FACTOR_TYPES, PortRole
from cffg.core.serialize import structure_hash
from cffg.resolution.sources import _load_blob


@dataclass(frozen=True)
class IntervalEdge:
    name: str
    dimension: int
    capacity: int
    slots: tuple[int, int]


@dataclass(frozen=True)
class IntervalNode:
    name: str
    kind: str
    own: tuple[int, ...]
    incoming: tuple[int, ...]
    blobs: tuple[str, ...]


@dataclass(frozen=True)
class IntervalForest(ImmutableRecord):
    fingerprint: str
    edges: dict
    nodes: tuple
    ports: dict
    boundaries: dict
    chart: dict
    order: tuple
    components: tuple
    # Maximum number of simultaneous sweeps needed from arbitrary initial
    # internal messages, given fixed source potentials and boundary messages.
    depth: int
    log_params: dict
    learn_plans: tuple = ()
    mu3_groups: tuple = ()
    semiring: str = "sum"


def _parameters(template, shapes, arrays, root):
    result = {}
    for name, shape in shapes.items():
        decl = template.blobs[name]
        path = Path(root) / decl.path if root is not None else None
        if arrays is not None and name in arrays:
            value = arrays[name]
        elif path is not None and path.is_file():
            value = _load_blob(path, name)
        elif decl.init:
            from cffg.inits import from_spec

            value = from_spec(decl.init, shape)
        else:
            raise CompileError(
                f"interval blob {name!r} has no array, file or initializer"
            )
        a = np.asarray(value)
        if a.shape != shape or a.dtype.kind not in "biuf" or not np.isfinite(a).all():
            raise CompileError(
                f"interval blob {name!r} must be finite with shape {shape}"
            )
        result[name] = a
    return result


def host_cells(lower, upper, height, count, *, dtype):
    """Validate source and working geometry; reject rounding away a supported cell or gap."""
    arrays = tuple(np.asarray(x) for x in (lower, upper, height))
    count = np.asarray(count)
    if not all(a.dtype in (np.float32, np.float64) for a in arrays):
        raise ValueError("interval endpoints and log heights must be floating point")
    low, high, h = arrays
    if low.ndim != 2 or high.shape != low.shape or h.shape != low.shape:
        raise ValueError(
            "interval endpoint and log-height arrays must have identical 2D shapes"
        )
    if (
        count.dtype.kind not in "iu"
        or count.shape != low.shape[:1]
        or np.any(count < 0)
        or np.any(count > low.shape[1])
    ):
        raise ValueError("invalid interval cell counts")
    live = np.arange(low.shape[1]) < count[:, None]
    for l, u, heights in (arrays, tuple(a.astype(dtype) for a in arrays)):
        if (
            np.any(live & ~(l < u))
            or np.any(live & ~np.isfinite(heights))
            or np.any(live[:, 1:] & (l[:, 1:] < u[:, :-1]))
        ):
            raise ValueError("invalid or unresolved interval boundaries or heights")
        if (
            np.any(live & (np.isfinite(low) != np.isfinite(l)))
            or np.any(live & (np.isfinite(high) != np.isfinite(u)))
            or np.any(
                live[:, 1:] & ((low[:, 1:] > high[:, :-1]) != (l[:, 1:] > u[:, :-1]))
            )
        ):
            raise ValueError(
                "interval boundaries or gaps unresolved in working precision"
            )
    return (
        *(np.where(live, a, 0).astype(dtype) for a in arrays),
        count.astype(np.int32),
    )


def source_message(params, blobs, dtype):
    lo, hi, h, count, tails = (np.asarray(params[k]) for k in blobs)
    if any(v.dtype not in (np.float32, np.float64) for v in (lo, hi, h)):
        raise ValueError("interval endpoints and log heights must be floats")
    if (
        tails.dtype != np.bool_
        or count.dtype.kind not in "iu"
        or np.any(count < 0)
        or np.any(count > lo.shape[1])
        or np.any(tails & (count[:, None] == 0))
    ):
        raise ValueError("invalid interval tail flags or counts")
    # Keep the bundle's finite-numeric-blob contract. Only explicitly flagged
    # outer cells acquire an infinite endpoint in the executable message.
    index = np.arange(lo.shape[1])
    left = tails[:, 0, None] & (index == 0)
    right = tails[:, 1, None] & (index == count[:, None] - 1)
    if np.any(left & (lo != 0)) or np.any(right & (hi != 0)):
        raise ValueError("unbounded interval endpoint placeholders must be zero")
    return host_cells(
        np.where(left, -np.inf, lo), np.where(right, np.inf, hi), h, count, dtype=dtype
    )


def check_boundaries(values, dtype):
    for row in values:
        unique = np.unique(row[np.isfinite(row)])
        cast = unique.astype(dtype)
        if np.any(cast[1:] <= cast[:-1]) or not np.isfinite(cast).all():
            raise ValueError(
                "distinct interval boundaries unresolved in working precision"
            )


def compile_interval_forest(template, arrays, root, semiring):
    if (
        semiring != "sum"
        or template.constraints
        or template.learnables
        or template.agent is not None
        or any(v.family != "interval" for v in template.variables.values())
    ):
        raise CompileError(
            "interval forests require interval-only sum-product without constraints or learning"
        )
    if any(p.role != PortRole.DATA for p in template.ports.values()):
        raise CompileError("interval forest exterior ports must be DATA likelihoods")
    incidences = {name: [] for name in sorted(template.variables)}
    shapes = {}
    for name, f in sorted(template.factors.items()):
        if (
            f.type not in ("interval_activation", "interval_potential", "equality")
            or f.param_edges
        ):
            raise CompileError(
                f"interval forest cannot execute factor {name!r} of type {f.type!r}"
            )
        roles = FACTOR_TYPES[f.type].edge_roles
        for role in roles:
            incidences[f.edges[role]].append((name, role))
        var = template.variables[f.edges[roles[0]]]
        dim, cap = var.interval_dim, var.interval_capacity
        for role, blob in f.param_blobs.items():
            shape = (dim,)
            if role in ("lower", "upper", "log_height"):
                shape = (dim, cap)
            elif role == "tails":
                shape = (dim, 2)
            if shapes.setdefault(blob, shape) != shape:
                raise CompileError(
                    f"incompatible shapes for shared interval blob {blob!r}"
                )
    params = _parameters(template, shapes, arrays, root)
    edges, owners = {}, {}
    for index, (name, ends) in enumerate(incidences.items()):
        var = template.variables[name]
        slots = (2 * index, 2 * index + 1)
        edges[name] = IntervalEdge(name, var.interval_dim, var.interval_capacity, slots)
        owners.update(zip(ends, slots[: len(ends)], strict=True))
    ports = {p.name: p.variable for p in template.ports.values()}
    ordered_nodes = []
    dependencies = {}
    outputs = {}
    for name, f in sorted(template.factors.items()):
        ft = FACTOR_TYPES[f.type]
        own = tuple(owners[name, role] for role in ft.edge_roles)
        incoming = tuple(i ^ 1 for i in own)
        blobs = tuple(f.param_blobs[role] for role in ft.param_roles)
        node = IntervalNode(name, f.type, own, incoming, blobs)
        try:
            if f.type == "interval_activation":
                for role, blob in f.param_blobs.items():
                    a = params[blob]
                    if a.dtype not in (np.float32, np.float64):
                        raise ValueError(
                            "activation vectors must be float32 or float64"
                        )
                    narrowed = a.astype(jax.dtypes.canonicalize_dtype(a.dtype))
                    if not np.isfinite(narrowed).all() or (
                        role in ("left", "right") and np.any(narrowed <= 0)
                    ):
                        raise ValueError(
                            "nonfinite vector or nonpositive slope in working precision"
                        )
            elif f.type == "interval_potential":
                dtype = jax.dtypes.canonicalize_dtype(
                    np.result_type(*(params[k] for k in blobs[:3]))
                )
                source_message(params, blobs, dtype)
        except ValueError as exc:
            raise CompileError(f"interval factor {name!r}: {exc}") from exc
        for i, slot in enumerate(own):
            dependencies[slot] = [inc for j, inc in enumerate(incoming) if i != j]
            outputs[slot] = (len(ordered_nodes), i)
        ordered_nodes.append(node)
    try:
        slots = tuple(TopologicalSorter(dependencies).static_order())
    except CycleError as exc:
        raise CompileError("interval sum-product requires an acyclic forest") from exc
    levels = {}
    for slot in slots:
        levels[slot] = (
            1 + max((levels[s] for s in dependencies[slot]), default=0)
            if slot in outputs
            else 0
        )
    order = tuple(outputs[s] for s in slots if s in outputs)
    # Connected components determine the independent log normalizers. Every
    # edge in a completed tree integrates the same unnormalized joint mass.
    adjacency = [set() for _ in edges]
    for node in ordered_nodes:
        first = node.own[0] // 2
        for slot in node.own[1:]:
            other = slot // 2
            adjacency[first].add(other)
            adjacency[other].add(first)
    components = []
    remaining = set(range(len(edges)))
    while remaining:
        root_edge = min(remaining)
        members, todo = {root_edge}, [root_edge]
        while todo:
            unseen = adjacency[todo.pop()] - members
            members.update(unseen)
            todo.extend(unseen)
        remaining -= members
        components.append(tuple(sorted(members)))
    # Boundaries expressed in the same chart must retain their ordering
    # during source -> working-precision conversion. Include equality
    # copies and their activation knots, even for a one-cell prior.
    charts = {e: [e] for e in edges}
    points = {e: [] for e in edges}
    names_by_index = tuple(edges)
    for node in ordered_nodes:
        names = [names_by_index[slot // 2] for slot in node.own]
        if node.kind == "interval_activation":
            for name, blob in zip(names, node.blobs[:2], strict=True):
                points[name].append(params[blob][:, None])
        elif node.kind == "interval_potential":
            l, u, _, n = source_message(params, node.blobs, np.float64)
            live = np.arange(l.shape[1]) < n[:, None]
            points[names[0]].extend(np.where(live, v, np.nan) for v in (l, u))
        else:
            group = sorted({e for name in names for e in charts[name]})
            for e in group:
                charts[e] = group
    boundaries = {}
    floats = [a.dtype for a in params.values() if a.dtype.kind == "f"]
    dtype = (
        jax.dtypes.canonicalize_dtype(np.result_type(*floats)) if floats else np.float32
    )
    for e, group in charts.items():
        if e in boundaries:
            continue
        known = [v for name in group for v in points[name]]
        values = (
            np.concatenate(known, axis=1)
            if known
            else np.empty((edges[e].dimension, 0))
        )
        try:
            check_boundaries(values, dtype)
        except ValueError as exc:
            raise CompileError(f"interval chart {e!r}: {exc}") from exc
        boundaries.update(dict.fromkeys(group, values))
    return IntervalForest(
        structure_hash(template),
        edges,
        tuple(ordered_nodes),
        ports,
        boundaries,
        {e: min(group) for e, group in charts.items()},
        order,
        tuple(components),
        max(levels.values()),
        params,
    )

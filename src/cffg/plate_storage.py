"""Predict persistent State storage directly from a compact plate source."""

from __future__ import annotations

from collections import Counter
from math import prod
from numbers import Integral

import numpy as np

from cffg import kernels
from cffg.core.ir import Broadcast, Chain, ConstraintKind, PortRole
from cffg.core.plate_grid import PlateGrid
from cffg.core.plates import _with_shared_owners
from cffg.core.validate import _copy_port, validate
from cffg.resource_analysis import STATE_ARRAYS


_FIXED_ROW_TYPES = frozenset({
    "categorical", "matrix_cat", "transition_mixture", "gaussian",
    "gaussian_linear", "gamma", "beta", "poisson_obs", "bernoulli_obs",
})


class UnknownStorageLayout(ValueError):
    """The registered compiler path has no preallocation byte formula."""


def _checked(value, label):
    if value < 0 or value > (1 << 63) - 1:
        raise OverflowError(f"{label} exceeds the signed 64-bit storage range")
    return value


def _dimension(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"expected a positive integer dimension, got {value!r}")
    return int(value)


def _widths(variable):
    if variable.ef:
        return {"ef_eta": 4}  # two slots, two natural parameters each
    if variable.dim:
        dim = _dimension(variable.dim)
        return {"g_eta": 2 * dim, "g_lam": 2 * dim * dim}
    return {"messages": 2 * _dimension(variable.cardinality)}


def _port_origin(fragment, path):
    """Find the domain and leaf declaration behind one validated boundary path."""
    port = fragment.ports.get(path)
    if port is not None:
        return port.role, fragment.variables[port.variable], (path,)
    for name, spec in fragment.plates.items():
        prefix = name + "/"
        if not path.startswith(prefix):
            continue
        suffix = path[len(prefix):]
        grid = PlateGrid(spec.dims)
        role, _ = _copy_port(spec, grid, suffix)
        if role is None:
            continue
        for bond in spec.bonds:
            if isinstance(bond, Broadcast) and suffix == bond.hub:
                _, variable, _ = _port_origin(spec.fragment, bond.port)
                return PortRole.GLUE, variable, ()
        child_path = suffix.split("/", len(grid.dims))[-1]
        _, variable, leaf = _port_origin(spec.fragment, child_path)
        return role, variable, (name, *leaf)
    raise ValueError(f"unknown boundary port {path!r}")


def _wiring(spec, grid, parents, path, arenas, topology, consumed):
    """Count repeated Chain endpoint sets and balanced ternary trees."""
    used = {}
    hubs = set()
    fragment = spec.fragment

    def consume(name, box):
        for previous in used.setdefault(name, []):
            if all(max(a, c) < min(b, d) for (a, b), (c, d) in zip(previous, box)):
                raise ValueError(f"wiring rules consume port {name!r} in the same copy")
        used[name].append(box)

    for rule in spec.bonds:
        box = [(0, size) for size in grid.dims]
        if isinstance(rule, Chain):
            axis, offset = int(rule.axis), int(rule.offset)
            length = grid.dims[axis]
            if offset >= length:
                continue
            _, a, _ = _port_origin(fragment, rule.port_a)
            _, b, _ = _port_origin(fragment, rule.port_b)
            if a.domain_key != b.domain_key:
                raise ValueError(f"Chain ports must have equal domains: {a.name}, {b.name}")
            box[axis] = (0, length - offset)
            consume(rule.port_a, box.copy())
            box[axis] = (offset, length)
            consume(rule.port_b, box)
            pairs = parents * (grid.count // length) * (length - offset)
            topology["chain_bonds"] += pairs
            topology["ports"] -= 2 * pairs
            added_edges = -pairs
            variable = b
        elif isinstance(rule, Broadcast):
            consume(rule.port, box)
            if rule.hub in hubs:
                raise ValueError(f"duplicate Broadcast hub {rule.hub!r}")
            hubs.add(rule.hub)
            role, variable, leaf = _port_origin(fragment, rule.port)
            if role is not PortRole.GLUE:
                consumed[path + leaf] += parents * grid.count
            # N=1 renames the branch. Otherwise add N-2 internal edges plus the hub.
            added_edges = parents * (grid.count - 1)
            topology["broadcast_equalities"] += added_edges
            topology["factors"] += added_edges
            topology["ports"] -= added_edges
        else:
            continue
        topology["edges"] += added_edges
        for name, width in _widths(variable).items():
            arenas[name] += width * added_edges


def _factor_rows(factor, fragment, n, data, mean_field, arenas, leaves):
    """Count physical packed rows and yield (source owner, shape) for logical storage."""
    variables = {r: fragment.variables[v] for r, v in factor.edges.items()}
    if factor.type == "equality":
        if any(v.dim and data.get(v.name, 0) for v in variables.values()):
            raise UnknownStorageLayout("Gaussian DATA taps on equalities require compilation")
        return
    kernel = kernels.kernel_for(factor.type)
    if (factor.type not in _FIXED_ROW_TYPES or kernel is None
            or kernel is not kernels._builtin_kernels.get(factor.type)
            or factor.param_edges or factor.function):
        raise UnknownStorageLayout(f"packed storage for {factor.type!r} requires compilation")
    if kernel.arena == "ef":
        for name in factor.param_blobs.values():
            yield name, ()
        for key in kernel.param_keys:
            leaves[f"ef:{factor.type}:{kernel.family}:{key}"] += 2 * n
        if kernel.observed_out:
            if data.get(factor.edges["out"], 0) != n:
                raise ValueError(f"observation factor {factor.name!r} needs a DATA output")
            arenas["ef_mask"] += n
            arenas["ef_obs"] += n
        return
    binding = kernel.binding
    sizes = {r: _dimension(v.dim or v.cardinality) for r, v in variables.items()}
    shape_id = "x".join(str(sizes[role]) for role in binding.roles)
    if kernel.arena == "discrete":
        mode = "vmp" if factor.name in mean_field else "bp"
        groups = {f"{mode}:{factor.type}:{shape_id}": n}
    else:
        conditioned = 0
        groups = {}
        for role, variable in variables.items():
            rows = data.get(variable.name, 0)
            if rows:
                if binding.conditioning is None or role != binding.conditioning.role:
                    raise ValueError(f"unsupported Gaussian DATA on {factor.name}/{role}")
                conditioned = rows
                kind = binding.conditioning.plan_type
                groups[f"gauss:{kind}:{shape_id}"] = rows
                arenas["g_mask"] += rows
                arenas["g_obs"] += rows * sizes[role]
        if n > conditioned:
            groups[f"gauss:{factor.type}:{shape_id}"] = n - conditioned
    for role, param in binding.parameters.items():
        shape = tuple(_dimension(s) for s in param.shape(sizes))
        yield factor.param_blobs[role], shape
        for group, rows in groups.items():
            paths = kernels.parameter_paths(kernel, group)
            leaves[paths[param.key]] += rows * prod(shape)


def _source_owner(name, scopes):
    """Lift a logical owner through its enclosing lexical scopes, inside out."""
    copies = 1
    private = False
    for plate, spec, grid in reversed(scopes):
        if name in spec.per_copy_blobs or name not in spec.fragment.blobs:
            private = True
            copies *= grid.count
            name = f"{plate}/{{index}}/{name}"
        else:
            name = spec.shared_name(name)
    return name, private, copies


def estimate_plate_storage(template, *, dtype=None, source_dtype=np.float64, batch_size=1):
    """Count array payload bytes without expanding plates or reading any parameters.

    `dtype` selects the model's f32/f64 execution precision (default: current
    JAX setting). Logical blob sizes use the declared common `source_dtype`;
    these are in-memory numerical payloads, excluding file headers. Batch size
    multiplies every initialized State leaf but does not duplicate source blobs.
    Native discrete table factors, affine Gaussians, conjugate scalar EF and
    equality use fixed packing. Other layouts raise UnknownStorageLayout.
    """
    import jax

    if dtype is None:
        dtype = np.float64 if jax.config.x64_enabled else np.float32
    dtype, source_dtype = np.dtype(dtype), np.dtype(source_dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("execution dtype must be float32 or float64")
    if source_dtype.kind not in "fiu":
        raise ValueError("logical source dtype must be a real numeric type")
    batch_size = _checked(_dimension(batch_size), "batch size")
    validate(template)
    _with_shared_owners(template)
    arenas = Counter({name: 0 for name in STATE_ARRAYS})
    leaves = Counter()
    blobs = {}
    topology = dict(edges=0, factors=0, ports=0, chain_bonds=0, broadcast_equalities=0)

    consumed = Counter()

    def fragment_counts(fragment, n, scopes, path):
        topology["edges"] += n * len(fragment.variables)
        topology["factors"] += n * len(fragment.factors)
        topology["ports"] += n * len(fragment.ports)
        for variable in fragment.variables.values():
            for name, width in _widths(variable).items():
                arenas[name] += n * width
        data = {}
        for name, port in fragment.ports.items():
            remaining = n - consumed[path + (name,)]
            if not 0 <= remaining <= n:
                raise ValueError(f"overlapping plate bonds on {name!r}")
            if port.role is PortRole.DATA:
                data[port.variable] = remaining
            if port.role in (PortRole.PRIOR, PortRole.GOAL) and remaining:
                raise ValueError(f"unterminated {port.role.value} port {port.name!r}")
        if any(c.kind not in (ConstraintKind.MEAN_FIELD, ConstraintKind.DELTA)
               for c in fragment.constraints):
            raise UnknownStorageLayout("specialized constraint storage requires compilation")
        mean_field = {c.target for c in fragment.constraints if c.kind is ConstraintKind.MEAN_FIELD}

        for factor in fragment.factors.values():
            for name, shape in _factor_rows(factor, fragment, n, data, mean_field, arenas, leaves):
                owner, private, copies = _source_owner(name, scopes)
                entry = {
                    "shape": shape, "copies": copies, "per_copy": private,
                    "bytes": _checked(prod(shape) * copies * source_dtype.itemsize, owner),
                }
                if owner in blobs and blobs[owner] != entry:
                    raise ValueError(f"source owner {owner!r} has incompatible shapes or multiplicity")
                blobs[owner] = entry

        for name, spec in fragment.plates.items():
            grid = PlateGrid(spec.dims)
            child_path = (*path, name)
            _wiring(spec, grid, n, child_path, arenas, topology, consumed)
            child_scopes = (*scopes, (name, spec, grid))
            fragment_counts(spec.fragment, n * grid.count, child_scopes, child_path)

    fragment_counts(template, 1, (), ())
    for name, number in topology.items():
        _checked(number, name)
    multiplier = dtype.itemsize * batch_size
    arrays = {name: _checked(count * multiplier, name) for name, count in arenas.items()}
    parameters = {f"[{name!r}]": _checked(count * multiplier, name)
                  for name, count in leaves.items()}
    source_bytes = _checked(sum(blob["bytes"] for blob in blobs.values()), "source bytes")
    parameter_bytes = _checked(sum(parameters.values()), "parameter bytes")
    return {
        "dtype": dtype.name, "source_dtype": source_dtype.name, "batch_size": batch_size,
        "arenas": arrays, "parameter_leaves": parameters, "parameter_bytes": parameter_bytes,
        "state_bytes": _checked(sum(arrays.values()) + parameter_bytes, "state bytes"),
        "source_blobs": blobs, "source_bytes": source_bytes,
        "topology": topology,
    }

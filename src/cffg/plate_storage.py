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
from cffg.core.validate import validate
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


def _wiring(spec, grid, edges, topology):
    """Count rectangular Chain endpoint sets and balanced ternary trees."""
    used = {}
    hubs = set()
    removed = set()
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
            a, b = (fragment.variables[fragment.ports[p].variable]
                    for p in (rule.port_a, rule.port_b))
            if a.domain_key != b.domain_key:
                raise ValueError(f"Chain ports must have equal domains: {a.name}, {b.name}")
            box[axis] = (0, length - offset)
            consume(rule.port_a, box.copy())
            box[axis] = (offset, length)
            consume(rule.port_b, box)
            pairs = (grid.count // length) * (length - offset)
            topology["chain_bonds"] += pairs
            topology["ports"] -= 2 * pairs
            added_edges = -pairs
            variable = b
        elif isinstance(rule, Broadcast):
            consume(rule.port, box)
            if rule.hub in hubs:
                raise ValueError(f"duplicate Broadcast hub {rule.hub!r}")
            hubs.add(rule.hub)
            port = fragment.ports[rule.port]
            removed.add(port.variable)
            variable = fragment.variables[port.variable]
            # N=1 renames the branch. Otherwise add N-2 internal edges plus the hub.
            added_edges = grid.count - 1
            topology["broadcast_equalities"] += added_edges
            topology["factors"] += added_edges
            topology["ports"] -= added_edges
        else:
            continue
        topology["edges"] += added_edges
        for name, width in _widths(variable).items():
            edges[name] += width * added_edges
    return removed


def _factor_rows(factor, fragment, n, data, mean_field, arenas, leaves):
    """Count physical packed rows and yield (source owner, shape) for logical storage."""
    variables = {r: fragment.variables[v] for r, v in factor.edges.items()}
    if factor.type == "equality":
        if any(v.dim and v.name in data for v in variables.values()):
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
            if factor.edges["out"] not in data:
                raise ValueError(f"observation factor {factor.name!r} needs a DATA output")
            arenas["ef_mask"] += n
            arenas["ef_obs"] += n
        return
    binding = kernel.binding
    sizes = {r: _dimension(v.dim or v.cardinality) for r, v in variables.items()}
    shape_id = "x".join(str(sizes[role]) for role in binding.roles)
    if kernel.arena == "discrete":
        mode = "vmp" if factor.name in mean_field else "bp"
        group = f"{mode}:{factor.type}:{shape_id}"
    else:
        kind = factor.type
        for role, variable in variables.items():
            if variable.name in data:
                if binding.conditioning is None or role != binding.conditioning.role:
                    raise ValueError(f"unsupported Gaussian DATA on {factor.name}/{role}")
                kind = binding.conditioning.plan_type
                arenas["g_mask"] += n
                arenas["g_obs"] += n * sizes[role]
        group = f"gauss:{kind}:{shape_id}"
    paths = kernels.parameter_paths(kernel, group)
    for role, param in binding.parameters.items():
        shape = tuple(_dimension(s) for s in param.shape(sizes))
        yield factor.param_blobs[role], shape
        leaves[paths[param.key]] += n * prod(shape)


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
    if any(spec.fragment.plates for spec in template.plates.values()):
        raise UnknownStorageLayout("nested plate storage needs lexical owner counts")
    _with_shared_owners(template)
    arenas = Counter({name: 0 for name in STATE_ARRAYS})
    leaves = Counter()
    blobs = {}
    topology = dict(edges=0, factors=0, ports=0, chain_bonds=0, broadcast_equalities=0)

    def fragment_counts(fragment, n, plate="", spec=None, grid=None):
        topology["edges"] += n * len(fragment.variables)
        topology["factors"] += n * len(fragment.factors)
        topology["ports"] += n * len(fragment.ports)
        for variable in fragment.variables.values():
            for name, width in _widths(variable).items():
                arenas[name] += n * width
        removed = _wiring(spec, grid, arenas, topology) if spec else set()
        data = {p.variable for p in fragment.ports.values() if p.role is PortRole.DATA} - removed
        for port in fragment.ports.values():
            if port.role in (PortRole.PRIOR, PortRole.GOAL) and port.variable not in removed:
                raise ValueError(f"unterminated {port.role.value} port {port.name!r}")
        if any(c.kind not in (ConstraintKind.MEAN_FIELD, ConstraintKind.DELTA)
               for c in fragment.constraints):
            raise UnknownStorageLayout("specialized constraint storage requires compilation")
        mean_field = {c.target for c in fragment.constraints if c.kind is ConstraintKind.MEAN_FIELD}

        for factor in fragment.factors.values():
            for name, shape in _factor_rows(factor, fragment, n, data, mean_field, arenas, leaves):
                private = spec is not None and name in spec.per_copy_blobs
                owner = spec.shared_name(name) if spec else name
                if private:
                    owner = f"{plate}/{{index}}/{name}"
                copies = n if private else 1
                entry = {
                    "shape": shape, "copies": copies, "per_copy": private,
                    "bytes": _checked(prod(shape) * copies * source_dtype.itemsize, owner),
                }
                if owner in blobs and blobs[owner] != entry:
                    raise ValueError(f"source owner {owner!r} has incompatible shapes or multiplicity")
                blobs[owner] = entry

    fragment_counts(template, 1)
    for plate, spec in template.plates.items():
        grid = PlateGrid(spec.dims)
        fragment_counts(spec.fragment, grid.count, plate, spec, grid)
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

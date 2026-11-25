"""Resolve one live observation row for all legs of a Gaussian equality class."""

from dataclasses import replace

import numpy as np

from cffg.compiled import GaussianTapRole
from cffg.compiler_errors import CompileError
from cffg.topology import prefix_offsets


def resolve_taps(topology, groups, observations):
    tap_edges = {}
    admitted = None
    for sensor in topology.data_candidates:
        endpoint = topology.factors[sensor.endpoint[0]]
        if not topology.variables[sensor.variable].dim or endpoint.type != "equality":
            continue
        if admitted is None:
            admitted = {
                member for group in groups
                if group.type == "equality" or (group.kernel is not None and group.kernel.impl_id in
                    ("cffg.gaussian.gaussian", "cffg.gaussian.gaussian_linear"))
                for member in group.members
            }
        row = observations.n_obs_rows
        pending = [sensor.variable]
        seen = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            if name in tap_edges:
                raise CompileError(f"Gaussian tap {sensor.port!r}: two observations on the same equality class")
            seen.add(name)
            tap_edges[name] = row
            for member, _ in topology.endpoints[name]:
                node = topology.factors[member]
                if member not in admitted:
                    raise CompileError(f"Gaussian tap {sensor.port!r}: unsupported endpoint "
                                       f"{member!r} ({node.type}); requires a Gaussian prior or affine factor")
                if node.type == "equality":
                    pending.extend(node.edges.values())
        observations.gauss_data[sensor.port] = (row, topology.variables[sensor.variable].dim)
        observations.n_obs_rows += 1
    if not tap_edges:
        return groups, tap_edges

    starts = prefix_offsets(observations.gaussian_lengths())
    resolved = []
    for group in groups:
        taps = []
        # Every role of a tapped equality belongs to the same class.
        roles = group.roles[:1] if group.type == "equality" else group.roles
        for i, role in enumerate(roles):
            rows = np.array([tap_edges.get(topology.factors[m].edges[role], -1)
                             for m in group.members], dtype=np.int64)
            if np.any(rows >= 0):
                indices = starts[rows.clip(min=0), None] + np.arange(group.dims[i])
                indices[rows < 0] = 0
                taps.append(GaussianTapRole(i, rows, indices))
        resolved.append(replace(group, taps=tuple(taps)) if taps else group)
    return resolved, tap_edges

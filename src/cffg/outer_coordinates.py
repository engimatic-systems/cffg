"""Compiled chart conversion with immutable layout and dynamic reference metrics."""

from functools import lru_cache
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class ChartMetrics(NamedTuple):
    forward: tuple[jax.Array | np.ndarray, ...]
    inverse: tuple[jax.Array | np.ndarray, ...]
    whitening: tuple[jax.Array | np.ndarray, ...]


class CoordinateProgram(NamedTuple):
    pack: Callable
    unpack: Callable
    vector_groups: tuple[tuple[int, ...], ...]
    precision_groups: tuple[tuple[int, ...], ...]

    def metrics(self, chart):
        return ChartMetrics(
            tuple(np.stack([chart.vectors[i].forward for i in group]) for group in self.vector_groups),
            tuple(np.stack([chart.vectors[i].inverse for i in group]) for group in self.vector_groups),
            tuple(np.stack([chart.precisions[i].whitening for i in group]) for group in self.precision_groups),
        )


def _key(array):
    return array.dtype.str, array.shape, array.tobytes()


def structural_signature(chart):
    """Layout identity; numerical metrics are passed afresh for each solve."""
    return (
        tuple((value.dtype.str, value.shape) for value in chart.base),
        tuple((_key(block.segment), _key(block.support), block.reference, _key(block.coordinates))
              for block in chart.categorical),
        tuple((block.field, _key(block.indices), block.forward.dtype.str, block.forward.shape,
               block.inverse.dtype.str, block.inverse.shape,
               None if block.aliases is None else _key(block.aliases)) for block in chart.vectors),
        tuple((_key(block.indices), _key(block.lower), block.whitening.dtype.str, block.whitening.shape)
              for block in chart.precisions),
        chart.ef_shape,
    )


def coordinate_program(chart):
    return _program(structural_signature(chart))


def _array(key):
    dtype, shape, values = key
    return np.frombuffer(values, dtype=dtype).reshape(shape)


@lru_cache(maxsize=128)
def _program(signature):
    base, categorical, vectors, precisions, _ = signature
    fields = dict(zip(("messages", "g_eta", "g_lam", "ef_eta"), base, strict=True))
    active_dtypes = ([base[0][0]] if any(_array(block[3]).size for block in categorical) else [])
    active_dtypes.extend(dtype for field, _, dtype, *_ in vectors)
    active_dtypes.extend(fields[field][0] for field, *_ in vectors)
    active_dtypes.extend(dtype for _, _, dtype, *_ in precisions)
    if precisions:
        active_dtypes.append(base[2][0])
    offset = 0
    categorical_coordinates, references, support = [], [], []
    for _, active, reference, indices in categorical:
        indices = _array(indices)
        categorical_coordinates.extend(indices)
        references.extend([reference] * len(indices))
        support.extend(_array(active))
        offset += len(indices)
    categorical_width = offset
    vector_groups, vector_layout = {}, {}
    for member, (field, indices, *_metrics, aliases) in enumerate(vectors):
        indices = _array(indices)
        width = len(indices)
        group = field, width
        vector_groups.setdefault(group, []).append(member)
        targets = indices if aliases is None else _array(aliases).reshape(-1)
        vector_layout.setdefault(group, []).append((indices, np.arange(offset, offset + width), targets))
        offset += width
    precision_groups, precision_layout = {}, {}
    for member, (indices, lower, *_metric) in enumerate(precisions):
        indices, lower = _array(indices), _array(lower)
        dimension = len(indices)
        width = len(lower)
        group = dimension, tuple(map(tuple, lower))
        precision_groups.setdefault(group, []).append(member)
        precision_layout.setdefault(group, []).append((indices, lower, np.arange(offset, offset + width)))
        offset += width
    size = offset
    categorical_coordinates = np.asarray(categorical_coordinates, dtype=np.int64)
    references = np.asarray(references, dtype=np.int64)
    support = np.asarray(support, dtype=np.int64)
    grouped_vectors = []
    for (field, width), rows in vector_layout.items():
        indices, coordinates, targets = zip(*rows, strict=True)
        sources = np.concatenate([member * width + np.tile(np.arange(width), len(target) // width)
                                  for member, target in enumerate(targets)])
        grouped_vectors.append((field, np.stack(indices), np.stack(coordinates), np.concatenate(targets), sources))
    grouped_precisions = []
    for rows in precision_layout.values():
        indices, lower, coordinates = zip(*rows, strict=True)
        grouped_precisions.append((np.stack(indices), lower[0][:, 0], lower[0][:, 1], np.stack(coordinates)))

    @jax.jit
    def pack(arrays, metrics):
        dtype = jnp.result_type(*(np.dtype(value) for value in active_dtypes)) if active_dtypes else arrays.messages.dtype
        coordinates = jnp.zeros(size, dtype=dtype)
        if categorical_width:
            coordinates = coordinates.at[:categorical_width].set(
                arrays.messages[categorical_coordinates] - arrays.messages[references])
        for (field, indices, targets, _, _), forward in zip(grouped_vectors, metrics.forward, strict=True):
            values = getattr(arrays, field)[indices]
            coordinates = coordinates.at[targets].set(jnp.einsum("bij,bj->bi", forward, values))
        for (indices, rows, cols, targets), whitening in zip(grouped_precisions, metrics.whitening, strict=True):
            matrix = arrays.g_lam[indices]
            reduced = jnp.linalg.solve(whitening, jnp.linalg.solve(whitening, matrix).swapaxes(-1, -2)).swapaxes(-1, -2)
            coordinates = coordinates.at[targets].set(reduced[:, rows, cols])
        return coordinates

    @jax.jit
    def unpack(coordinates, arrays, metrics):
        messages = arrays.messages.at[support].set(0.)
        if categorical_width:
            messages = messages.at[categorical_coordinates].set(coordinates[:categorical_width])
        fields = {"messages": messages, "g_eta": arrays.g_eta, "ef_eta": arrays.ef_eta}
        for (field, _, source, targets, aliases), inverse in zip(grouped_vectors, metrics.inverse, strict=True):
            values = jnp.einsum("bij,bj->bi", inverse, coordinates[source]).reshape(-1)
            fields[field] = fields[field].at[targets].set(values[aliases])
        precision = arrays.g_lam
        for (indices, rows, cols, source), whitening in zip(grouped_precisions, metrics.whitening, strict=True):
            reduced = jnp.zeros(indices.shape, dtype=precision.dtype)
            reduced = reduced.at[:, rows, cols].set(coordinates[source])
            reduced = reduced.at[:, cols, rows].set(coordinates[source])
            matrix = whitening @ reduced @ whitening.swapaxes(-1, -2)
            precision = precision.at[indices.reshape(-1)].set(matrix.reshape(-1))
        return type(arrays)(fields["messages"], fields["g_eta"], precision, fields["ef_eta"])

    return CoordinateProgram(pack, unpack, tuple(tuple(group) for group in vector_groups.values()),
                             tuple(tuple(group) for group in precision_groups.values()))

"""Gauge-reduced coordinates and structural charts for outer inference."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from cffg import efrules
from cffg.compiled import NEG, Arena
from cffg.outer_structure import OuterStructure

class OuterArrays(NamedTuple):
    messages: jax.Array
    g_eta: jax.Array
    g_lam: jax.Array
    ef_eta: jax.Array


@dataclass(frozen=True)
class _CategoricalBlock:
    segment: np.ndarray
    support: np.ndarray
    reference: int
    coordinates: np.ndarray


@dataclass(frozen=True)
class _VectorBlock:
    field: str
    indices: np.ndarray
    forward: np.ndarray
    inverse: np.ndarray
    aliases: np.ndarray | None = None


@dataclass(frozen=True)
class _PrecisionBlock:
    indices: np.ndarray
    lower: np.ndarray
    whitening: np.ndarray


@dataclass(frozen=True)
class OuterChart:
    """One exact-support, gauge-reduced, dimensionless message chart."""

    base: OuterArrays
    categorical: tuple[_CategoricalBlock, ...]
    vectors: tuple[_VectorBlock, ...]
    precisions: tuple[_PrecisionBlock, ...]
    block_slices: tuple[tuple[int, int], ...]
    proper_reference: bool
    ef_shape: tuple[int, ...]

    @property
    def size(self) -> int:
        return self.block_slices[-1][1] if self.block_slices else 0

    @cached_property
    def program(self):
        from cffg.outer_coordinates import coordinate_program
        return coordinate_program(self)

    @cached_property
    def metrics(self):
        return self.program.metrics(self)

    def pack(self, arrays: OuterArrays, metrics=None) -> jax.Array:
        return self.program.pack(arrays, self.metrics if metrics is None else metrics)

    def unpack(self, coordinates, base=None, metrics=None) -> OuterArrays:
        return self.program.unpack(coordinates, self.base if base is None else base,
                                   self.metrics if metrics is None else metrics)


    def support_changed(self, messages: jax.Array) -> bool:
        host = np.asarray(messages)
        base = np.asarray(self.base.messages)
        return any(
            not np.array_equal(
                host[block.segment] != NEG,
                base[block.segment] != NEG,
            )
            for block in self.categorical
        )

    def residual_scale(self, residual: jax.Array) -> jax.Array:
        eps_root = np.sqrt(np.finfo(np.asarray(residual).dtype).eps)
        scale = np.empty(residual.shape, dtype=np.asarray(residual).dtype)
        for start, stop in self.block_slices:
            block = np.asarray(residual[start:stop])
            rms = float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0
            scale[start:stop] = max(rms, eps_root)
        return jnp.asarray(scale)


def _linear_map(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return matrix, np.linalg.inv(matrix)


def _unit_scaled(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rms = float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0
    unit = max(rms, 1.0)
    eye = np.eye(values.size, dtype=values.dtype)
    return eye / unit, eye * unit


def build_outer_chart(
    arena: Arena | OuterStructure,
    arrays: OuterArrays,
    eliminated: dict[str, frozenset[int]] | None = None,
) -> OuterChart:
    """Build the exact active chart and fixed continuous reference metrics."""
    eliminated = eliminated or {}
    discrete_eliminated = eliminated.get("messages", frozenset())
    gaussian_eliminated = eliminated.get("gaussian", frozenset())
    ef_eliminated = eliminated.get("ef", frozenset())
    messages = np.asarray(arrays.messages)
    dtype = messages.dtype
    categorical: list[_CategoricalBlock] = []
    vectors: list[_VectorBlock] = []
    precisions: list[_PrecisionBlock] = []
    proper = True

    parameter_aliases: dict[int, np.ndarray] = {}
    duplicate_slots: set[int] = set()
    for plan in arena.goal_parameter_plans:
        for slots in (plan.prior_scatter, plan.count_scatter):
            if len(slots) > 1:
                parameter_aliases[int(slots[0])] = arena.message_indices(
                    slots, plan.cardinality
                )
                duplicate_slots.update(int(slot) for slot in slots[1:])

    for edge in sorted(arena.edges.values(), key=lambda item: item.name):
        if not edge.dim and not edge.ef and not edge.param_shape:
            for slot in (edge.slot0, edge.slot1):
                if slot in discrete_eliminated:
                    continue
                indices = arena.message_indices(slot, edge.card)
                support = indices[messages[indices] != NEG]
                if not support.size:
                    proper = False
                    continue
                block = _CategoricalBlock(
                    segment=indices,
                    support=support,
                    reference=int(support[0]),
                    coordinates=support[1:],
                )
                categorical.append(block)
        elif edge.param_shape:
            width = int(np.prod(edge.param_shape))
            for slot in (edge.slot0, edge.slot1):
                if slot in discrete_eliminated or slot in duplicate_slots:
                    continue
                indices = arena.message_indices(slot, width)
                forward, inverse = _unit_scaled(messages[indices])
                vectors.append(
                    _VectorBlock(
                        "messages", indices, forward, inverse, parameter_aliases.get(slot)
                    )
                )

    g_eta = np.asarray(arrays.g_eta)
    g_lam = np.asarray(arrays.g_lam)
    for edge in sorted(arena.edges.values(), key=lambda item: item.name):
        if not edge.dim:
            continue
        dim = edge.dim
        eta_indices = [
            arena.g_eta_indices(slot, dim) for slot in (edge.slot0, edge.slot1)
        ]
        lam_indices = [
            arena.g_lam_indices(slot, dim) for slot in (edge.slot0, edge.slot1)
        ]
        aggregate = sum(
            g_lam[index.reshape(-1)].reshape(dim, dim)
            for index in lam_indices
        )
        aggregate = 0.5 * (aggregate + aggregate.T)
        operator_scale = float(np.linalg.norm(aggregate, ord=2))
        rank_floor = np.finfo(dtype).eps * dim * operator_scale
        if np.linalg.eigvalsh(aggregate).min(initial=np.inf) > rank_floor:
            whitening = np.linalg.cholesky(aggregate)
        else:
            whitening = np.eye(dim, dtype=dtype)
            proper = False
        eta_forward, eta_inverse = _linear_map(
            np.linalg.inv(whitening)
        )
        lower = np.argwhere(np.tril(np.ones((dim, dim), dtype=bool)))
        for slot, eta_index, lam_index in zip(
            (edge.slot0, edge.slot1), eta_indices, lam_indices
        ):
            if slot in gaussian_eliminated:
                continue
            vectors.append(
                _VectorBlock(
                    "g_eta", eta_index, eta_forward, eta_inverse
                )
            )
            precisions.append(
                _PrecisionBlock(lam_index, lower, whitening)
            )

    ef_eta = np.asarray(arrays.ef_eta)
    for edge in sorted(arena.edges.values(), key=lambda item: item.name):
        if not edge.ef:
            continue
        family = efrules.JFAMILIES[edge.ef]
        aggregate = ef_eta[edge.slot0] + ef_eta[edge.slot1]
        active = np.arange(aggregate.size, dtype=np.int64)
        if bool(np.asarray(family.is_proper(jnp.asarray(aggregate)))):
            fisher = np.asarray(
                jax.jacfwd(family.mean_stats)(jnp.asarray(aggregate))
            )
            fisher = 0.5 * (fisher + fisher.T)
            scale = max(float(np.linalg.norm(fisher, ord=2)), 1.0)
            rank_floor = np.finfo(dtype).eps * aggregate.size * scale
            if np.linalg.eigvalsh(fisher).min(initial=np.inf) > rank_floor:
                metric = np.linalg.cholesky(fisher).T
            else:
                metric = np.eye(aggregate.size, dtype=dtype)
                proper = False
        else:
            metric = np.eye(aggregate.size, dtype=dtype)
            proper = False
        forward, inverse = _linear_map(metric)
        for slot in (edge.slot0, edge.slot1):
            if slot in ef_eliminated:
                continue
            indices = slot * ef_eta.shape[1] + active
            vectors.append(
                _VectorBlock("ef_eta", indices, forward, inverse)
            )

    flat_arrays = OuterArrays(
        arrays.messages,
        arrays.g_eta,
        arrays.g_lam,
        arrays.ef_eta.reshape(-1),
    )
    ordered_widths = [
        *(block.coordinates.size for block in categorical),
        *(block.indices.size for block in vectors),
        *(block.lower.shape[0] for block in precisions),
    ]
    ordered_slices: list[tuple[int, int]] = []
    offset = 0
    for width in ordered_widths:
        if width:
            ordered_slices.append((offset, offset + width))
            offset += width
    return OuterChart(
        flat_arrays,
        tuple(categorical),
        tuple(vectors),
        tuple(precisions),
        tuple(ordered_slices),
        proper,
        tuple(arrays.ef_eta.shape),
    )


def chart_arrays(
    chart: OuterChart,
    coordinates: jax.Array,
    base: OuterArrays | None = None,
    metrics=None,
) -> OuterArrays:
    arrays = chart.unpack(coordinates, base, metrics)
    return arrays._replace(ef_eta=arrays.ef_eta.reshape(chart.ef_shape))


def chart_coordinates(chart: OuterChart, arrays: OuterArrays, metrics=None) -> jax.Array:
    return chart.pack(arrays._replace(ef_eta=arrays.ef_eta.reshape(-1)), metrics)


def outer_chart_signature(chart: OuterChart) -> tuple:
    """Exact cache identity for a chart's support, dtype, and metrics."""
    def array_key(value: np.ndarray) -> tuple:
        return value.dtype.str, value.shape, value.tobytes()

    return (
        np.asarray(chart.base.messages).dtype.str,
        tuple(
            (
                array_key(block.segment),
                array_key(block.support),
                block.reference,
                array_key(block.coordinates),
            )
            for block in chart.categorical
        ),
        tuple(
            (
                block.field,
                array_key(block.indices),
                array_key(block.forward),
                array_key(block.inverse),
                None if block.aliases is None else array_key(block.aliases),
            )
            for block in chart.vectors
        ),
        tuple(
            (
                array_key(block.indices),
                array_key(block.lower),
                array_key(block.whitening),
            )
            for block in chart.precisions
        ),
        chart.ef_shape,
    )

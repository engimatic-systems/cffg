"""Gauge-reduced matrix-free globalization for mixed-GFE outer inference.

The scientific objective remains the engine's Bethe/generalised free-energy
ledger. This module uses the fixed-point residual only as a numerical merit
for trust-region step acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from cffg import efrules
from cffg.compiled import NEG, Arena
from cffg.runtime_types import OuterTerminationReason as OuterTerminationReason




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

    def pack(self, arrays: OuterArrays) -> jax.Array:
        pieces: list[jax.Array] = []
        for block in self.categorical:
            if block.coordinates.size:
                pieces.append(
                    arrays.messages[block.coordinates]
                    - arrays.messages[block.reference]
                )
        for block in self.vectors:
            values = getattr(arrays, block.field)[block.indices]
            pieces.append(jnp.asarray(block.forward) @ values)
        for block in self.precisions:
            dim = block.indices.shape[0]
            matrix = arrays.g_lam[block.indices.reshape(-1)].reshape(dim, dim)
            whitening = jnp.asarray(block.whitening)
            reduced = jnp.linalg.solve(
                whitening, jnp.linalg.solve(whitening, matrix).T
            ).T
            pieces.append(reduced[block.lower[:, 0], block.lower[:, 1]])
        if not pieces:
            return jnp.zeros((0,), dtype=self.base.messages.dtype)
        return jnp.concatenate([jnp.ravel(piece) for piece in pieces])

    def unpack(
        self,
        coordinates: jax.Array,
        base: OuterArrays | None = None,
    ) -> OuterArrays:
        arrays = self.base if base is None else base
        offset = 0
        messages = arrays.messages
        for block in self.categorical:
            messages = messages.at[block.support].set(0.0)
            width = block.coordinates.size
            if width:
                messages = messages.at[block.coordinates].set(
                    coordinates[offset : offset + width]
                )
                offset += width

        fields = {
            "messages": messages,
            "g_eta": arrays.g_eta,
            "ef_eta": arrays.ef_eta,
        }
        for block in self.vectors:
            width = block.indices.size
            values = (
                jnp.asarray(block.inverse)
                @ coordinates[offset : offset + width]
            )
            targets = block.indices if block.aliases is None else block.aliases
            fields[block.field] = fields[block.field].at[targets].set(
                values
            )
            offset += width

        g_lam = arrays.g_lam
        for block in self.precisions:
            width = block.lower.shape[0]
            reduced = jnp.zeros(block.indices.shape, dtype=g_lam.dtype)
            values = coordinates[offset : offset + width]
            rows, cols = block.lower[:, 0], block.lower[:, 1]
            reduced = reduced.at[rows, cols].set(values)
            reduced = reduced.at[cols, rows].set(values)
            whitening = jnp.asarray(block.whitening)
            matrix = whitening @ reduced @ whitening.T
            g_lam = g_lam.at[block.indices.reshape(-1)].set(matrix.reshape(-1))
            offset += width

        return OuterArrays(
            fields["messages"], fields["g_eta"], g_lam, fields["ef_eta"]
        )

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
    arena: Arena,
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
        operator_scale = max(float(np.linalg.norm(aggregate, ord=2)), 1.0)
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
) -> OuterArrays:
    arrays = chart.unpack(coordinates, base)
    return arrays._replace(ef_eta=arrays.ef_eta.reshape(chart.ef_shape))


def chart_coordinates(chart: OuterChart, arrays: OuterArrays) -> jax.Array:
    return chart.pack(arrays._replace(ef_eta=arrays.ef_eta.reshape(-1)))


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


@dataclass(frozen=True)
class TrustStep:
    step: jax.Array
    predicted_reduction: float
    gradient_norm: float
    hit_boundary: bool
    linear_exhausted: bool
    finite: bool


class OuterKernelPlan(NamedTuple):
    trust: Callable
    root_diagnostics: Callable
    context: object
    compiled: bool


def _make_root_diagnostics(
    residual_fn: Callable[[jax.Array], jax.Array],
):
    """Compile the residual merit and its matrix-free gradient at a root."""
    @jax.jit
    def diagnostics(coordinates, residual_scale, context=None):
        def scaled(value):
            residual = (
                residual_fn(value)
                if context is None
                else residual_fn(value, context)
            )
            return residual / residual_scale

        residual, pullback = jax.vjp(scaled, coordinates)
        gradient = pullback(residual)[0]
        merit = 0.5 * jnp.vdot(residual, residual).real
        return merit, jnp.linalg.norm(gradient)

    return diagnostics


def _boundary_distance(point, direction, radius):
    a = jnp.vdot(direction, direction).real
    b = 2.0 * jnp.vdot(point, direction).real
    c = jnp.vdot(point, point).real - radius * radius
    discriminant = jnp.maximum(b * b - 4.0 * a * c, 0.0)
    return (-b + jnp.sqrt(discriminant)) / (2.0 * a)


def _make_trust_kernel(
    residual_fn: Callable[[jax.Array], jax.Array],
    dimension: int,
    max_linear_iters: int | None = None,
):
    """Compile one reusable matrix-free Steihaug kernel for a fixed chart."""
    if max_linear_iters is None:
        max_linear_iters = max(1, dimension)

    @jax.jit
    def kernel(
        coordinates, residual_scale, radius,
        accelerated_step, has_accelerated_step, context=None,
    ):
        def scaled(value):
            residual = (
                residual_fn(value)
                if context is None
                else residual_fn(value, context)
            )
            return residual / residual_scale
        residual, jvp = jax.linearize(scaled, coordinates)
        _, pullback = jax.vjp(scaled, coordinates)
        gradient = pullback(residual)[0]
        gradient_norm = jnp.linalg.norm(gradient)
        eps = jnp.finfo(coordinates.dtype).eps
        target = jnp.sqrt(eps) * jnp.maximum(gradient_norm, 1.0)

        def gauss_product(value):
            return pullback(jvp(value))[0]

        def body(carry):
            i, step, linear_residual, direction, done, hit_boundary = carry
            image = gauss_product(direction)
            curvature = jnp.vdot(direction, image).real
            curvature_floor = (
                eps
                * jnp.maximum(jnp.linalg.norm(direction), 1.0)
                * jnp.maximum(jnp.linalg.norm(image), 1.0)
            )
            rr = jnp.vdot(linear_residual, linear_residual).real
            alpha = rr / jnp.where(curvature != 0.0, curvature, 1.0)
            candidate = step + alpha * direction
            use_boundary = (
                ~jnp.isfinite(curvature)
                | (curvature <= curvature_floor)
                | (jnp.linalg.norm(candidate) >= radius)
            )
            tau = _boundary_distance(step, direction, radius)
            boundary_step = step + tau * direction
            next_residual = linear_residual - alpha * image
            converged = jnp.linalg.norm(next_residual) <= target
            next_rr = jnp.vdot(next_residual, next_residual).real
            beta = next_rr / jnp.where(rr != 0.0, rr, 1.0)
            next_direction = next_residual + beta * direction
            return (
                i + 1,
                jnp.where(use_boundary, boundary_step, candidate),
                jnp.where(
                    use_boundary, linear_residual, next_residual
                ),
                jnp.where(use_boundary, direction, next_direction),
                done | use_boundary | converged,
                hit_boundary | use_boundary,
            )

        def condition(carry):
            i, _, _, _, done, _ = carry
            return (i < max_linear_iters) & ~done

        initial = (
            jnp.asarray(0),
            jnp.zeros_like(coordinates),
            -gradient,
            -gradient,
            jnp.asarray(False),
            jnp.asarray(False),
        )
        iterations, step, _, _, done, hit_boundary = jax.lax.while_loop(
            condition, body, initial
        )

        def predicted(candidate):
            image = jvp(candidate)
            return -(
                jnp.vdot(gradient, candidate).real
                + 0.5 * jnp.vdot(image, image).real
            )

        # Certify the truncated solve against the matrix-free Cauchy point.
        gg = jnp.vdot(gradient, gradient).real
        bg = gauss_product(gradient)
        gbg = jnp.vdot(gradient, bg).real
        boundary_alpha = radius / jnp.where(
            gradient_norm != 0.0, gradient_norm, 1.0
        )
        alpha = jnp.where(
            gbg > eps * jnp.maximum(gg, 1.0),
            jnp.minimum(gg / gbg, boundary_alpha),
            boundary_alpha,
        )
        cauchy = -alpha * gradient
        use_cauchy = predicted(step) < 0.5 * predicted(cauchy)
        step = jnp.where(use_cauchy, cauchy, step)
        hit_boundary = hit_boundary | (
            use_cauchy & (alpha == boundary_alpha)
        )
        fixed_point_step = residual * residual_scale
        fractions = jnp.asarray(
            [1.0, 0.5, 0.25, 0.125], dtype=coordinates.dtype
        )
        candidates = fractions[:, None] * fixed_point_step[None, :]
        candidate_norms = jnp.linalg.norm(candidates, axis=1)
        candidate_scales = jnp.minimum(
            1.0,
            radius / jnp.where(candidate_norms != 0.0, candidate_norms, 1.0),
        )
        candidates = candidate_scales[:, None] * candidates
        candidate_predictions = jax.vmap(predicted)(candidates)
        best = jnp.argmax(candidate_predictions)
        fixed_point_step = candidates[best]
        fixed_point_prediction = candidate_predictions[best]
        use_fixed_point = fixed_point_prediction > predicted(step)
        step = jnp.where(use_fixed_point, fixed_point_step, step)
        hit_boundary = hit_boundary | (
            use_fixed_point & (candidate_scales[best] < 1.0)
        )
        accelerated_norm = jnp.linalg.norm(accelerated_step)
        accelerated_scale = jnp.minimum(
            1.0,
            radius / jnp.where(accelerated_norm != 0.0, accelerated_norm, 1.0),
        )
        accelerated_step = accelerated_scale * accelerated_step
        blend_fractions = jnp.asarray(
            [1.0, 0.5, 0.25, 0.125], dtype=coordinates.dtype
        )
        accelerated_candidates = (
            blend_fractions[:, None] * accelerated_step[None, :]
            + (1.0 - blend_fractions[:, None]) * step[None, :]
        )
        accelerated_norms = jnp.linalg.norm(
            accelerated_candidates, axis=1
        )
        accelerated_scales = jnp.minimum(
            1.0,
            radius / jnp.where(
                accelerated_norms != 0.0, accelerated_norms, 1.0
            ),
        )
        accelerated_candidates = (
            accelerated_scales[:, None] * accelerated_candidates
        )
        accelerated_predictions = jax.vmap(predicted)(
            accelerated_candidates
        )
        accelerated_best = jnp.argmax(accelerated_predictions)
        use_accelerated = has_accelerated_step & (
            accelerated_predictions[accelerated_best] > predicted(step)
        )
        step = jnp.where(
            use_accelerated,
            accelerated_candidates[accelerated_best],
            step,
        )
        hit_boundary = hit_boundary | (
            use_accelerated
            & (accelerated_scales[accelerated_best] < 1.0)
        )
        prediction = predicted(step)
        finite = (
            jnp.all(jnp.isfinite(residual))
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(step))
            & jnp.isfinite(prediction)
        )
        exhausted = (iterations >= max_linear_iters) & ~done
        return (
            step,
            prediction,
            gradient_norm,
            hit_boundary,
            exhausted,
            finite,
            accelerated_candidates,
            accelerated_predictions,
            candidates,
            candidate_predictions,
            iterations,
        )

    return kernel


@dataclass(frozen=True)
class OuterSolveResult:
    arrays: OuterArrays
    reason: OuterTerminationReason
    iterations: int
    message_deltas: tuple[float, ...]
    free_energies: tuple[float, ...]
    outer_residual: float
    merit: float
    gradient_norm: float
    trust_radius: float
    accepted_steps: int
    rejected_steps: int
    inner_residual: float
    inner_code: int
    outer_map_evaluations: int = 0
    trial_evaluations: int = 0
    linear_iterations: int = 0
    solver_compilations: int = 0


def _finite(arrays: OuterArrays) -> bool:
    return all(
        bool(np.isfinite(np.asarray(value)).all()) for value in arrays
    )


def _array_delta(left: OuterArrays, right: OuterArrays) -> float:
    return max(
        (
            float(jnp.max(jnp.abs(a - b)))
            for a, b in zip(left, right)
            if a.size
        ),
        default=0.0,
    )


def solve_outer(
    initial: OuterArrays,
    arena: Arena,
    outer_map: Callable[
        [OuterArrays], tuple[OuterArrays, jax.Array, jax.Array]
    ],
    energy: Callable[[OuterArrays], float],
    infeasible: Callable[[OuterArrays], bool],
    *,
    n_iters: int,
    tol: float,
    eliminated: dict[str, frozenset[int]] | None = None,
    kernel_factory: Callable[[OuterChart], OuterKernelPlan] | None = None,
    candidate_limit: int = 128,
    _support_restarts: int = 0,
) -> OuterSolveResult:
    """Globalize one fixed-support deterministic outer map."""
    current = initial
    dtype = np.asarray(initial.messages).dtype
    eps = np.finfo(dtype).eps
    effective_tol = max(float(tol), 64.0 * eps)
    energies: list[float] = []
    deltas: list[float] = []
    accepted = rejected = 0
    restarts = _support_restarts
    radius = 1.0
    merit = gradient_norm = outer_residual = float("inf")
    inner_residual = float("inf")
    inner_code = 0
    outer_map_evaluations = 0
    trial_evaluations = 0
    linear_iterations = 0
    solver_compilations = 0

    initial_energy = energy(current)
    if np.isfinite(initial_energy):
        energies.append(initial_energy)

    # Discover structural support and a proper continuous reference before
    # freezing the chart. A changed face restarts exactly; no coordinate is
    # floored into the old face.
    restart_limit = max(
        sum(
            2
            for edge in arena.edges.values()
            if not edge.dim and not edge.ef and not edge.param_shape
            and edge.slot0 not in (eliminated or {}).get("messages", ())
        ),
        1,
    )
    while True:
        chart = build_outer_chart(arena, current, eliminated)
        coordinates = chart_coordinates(chart, current)

        def evaluate(value):
            nonlocal outer_map_evaluations
            outer_map_evaluations += 1
            mapped, local_residual, local_code = outer_map(
                chart_arrays(chart, value)
            )
            return chart_coordinates(chart, mapped), mapped, local_residual, local_code

        mapped_coordinates, mapped, local_residual, local_code = evaluate(
            coordinates
        )
        inner_residual = float(local_residual)
        inner_code = int(local_code)
        if inner_code:
            return OuterSolveResult(
                current, OuterTerminationReason.INNER_FAILURE, 0,
                tuple(deltas), tuple(energies), outer_residual, merit,
                gradient_norm, radius, accepted, rejected,
                inner_residual, inner_code,
            )
        if not _finite(current) or not _finite(mapped):
            return OuterSolveResult(
                current, OuterTerminationReason.NONFINITE_EVALUATION, 0,
                tuple(deltas), tuple(energies), float("inf"), float("inf"),
                float("inf"), radius, accepted, rejected,
                inner_residual, inner_code,
            )
        if infeasible(current) or infeasible(mapped):
            return OuterSolveResult(
                current, OuterTerminationReason.INFEASIBLE, 0,
                tuple(deltas), tuple(energies), float("inf"), float("inf"),
                float("inf"), radius, accepted, rejected,
                inner_residual, inner_code,
            )
        if chart.support_changed(mapped.messages) or not chart.proper_reference:
            restarts += 1
            if restarts > restart_limit:
                return OuterSolveResult(
                    current,
                    OuterTerminationReason.SUPPORT_RESTART_EXHAUSTED,
                    0, tuple(deltas), tuple(energies), float("inf"),
                    float("inf"), float("inf"), radius, accepted, rejected,
                    inner_residual, inner_code,
                )
            current = mapped
            continue
        break

    residual_fn = lambda value: evaluate(value)[0] - value
    initial_residual = mapped_coordinates - coordinates
    residual_scale = chart.residual_scale(initial_residual)
    radius = max(float(jnp.linalg.norm(initial_residual)), 1.0)
    radius_ceiling = (
        max(
            float(jnp.linalg.norm(coordinates)),
            radius,
            np.sqrt(max(chart.size, 1)),
        )
        / np.sqrt(eps)
    )
    shadow_coordinates = np.asarray(coordinates).copy()
    shadow_history_coordinates: list[np.ndarray] = []
    shadow_history_residuals: list[np.ndarray] = []

    # A damped fixed-point sequence is a cheap candidate generator for the
    # common benign case. It advances the arena arrays directly, avoiding
    # chart dispatch on every sweep, but never certifies itself.
    candidate_iterations = 0
    for _ in range(min(candidate_limit, n_iters // 2)):
        raw_residual = _array_delta(current, mapped)
        if raw_residual <= effective_tol:
            coordinates = chart_coordinates(chart, current)
            mapped_coordinates = chart_coordinates(chart, mapped)
            residual = mapped_coordinates - coordinates
            outer_residual = float(
                jnp.max(
                    jnp.abs(residual) / (1.0 + jnp.abs(coordinates)),
                    initial=0.0,
                )
            )
            current_energy = energy(current)
            mapped_energy = energy(mapped)
            ledger_stationary = (
                np.isfinite(current_energy)
                and np.isfinite(mapped_energy)
                and abs(mapped_energy - current_energy)
                <= effective_tol * (1.0 + abs(current_energy))
            )
            if (
                outer_residual <= effective_tol
                and ledger_stationary
                and chart.proper_reference
            ):
                scaled_residual = residual / residual_scale
                merit = 0.5 * float(
                    jnp.vdot(scaled_residual, scaled_residual).real
                )
                if kernel_factory is None:
                    root_diagnostics = _make_root_diagnostics(residual_fn)
                    root_merit, root_gradient = root_diagnostics(
                        coordinates, residual_scale
                    )
                else:
                    plan = kernel_factory(chart)
                    solver_compilations += int(plan.compiled)
                    root_merit, root_gradient = plan.root_diagnostics(
                        coordinates, residual_scale, plan.context
                    )
                outer_map_evaluations += 1
                merit = float(root_merit)
                gradient_norm = float(root_gradient)
                deltas.append(outer_residual)
                if not energies or energies[-1] != current_energy:
                    energies.append(current_energy)
                if energies[-1] != mapped_energy:
                    energies.append(mapped_energy)
                return OuterSolveResult(
                    mapped, OuterTerminationReason.ROOT_CONVERGED,
                    candidate_iterations, tuple(deltas), tuple(energies),
                    outer_residual, merit, gradient_norm, radius, accepted,
                    rejected, inner_residual, inner_code,
                    outer_map_evaluations, trial_evaluations,
                    linear_iterations, solver_compilations,
                )

        next_arrays = OuterArrays(
            *(0.5 * left + 0.5 * right
              for left, right in zip(current, mapped))
        )
        deltas.append(_array_delta(current, next_arrays))
        current = next_arrays
        candidate_iterations += 1
        mapped, local_residual, local_code = outer_map(current)
        outer_map_evaluations += 1
        inner_residual = float(local_residual)
        inner_code = int(local_code)
        if inner_code:
            return OuterSolveResult(
                current, OuterTerminationReason.INNER_FAILURE,
                candidate_iterations, tuple(deltas), tuple(energies),
                outer_residual, merit, gradient_norm, radius, accepted,
                rejected, inner_residual, inner_code,
            )
        # ``current`` is a convex blend of the previous two already-valid
        # states on the same exact support face, so only the fresh map output
        # needs another finite/feasibility check here.
        if not _finite(mapped):
            return OuterSolveResult(
                current, OuterTerminationReason.NONFINITE_EVALUATION,
                candidate_iterations, tuple(deltas), tuple(energies),
                float("inf"), float("inf"), float("inf"), radius,
                accepted, rejected, inner_residual, inner_code,
            )
        if infeasible(mapped):
            return OuterSolveResult(
                current, OuterTerminationReason.INFEASIBLE,
                candidate_iterations, tuple(deltas), tuple(energies),
                float("inf"), float("inf"), float("inf"), radius,
                accepted, rejected, inner_residual, inner_code,
            )
        if chart.support_changed(mapped.messages):
            restarts += 1
            if restarts > restart_limit:
                return OuterSolveResult(
                    current,
                    OuterTerminationReason.SUPPORT_RESTART_EXHAUSTED,
                    candidate_iterations, tuple(deltas), tuple(energies),
                    float("inf"), float("inf"), float("inf"), radius,
                    accepted, rejected, inner_residual, inner_code,
                )
            restarted = solve_outer(
                mapped, arena, outer_map, energy, infeasible,
                n_iters=max(n_iters - candidate_iterations, 0),
                tol=tol,
                eliminated=eliminated,
                kernel_factory=kernel_factory,
                candidate_limit=candidate_limit,
                _support_restarts=restarts,
            )
            return OuterSolveResult(
                restarted.arrays,
                restarted.reason,
                candidate_iterations + restarted.iterations,
                tuple((*deltas, *restarted.message_deltas)),
                tuple((*energies, *restarted.free_energies)),
                restarted.outer_residual,
                restarted.merit,
                restarted.gradient_norm,
                restarted.trust_radius,
                accepted + restarted.accepted_steps,
                rejected + restarted.rejected_steps,
                restarted.inner_residual,
                restarted.inner_code,
                outer_map_evaluations + restarted.outer_map_evaluations,
                trial_evaluations + restarted.trial_evaluations,
                linear_iterations + restarted.linear_iterations,
                solver_compilations + restarted.solver_compilations,
            )

    n_iters -= candidate_iterations
    chart = build_outer_chart(arena, current, eliminated)
    coordinates = chart_coordinates(chart, current)

    def evaluate(value):
        nonlocal outer_map_evaluations
        outer_map_evaluations += 1
        mapped, local_residual, local_code = outer_map(
            chart_arrays(chart, value)
        )
        return chart_coordinates(chart, mapped), mapped, local_residual, local_code

    residual_fn = lambda value: evaluate(value)[0] - value
    mapped_coordinates, mapped, local_residual, local_code = evaluate(coordinates)
    inner_residual = float(local_residual)
    inner_code = int(local_code)
    initial_residual = mapped_coordinates - coordinates
    residual_scale = chart.residual_scale(initial_residual)
    radius = max(float(jnp.linalg.norm(initial_residual)), 1.0)
    radius_ceiling = (
        max(
            float(jnp.linalg.norm(coordinates)),
            radius,
            np.sqrt(max(chart.size, 1)),
        )
        / np.sqrt(eps)
    )
    shadow_coordinates = np.asarray(coordinates).copy()
    shadow_history_coordinates = []
    shadow_history_residuals = []
    if kernel_factory is None:
        plan = OuterKernelPlan(
            _make_trust_kernel(residual_fn, chart.size),
            _make_root_diagnostics(residual_fn),
            None,
            True,
        )
    else:
        plan = kernel_factory(chart)
    solver_compilations += int(plan.compiled)
    trust_kernel = plan.trust

    for iteration in range(n_iters + 1):
        mapped_coordinates, mapped, local_residual, local_code = evaluate(
            coordinates
        )
        inner_residual = float(local_residual)
        inner_code = int(local_code)
        if inner_code:
            return OuterSolveResult(
                current, OuterTerminationReason.INNER_FAILURE,
                candidate_iterations + iteration,
                tuple(deltas), tuple(energies), outer_residual, merit,
                gradient_norm, radius, accepted, rejected,
                inner_residual, inner_code,
            )
        if not _finite(current) or not _finite(mapped):
            return OuterSolveResult(
                current, OuterTerminationReason.NONFINITE_EVALUATION,
                candidate_iterations + iteration,
                tuple(deltas), tuple(energies), float("inf"), float("inf"),
                float("inf"), radius, accepted, rejected,
                inner_residual, inner_code,
            )
        if infeasible(current) or infeasible(mapped):
            return OuterSolveResult(
                current, OuterTerminationReason.INFEASIBLE,
                candidate_iterations + iteration,
                tuple(deltas), tuple(energies), float("inf"), float("inf"),
                float("inf"), radius, accepted, rejected,
                inner_residual, inner_code,
            )
        if chart.support_changed(mapped.messages):
            restarts += 1
            if restarts > restart_limit:
                return OuterSolveResult(
                    current,
                    OuterTerminationReason.SUPPORT_RESTART_EXHAUSTED,
                    candidate_iterations + iteration,
                    tuple(deltas), tuple(energies), float("inf"),
                    float("inf"), float("inf"), radius, accepted, rejected,
                    inner_residual, inner_code,
                )
            restarted = solve_outer(
                mapped,
                arena,
                outer_map,
                energy,
                infeasible,
                n_iters=max(n_iters - iteration, 0),
                tol=tol,
                eliminated=eliminated,
                kernel_factory=kernel_factory,
                candidate_limit=candidate_limit,
                _support_restarts=restarts,
            )
            return OuterSolveResult(
                restarted.arrays,
                restarted.reason,
                candidate_iterations + iteration + restarted.iterations,
                tuple((*deltas, *restarted.message_deltas)),
                tuple((*energies, *restarted.free_energies)),
                restarted.outer_residual,
                restarted.merit,
                restarted.gradient_norm,
                restarted.trust_radius,
                accepted + restarted.accepted_steps,
                rejected + restarted.rejected_steps,
                restarted.inner_residual,
                restarted.inner_code,
                outer_map_evaluations + restarted.outer_map_evaluations,
                trial_evaluations + restarted.trial_evaluations,
                linear_iterations + restarted.linear_iterations,
                solver_compilations + restarted.solver_compilations,
            )
        residual = mapped_coordinates - coordinates
        outer_residual = float(
            jnp.max(
                jnp.abs(residual) / (1.0 + jnp.abs(coordinates)),
                initial=0.0,
            )
        )
        current_energy = energy(current)
        mapped_energy = energy(mapped)
        ledger_stationary = (
            np.isfinite(current_energy)
            and np.isfinite(mapped_energy)
            and abs(mapped_energy - current_energy)
            <= effective_tol * (1.0 + abs(current_energy))
        )
        if (
            outer_residual <= effective_tol
            and ledger_stationary
            and chart.proper_reference
        ):
            scaled_residual = residual / residual_scale
            merit = 0.5 * float(
                jnp.vdot(scaled_residual, scaled_residual).real
            )
            root_step = trust_kernel(
                coordinates,
                residual_scale,
                jnp.asarray(radius, coordinates.dtype),
                jnp.zeros_like(coordinates),
                jnp.asarray(False),
                plan.context,
            )
            outer_map_evaluations += 1
            linear_iterations += int(root_step[10])
            gradient_norm = float(root_step[2])
            deltas.append(outer_residual)
            if not energies or energies[-1] != current_energy:
                energies.append(current_energy)
            return OuterSolveResult(
                mapped, OuterTerminationReason.ROOT_CONVERGED,
                candidate_iterations + iteration,
                tuple(deltas), tuple(energies), outer_residual, merit,
                gradient_norm,
                radius, accepted, rejected, inner_residual, inner_code,
                outer_map_evaluations, trial_evaluations,
                linear_iterations, solver_compilations,
            )
        if iteration == n_iters:
            break

        scaled_residual = residual / residual_scale
        merit = 0.5 * float(jnp.vdot(scaled_residual, scaled_residual).real)
        host_coordinates = np.asarray(coordinates)
        shadow_mapped_coordinates, shadow_mapped, shadow_inner, shadow_code = (
            evaluate(jnp.asarray(shadow_coordinates))
        )
        shadow_residual = (
            np.asarray(shadow_mapped_coordinates) - shadow_coordinates
        )
        shadow_arrays = chart_arrays(
            chart, jnp.asarray(shadow_coordinates)
        )
        shadow_outer_residual = float(
            np.max(
                np.abs(shadow_residual)
                / (1.0 + np.abs(shadow_coordinates)),
                initial=0.0,
            )
        )
        shadow_energy = energy(shadow_arrays)
        shadow_mapped_energy = energy(shadow_mapped)
        shadow_root = (
            int(shadow_code) == 0
            and np.isfinite(float(shadow_inner))
            and _finite(shadow_arrays)
            and _finite(shadow_mapped)
            and not infeasible(shadow_arrays)
            and not infeasible(shadow_mapped)
            and not chart.support_changed(shadow_mapped.messages)
            and shadow_outer_residual <= effective_tol
            and np.isfinite(shadow_energy)
            and np.isfinite(shadow_mapped_energy)
            and abs(shadow_mapped_energy - shadow_energy)
            <= effective_tol * (1.0 + abs(shadow_energy))
        )
        if shadow_root:
            shadow_scaled = jnp.asarray(shadow_residual) / residual_scale
            shadow_merit = 0.5 * float(
                jnp.vdot(shadow_scaled, shadow_scaled).real
            )
            shadow_step = trust_kernel(
                jnp.asarray(shadow_coordinates),
                residual_scale,
                jnp.asarray(radius, coordinates.dtype),
                jnp.zeros_like(coordinates),
                jnp.asarray(False),
                plan.context,
            )
            outer_map_evaluations += 1
            linear_iterations += int(shadow_step[10])
            deltas.append(shadow_outer_residual)
            energies.append(shadow_energy)
            return OuterSolveResult(
                shadow_mapped,
                OuterTerminationReason.ROOT_CONVERGED,
                candidate_iterations + iteration,
                tuple(deltas),
                tuple(energies),
                shadow_outer_residual,
                shadow_merit,
                float(shadow_step[2]),
                radius,
                accepted + 1,
                rejected,
                float(shadow_inner),
                int(shadow_code),
                outer_map_evaluations,
                trial_evaluations,
                linear_iterations,
                solver_compilations,
            )
        shadow_history_coordinates.append(shadow_coordinates.copy())
        shadow_history_residuals.append(shadow_residual.copy())
        shadow_history_coordinates = shadow_history_coordinates[-6:]
        shadow_history_residuals = shadow_history_residuals[-6:]
        accelerated_step = np.zeros_like(host_coordinates)
        has_accelerated_step = False
        if len(shadow_history_coordinates) >= 2:
            ys = np.stack(shadow_history_coordinates)
            rs = np.stack(shadow_history_residuals)
            delta_y = np.diff(ys, axis=0).T
            delta_r = np.diff(rs, axis=0).T
            scaled_delta_r = delta_r / np.asarray(residual_scale)[:, None]
            rank_threshold = (
                eps * max(scaled_delta_r.shape)
            )
            coefficients = np.linalg.lstsq(
                scaled_delta_r,
                shadow_residual / np.asarray(residual_scale),
                rcond=rank_threshold,
            )[0]
            shadow_step = (
                shadow_residual
                - (delta_y + delta_r) @ coefficients
            )
        else:
            shadow_step = 0.5 * shadow_residual
        shadow_trial = shadow_coordinates + shadow_step
        shadow_valid = (
            int(shadow_code) == 0
            and np.isfinite(float(shadow_inner))
            and np.isfinite(shadow_trial).all()
            and _finite(shadow_mapped)
            and not chart.support_changed(shadow_mapped.messages)
        )
        if shadow_valid:
            shadow_coordinates = shadow_trial
            accelerated_step = shadow_coordinates - host_coordinates
            has_accelerated_step = bool(
                np.isfinite(accelerated_step).all()
            )
        radius_floor = np.sqrt(eps) * max(
            float(jnp.linalg.norm(coordinates)), 1.0
        )
        values = trust_kernel(
            coordinates,
            residual_scale,
            jnp.asarray(radius, coordinates.dtype),
            jnp.asarray(accelerated_step),
            jnp.asarray(has_accelerated_step),
            plan.context,
        )
        outer_map_evaluations += 1
        linear_iterations += int(values[10])
        step = TrustStep(
            values[0], float(values[1]), float(values[2]), bool(values[3]),
            bool(values[4]), bool(values[5]),
        )
        gradient_norm = step.gradient_norm
        gradient_floor = (
            eps
            * max(chart.size, 1)
            * max(float(jnp.linalg.norm(scaled_residual)), 1.0)
        )
        if not step.finite:
            return OuterSolveResult(
                current, OuterTerminationReason.NONFINITE_EVALUATION,
                candidate_iterations + iteration,
                tuple(deltas), tuple(energies), outer_residual,
                merit, gradient_norm, radius, accepted, rejected,
                inner_residual, inner_code,
            )
        if gradient_norm <= gradient_floor:
            return OuterSolveResult(
                current, OuterTerminationReason.STATIONARY_NONROOT,
                candidate_iterations + iteration,
                tuple(deltas), tuple(energies), outer_residual, merit,
                gradient_norm, radius, accepted, rejected,
                inner_residual, inner_code,
            )
        prediction_floor = eps * max(merit, np.finfo(dtype).tiny)
        if step.predicted_reduction <= prediction_floor:
            reason = (
                OuterTerminationReason.LINEAR_SOLVE_BUDGET
                if step.linear_exhausted
                else OuterTerminationReason.STAGNATION
            )
            return OuterSolveResult(
                current, reason, candidate_iterations + iteration,
                tuple(deltas), tuple(energies),
                outer_residual, merit, gradient_norm, radius, accepted,
                rejected, inner_residual, inner_code,
            )

        def candidate(candidate_step, prediction):
            nonlocal trial_evaluations
            trial_evaluations += 1
            candidate_coordinates = coordinates + candidate_step
            candidate_mapped_coordinates, candidate_mapped, candidate_inner, candidate_code = (
                evaluate(candidate_coordinates)
            )
            candidate_arrays = chart_arrays(chart, candidate_coordinates)
            candidate_residual = (
                candidate_mapped_coordinates - candidate_coordinates
            ) / residual_scale
            candidate_merit = 0.5 * float(
                jnp.vdot(candidate_residual, candidate_residual).real
            )
            candidate_energy = energy(candidate_arrays)
            valid = (
                int(candidate_code) == 0
                and np.isfinite(float(candidate_inner))
                and _finite(candidate_arrays)
                and _finite(candidate_mapped)
                and np.isfinite(candidate_merit)
                and np.isfinite(candidate_energy)
                and not infeasible(candidate_arrays)
                and not infeasible(candidate_mapped)
                and not chart.support_changed(candidate_mapped.messages)
            )
            actual_reduction = (
                merit - candidate_merit if valid else -float("inf")
            )
            ratio_value = actual_reduction / prediction
            return (
                candidate_coordinates,
                candidate_arrays,
                candidate_inner,
                candidate_code,
                candidate_merit,
                candidate_energy,
                valid,
                actual_reduction,
                ratio_value,
                float(jnp.linalg.norm(candidate_step)) >= radius,
        )

        trials = [candidate(step.step, step.predicted_reduction)]
        for fixed_candidate, fixed_prediction in zip(
            values[8], values[9]
        ):
            prediction = float(fixed_prediction)
            if (
                prediction >= 0.1 * step.predicted_reduction
                and not np.array_equal(
                    np.asarray(fixed_candidate),
                    np.asarray(step.step),
                )
            ):
                fixed_trial = candidate(fixed_candidate, prediction)
                trials.append(fixed_trial)
        if has_accelerated_step:
            for accelerated_candidate, accelerated_prediction in zip(
                values[6], values[7]
            ):
                prediction = float(accelerated_prediction)
                if (
                    prediction > prediction_floor
                    and not np.array_equal(
                        np.asarray(accelerated_candidate),
                        np.asarray(step.step),
                    )
                ):
                    trials.append(
                        candidate(accelerated_candidate, prediction)
                    )
        acceptable = [
            trial
            for trial in trials
            if trial[6] and trial[8] >= 0.01 and trial[7] > 0.0
        ]
        if acceptable:
            selected = max(acceptable, key=lambda trial: trial[7])
        else:
            selected = max(trials, key=lambda trial: trial[8])
        (
            trial_coordinates,
            trial,
            trial_inner,
            trial_code,
            trial_merit,
            trial_energy,
            trial_valid,
            actual,
            ratio,
            selected_boundary,
        ) = selected
        if ratio < 0.25:
            radius *= 0.25
        elif ratio > 0.75 and (step.hit_boundary or selected_boundary):
            radius = min(2.0 * radius, radius_ceiling)

        if trial_valid and ratio >= 0.01 and actual > 0.0:
            delta = _array_delta(current, trial)
            deltas.append(delta)
            current = trial
            coordinates = trial_coordinates
            energies.append(trial_energy)
            merit = trial_merit
            inner_residual = float(trial_inner)
            inner_code = int(trial_code)
            accepted += 1
        else:
            rejected += 1

        if radius <= radius_floor:
            return OuterSolveResult(
                current, OuterTerminationReason.TRUST_REGION_COLLAPSED,
                candidate_iterations + iteration + 1,
                tuple(deltas), tuple(energies),
                outer_residual, merit, gradient_norm, radius, accepted,
                rejected, inner_residual, inner_code,
                outer_map_evaluations, trial_evaluations,
                linear_iterations, solver_compilations,
            )

    return OuterSolveResult(
        current, OuterTerminationReason.ITERATION_BUDGET,
        candidate_iterations + n_iters,
        tuple(deltas), tuple(energies), outer_residual, merit,
        gradient_norm, radius, accepted, rejected,
        inner_residual, inner_code,
        outer_map_evaluations, trial_evaluations,
        linear_iterations, solver_compilations,
    )

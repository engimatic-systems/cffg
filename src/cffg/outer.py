"""Host control for gauge-reduced mixed-GFE outer inference.

The scientific objective remains the engine's Bethe/generalised free-energy
ledger. The fixed-point residual is a numerical merit for trust-step acceptance.
Coordinate and derivative-kernel exports retain their existing import paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from cffg.compiled import Arena
from cffg.failure_facts import FailureAt, FailureFact, empty_failure_at
from cffg.outer_chart import (
    OuterArrays as OuterArrays,
    OuterChart as OuterChart,
    _CategoricalBlock as _CategoricalBlock,
    _PrecisionBlock as _PrecisionBlock,
    _VectorBlock as _VectorBlock,
    _linear_map as _linear_map,
    _unit_scaled as _unit_scaled,
    build_outer_chart as build_outer_chart,
    chart_arrays as chart_arrays,
    chart_coordinates as chart_coordinates,
    outer_chart_signature as outer_chart_signature,
)
from cffg.outer_evidence import map_evidence, offset_failure, terminal_failure
from cffg.outer_structure import OuterStructure
from cffg.outer_trust import (
    OuterKernelPlan as OuterKernelPlan,
    TrustStep as TrustStep,
    _boundary_distance as _boundary_distance,
    _make_root_diagnostics as _make_root_diagnostics,
    _make_trust_kernel as _make_trust_kernel,
)
from cffg.runtime_types import OuterTerminationReason as OuterTerminationReason


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
    failure: FailureAt = field(default_factory=empty_failure_at)


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
    arena: Arena | OuterStructure,
    outer_map: Callable[
        [OuterArrays], tuple[OuterArrays, jax.Array, jax.Array]
        | tuple[OuterArrays, jax.Array, jax.Array, FailureFact]
    ],
    energy: Callable[[OuterArrays], float],
    infeasible: Callable[[OuterArrays], bool],
    *,
    n_iters: int,
    tol: float,
    eliminated: dict[str, frozenset[int]] | None = None,
    kernel_factory: Callable[[OuterChart], OuterKernelPlan] | None = None,
    candidate_limit: int = 128,
    adaptive_gaussian: bool = False,
    initial_failure: FailureAt | None = None,
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
    discovery_prefix = initial_failure or empty_failure_at()

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
            mapped, local_residual, local_code, fact = map_evidence(outer_map(
                chart_arrays(chart, value)
            ))
            return chart_coordinates(chart, mapped), mapped, local_residual, local_code, fact

        mapped_coordinates, mapped, local_residual, local_code, fact = evaluate(
            coordinates
        )
        inner_residual = float(local_residual)
        inner_code = int(local_code)
        discovery_failure = (
            discovery_prefix if int(discovery_prefix.fact.reason) != 0
            else terminal_failure(fact, 1)
        )
        if inner_code:
            return OuterSolveResult(
                current, OuterTerminationReason.INNER_FAILURE, 0,
                tuple(deltas), tuple(energies), outer_residual, merit,
                gradient_norm, radius, accepted, rejected,
                inner_residual, inner_code,
                failure=discovery_failure,
            )
        if not _finite(current) or not _finite(mapped):
            return OuterSolveResult(
                current, OuterTerminationReason.NONFINITE_EVALUATION, 0,
                tuple(deltas), tuple(energies), float("inf"), float("inf"),
                float("inf"), radius, accepted, rejected,
                inner_residual, inner_code,
                failure=discovery_failure,
            )
        if infeasible(current) or infeasible(mapped):
            return OuterSolveResult(
                current, OuterTerminationReason.INFEASIBLE, 0,
                tuple(deltas), tuple(energies), float("inf"), float("inf"),
                float("inf"), radius, accepted, rejected,
                inner_residual, inner_code,
                failure=discovery_failure,
            )
        if adaptive_gaussian and not chart.proper_reference:
            # Admission happens once, in the budgeted startup program.
            return OuterSolveResult(
                current, OuterTerminationReason.INNER_FAILURE, 0,
                tuple(deltas), tuple(energies), outer_residual, merit,
                gradient_norm, radius, accepted, rejected,
                float("inf"), 2, failure=discovery_failure,
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
                    failure=discovery_failure,
                )
            current = mapped
            discovery_prefix = empty_failure_at()
            continue
        break

    residual_fn = lambda value: evaluate(value)[0] - value
    initial_residual = mapped_coordinates - coordinates
    residual_scale = chart.residual_scale(initial_residual)
    if adaptive_gaussian:
        # Gaussian whitening already fixes units; initial residuals must not reweight equations.
        residual_scale = jnp.ones_like(initial_residual)
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
                return OuterSolveResult(
                    current, OuterTerminationReason.ROOT_CONVERGED,
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
        mapped, local_residual, local_code, fact = map_evidence(outer_map(current))
        outer_map_evaluations += 1
        inner_residual = float(local_residual)
        inner_code = int(local_code)
        if inner_code:
            return OuterSolveResult(
                current, OuterTerminationReason.INNER_FAILURE,
                candidate_iterations, tuple(deltas), tuple(energies),
                outer_residual, merit, gradient_norm, radius, accepted,
                rejected, inner_residual, inner_code,
                failure=terminal_failure(fact, candidate_iterations + 1),
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
                failure=terminal_failure(fact, candidate_iterations + 1),
            )
        if infeasible(mapped):
            return OuterSolveResult(
                current, OuterTerminationReason.INFEASIBLE,
                candidate_iterations, tuple(deltas), tuple(energies),
                float("inf"), float("inf"), float("inf"), radius,
                accepted, rejected, inner_residual, inner_code,
                failure=terminal_failure(fact, candidate_iterations + 1),
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
                    failure=terminal_failure(fact, candidate_iterations + 1),
                )
            restarted = solve_outer(
                mapped, arena, outer_map, energy, infeasible,
                n_iters=max(n_iters - candidate_iterations, 0),
                tol=tol,
                eliminated=eliminated,
                kernel_factory=kernel_factory,
                candidate_limit=candidate_limit,
                adaptive_gaussian=adaptive_gaussian,
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
                offset_failure(restarted.failure, candidate_iterations),
            )

    n_iters -= candidate_iterations
    chart = build_outer_chart(arena, current, eliminated)
    coordinates = chart_coordinates(chart, current)

    def evaluate(value):
        nonlocal outer_map_evaluations
        outer_map_evaluations += 1
        mapped, local_residual, local_code, fact = map_evidence(outer_map(
            chart_arrays(chart, value)
        ))
        return chart_coordinates(chart, mapped), mapped, local_residual, local_code, fact

    residual_fn = lambda value: evaluate(value)[0] - value
    mapped_coordinates, mapped, local_residual, local_code, fact = evaluate(coordinates)
    inner_residual = float(local_residual)
    inner_code = int(local_code)
    initial_residual = mapped_coordinates - coordinates
    residual_scale = chart.residual_scale(initial_residual)
    if adaptive_gaussian:
        # Whitened coordinates are dimensionless; initially exact rows retain unit weight.
        residual_scale = jnp.maximum(residual_scale, 1.0)
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
        mapped_coordinates, mapped, local_residual, local_code, fact = evaluate(
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
                failure=terminal_failure(fact, candidate_iterations + iteration + 1),
            )
        if not _finite(current) or not _finite(mapped):
            return OuterSolveResult(
                current, OuterTerminationReason.NONFINITE_EVALUATION,
                candidate_iterations + iteration,
                tuple(deltas), tuple(energies), float("inf"), float("inf"),
                float("inf"), radius, accepted, rejected,
                inner_residual, inner_code,
                failure=terminal_failure(fact, candidate_iterations + iteration + 1),
            )
        if infeasible(current) or infeasible(mapped):
            return OuterSolveResult(
                current, OuterTerminationReason.INFEASIBLE,
                candidate_iterations + iteration,
                tuple(deltas), tuple(energies), float("inf"), float("inf"),
                float("inf"), radius, accepted, rejected,
                inner_residual, inner_code,
                failure=terminal_failure(fact, candidate_iterations + iteration + 1),
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
                    failure=terminal_failure(fact, candidate_iterations + iteration + 1),
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
                adaptive_gaussian=adaptive_gaussian,
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
                offset_failure(restarted.failure, candidate_iterations + iteration),
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
            _, root_gradient = plan.root_diagnostics(
                coordinates,
                residual_scale,
                plan.context,
            )
            outer_map_evaluations += 1
            gradient_norm = float(root_gradient)
            deltas.append(outer_residual)
            if not energies or energies[-1] != current_energy:
                energies.append(current_energy)
            return OuterSolveResult(
                current, OuterTerminationReason.ROOT_CONVERGED,
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
        shadow_mapped_coordinates, shadow_mapped, shadow_inner, shadow_code, _ = (
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
            _, shadow_gradient = plan.root_diagnostics(
                jnp.asarray(shadow_coordinates),
                residual_scale,
                plan.context,
            )
            outer_map_evaluations += 1
            deltas.append(shadow_outer_residual)
            energies.append(shadow_energy)
            return OuterSolveResult(
                shadow_arrays,
                OuterTerminationReason.ROOT_CONVERGED,
                candidate_iterations + iteration,
                tuple(deltas),
                tuple(energies),
                shadow_outer_residual,
                shadow_merit,
                float(shadow_gradient),
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
        if not (
            int(shadow_code) == 0
            and np.isfinite(float(shadow_inner))
            and np.isfinite(shadow_residual).all()
            and np.isfinite(shadow_energy)
            and np.isfinite(shadow_mapped_energy)
        ):
            # A rejected auxiliary proposal cannot enter the least-squares history.
            shadow_coordinates = host_coordinates.copy()
            shadow_residual = np.asarray(residual).copy()
            shadow_history_coordinates.clear()
            shadow_history_residuals.clear()
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
            (candidate_mapped_coordinates, candidate_mapped, candidate_inner,
             candidate_code, candidate_fact) = evaluate(candidate_coordinates)
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
                candidate_fact,
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
            trial_fact,
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
                failure=(
                    terminal_failure(trial_fact, candidate_iterations + iteration + 1)
                    if not any(trial[6] for trial in trials)
                    else empty_failure_at()
                ),
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

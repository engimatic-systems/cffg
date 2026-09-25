"""Compiled matrix-free trust steps and residual-gradient diagnostics."""

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


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

"""Bounded, fully adaptive proposals; the existing root/energy tests decide success."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from cffg.outer_chart import chart_arrays, chart_coordinates


class ProposalResult(NamedTuple):
    coordinates: jax.Array
    rounds: jax.Array
    certified: jax.Array
    residual: jax.Array
    energy: jax.Array
    accepted: jax.Array
    rejected: jax.Array
    count: jax.Array
    deltas: jax.Array
    energies: jax.Array
    inner_residual: jax.Array
    radius: jax.Array


class _ProposalCarry(NamedTuple):
    trial: jax.Array
    retained: jax.Array
    history: jax.Array
    residuals: jax.Array
    rounds: jax.Array
    count: jax.Array
    certified: jax.Array
    root_residual: jax.Array
    energy: jax.Array
    accepted: jax.Array
    rejected: jax.Array
    radius: jax.Array
    deltas: jax.Array
    energies: jax.Array
    inner_residual: jax.Array


def _anderson(history, residuals, current, residual, count, newest, radius):
    """Constrained least squares in differences; rank guards change directions only."""
    active = (jnp.arange(len(history)) < count) & (jnp.arange(len(history)) != newest)
    differences = jnp.where(active[:, None], residuals - residual, 0.)
    scale = jnp.max(jnp.abs(differences), initial=0.)
    denominator = jnp.where(scale > 0., scale, 1.)
    u, singular, vt = jnp.linalg.svd((differences / denominator).T, full_matrices=False)
    eps = jnp.finfo(current.dtype).eps
    largest = jnp.max(singular, initial=0.)
    cutoff = eps * max(differences.shape) * largest
    # The scaled least-squares direction has no information-matrix ridge.
    squared = singular * singular + eps * largest * largest
    gain = jnp.where(singular > cutoff, singular / jnp.where(squared > 0., squared, 1.), 0.)
    coefficients = -(vt.T @ (gain * (u.T @ (residual / denominator))))
    coefficients = jnp.where(active, coefficients, 0.)
    step = .5 * residual + coefficients @ (history - current + .5 * differences)
    finite = jnp.isfinite(step).all()
    step = jnp.where(finite, step, .5 * residual)
    length = jnp.linalg.norm(step)
    fraction = jnp.minimum(1., radius / jnp.where(length > 0., length, 1.))
    return current + fraction * step


def make_adaptive_proposals(program, belief_kernel, history_size=6):
    """Compile proposals in one fixed chart; refresh every nonlinear map."""
    def finite(arrays):
        return jnp.all(jnp.stack([jnp.isfinite(value).all() for value in arrays]))

    @jax.jit
    def run(initial, context, budget, tolerance):
        history = jnp.zeros((history_size, len(initial)), dtype=initial.dtype)
        residuals = jnp.zeros_like(history)
        # Carry a separately retained admissible point when a proposal is rejected.
        initial_carry = _ProposalCarry(initial, initial, history, residuals, jnp.int32(0), jnp.int32(0),
                         jnp.asarray(False), jnp.asarray(jnp.inf, initial.dtype),
                         jnp.asarray(jnp.inf, initial.dtype), jnp.int32(0), jnp.int32(0),
                         jnp.asarray(1., initial.dtype),
                         jnp.zeros(128, initial.dtype), jnp.zeros(128, initial.dtype),
                         jnp.asarray(jnp.inf, initial.dtype))

        def condition(carry):
            return (carry.rounds < budget) & ~carry.certified

        def body(carry):
            (trial, retained, history, residuals, rounds, count, _, _, _, accepted,
             rejected, radius, deltas, energies, _) = carry
            arrays, reconstruction_fact = program.reconstruct_arrays(
                chart_arrays(program.chart, trial, context.base, context.metrics), context)
            mapped, local_residual, status, map_fact = program.map_arrays(arrays, context)
            successor, successor_fact = program.reconstruct_arrays(mapped, context)
            mapped_coordinates = chart_coordinates(program.chart, mapped, context.metrics)
            residual = mapped_coordinates - trial
            root_residual = jnp.max(jnp.abs(residual) / (1. + jnp.abs(trial)), initial=0.)
            energy = program.array_energy(arrays, context, belief_kernel)
            valid = (finite(arrays) & finite(mapped) & finite(successor) & jnp.isfinite(energy)
                     & jnp.isfinite(trial).all() & jnp.isfinite(residual).all()
                     & jnp.isfinite(root_residual)
                     & jnp.isfinite(local_residual) & (status == 0)
                     & (reconstruction_fact.reason == 0) & (map_fact.reason == 0)
                     & (successor_fact.reason == 0))

            def certify(_):
                successor_energy = program.array_energy(successor, context, belief_kernel)
                canonical, canonical_fact = program.reconstruct_arrays(
                    chart_arrays(program.chart, mapped_coordinates, context.base, context.metrics), context)
                canonical_energy = program.array_energy(canonical, context, belief_kernel)
                threshold = tolerance * (1. + jnp.abs(energy))
                return (finite(canonical) & (canonical_fact.reason == 0)
                        & jnp.isfinite(successor_energy) & jnp.isfinite(canonical_energy)
                        & (jnp.abs(successor_energy - energy) <= threshold)
                        & (jnp.abs(canonical_energy - energy) <= threshold))

            certified = jax.lax.cond(valid & (root_residual <= tolerance), certify,
                                     lambda _: jnp.asarray(False), operand=None)
            newest = count % history_size
            history = jax.lax.cond(valid, lambda h: h.at[newest].set(trial), lambda h: h, history)
            residuals = jax.lax.cond(valid, lambda h: h.at[newest].set(residual), lambda h: h, residuals)
            next_count = count + valid.astype(jnp.int32)
            deltas = jax.lax.cond(valid, lambda a: a.at[count].set(root_residual), lambda a: a, deltas)
            energies = jax.lax.cond(valid, lambda a: a.at[count].set(energy), lambda a: a, energies)
            initial_radius = jnp.linalg.norm(residual)
            initial_radius = jnp.where(jnp.isfinite(initial_radius), jnp.maximum(initial_radius, 1.), 1.)
            radius = jnp.where(rounds == 0, initial_radius, radius)

            def propose(_):
                return _anderson(history, residuals, trial, residual,
                                 jnp.minimum(next_count, history_size), newest, radius)

            next_trial = jax.lax.cond(
                certified, lambda _: trial,
                lambda _: jax.lax.cond(valid, propose, lambda _: retained + .5 * (trial - retained), operand=None),
                operand=None)
            return _ProposalCarry(next_trial, jnp.where(valid, trial, retained), history, residuals, rounds + 1,
                    next_count, certified, root_residual, energy,
                    accepted + (valid & (rounds > 0)).astype(jnp.int32), rejected + (~valid).astype(jnp.int32),
                    radius, deltas, energies, local_residual)

        value = jax.lax.while_loop(condition, body, initial_carry)
        return ProposalResult(value.retained, value.rounds, value.certified, value.root_residual,
                              value.energy, value.accepted, value.rejected, value.count,
                              value.deltas, value.energies, value.inner_residual, value.radius)

    return run

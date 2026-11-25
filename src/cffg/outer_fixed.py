"""Fixed-support trust iterations with a host handoff at certification boundaries."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from cffg.compiled import NEG
from cffg.outer_candidate import array_delta, finite
from cffg.outer_chart import chart_arrays, chart_coordinates


class Progress(NamedTuple):
    arrays: object
    value: jax.Array
    shadow: jax.Array
    points: jax.Array
    residuals: jax.Array
    history: jax.Array
    radius: jax.Array
    norm: jax.Array
    merit: jax.Array
    gradient: jax.Array
    inner: jax.Array
    accepted: jax.Array
    rejected: jax.Array
    maps: jax.Array
    trials: jax.Array
    linear: jax.Array
    deltas: jax.Array
    energies: jax.Array
    advance: jax.Array


class Evaluation(NamedTuple):
    residual: jax.Array
    inner: jax.Array
    valid: jax.Array
    support_known: jax.Array
    energy: jax.Array
    mapped_energy: jax.Array


def _anderson(points, residuals, count, scale):
    """Use the last six observations, with zero columns for unused differences."""
    delta_y = jnp.diff(points, axis=0).T
    delta_r = jnp.diff(residuals, axis=0).T
    active = jnp.arange(5) >= 6 - count
    matrix = jnp.where(active, delta_r / scale[:, None], 0.)
    u, singular, vt = jnp.linalg.svd(matrix, full_matrices=False)
    cutoff = jnp.finfo(scale.dtype).eps * jnp.maximum(scale.size, count - 1) * singular[0]
    present = singular > cutoff
    coefficients = vt.T @ jnp.where(present, (u.T @ (residuals[-1] / scale)) /
                                    jnp.where(present, singular, 1.), 0.)
    displacement = jnp.where(active, delta_y + delta_r, 0.) @ coefficients
    return residuals[-1] - displacement


class TrialSelection(NamedTuple):
    index: jax.Array
    ratio: jax.Array
    reduction: jax.Array
    merit: jax.Array
    energy: jax.Array
    inner: jax.Array
    accepted: jax.Array
    support_known: jax.Array
    count: jax.Array


def _select_trial(evaluate, value, scale, merit, base, moves, predictions, mask):
    def compare(i, best):
        def assess():
            trial = evaluate(value + moves[i])
            scaled = trial.residual / scale
            trial_merit = .5 * jnp.vdot(scaled, scaled).real
            valid = trial.valid & jnp.isfinite(trial_merit) & jnp.isfinite(trial.energy)
            reduction = jnp.where(valid, merit - trial_merit, -jnp.inf)
            ratio = reduction / predictions[i]
            accept = valid & (ratio >= .01) & (reduction > 0.)
            # First maximum preserves the same proposal order and tie rule as the host.
            better = ((accept & (~best.accepted | (reduction > best.reduction)))
                      | (~accept & ~best.accepted & (ratio > best.ratio)) | (i == 0))
            return TrialSelection(
                jnp.where(better, jnp.int32(i), best.index), jnp.where(better, ratio, best.ratio),
                jnp.where(better, reduction, best.reduction), jnp.where(better, trial_merit, best.merit),
                jnp.where(better, trial.energy, best.energy), jnp.where(better, trial.inner, best.inner),
                best.accepted | accept, best.support_known & trial.support_known, best.count + 1,
            )
        return jax.lax.cond(mask[i], assess, lambda: best)

    initial = TrialSelection(jnp.int32(0), -jnp.inf, -jnp.inf, merit, base.energy, base.inner,
                             False, True, jnp.int32(0))
    return jax.lax.fori_loop(0, moves.shape[0], compare, initial)


def make_fixed_controller(program, trust, ledger, capacity):
    chart = program.chart

    def run(current, value, scale, radius, ceiling, inner, context, budget, tol):
        dtype = value.dtype
        eps = jnp.finfo(dtype).eps
        support = program.reconstruct_arrays(current, context)[0].messages != NEG
        face = jnp.asarray([index for block in chart.categorical for index in block.segment],
                           dtype=jnp.int32)
        active = context.base.messages[face] != NEG

        def evaluate(value):
            raw = chart_arrays(chart, value, context.base, context.metrics)
            arrays = program.reconstruct_arrays(raw, context)[0]
            mapped, residual, status, _ = program.map_arrays(arrays, context)
            mapped_value = chart_coordinates(chart, mapped, context.metrics)
            rebuilt = program.reconstruct_arrays(mapped, context)[0]
            known = (jnp.all((arrays.messages != NEG) == support)
                     & jnp.all((rebuilt.messages != NEG) == support))
            valid = (finite(raw) & finite(mapped) & jnp.isfinite(residual) & (status == 0)
                     & jnp.all((mapped.messages[face] != NEG) == active))
            return Evaluation(mapped_value - value, residual, valid, known,
                              program.array_energy(arrays, context, ledger),
                              program.array_energy(rebuilt, context, ledger))

        def root_candidate(ev, value):
            norm = jnp.max(jnp.abs(ev.residual) / (1. + jnp.abs(value)), initial=0.)
            stationary = (jnp.isfinite(ev.energy) & jnp.isfinite(ev.mapped_energy)
                          & (jnp.abs(ev.energy - ev.mapped_energy) <= tol * (1. + jnp.abs(ev.energy))))
            return norm, (norm <= tol) & stationary

        def iteration(row):
            base = evaluate(row.value)
            norm, root = root_candidate(base, row.value)
            # Existing terminal policy independently classifies and certifies the returned point.
            stop = row._replace(advance=False, maps=row.maps + 1)

            def propose():
                shadow = evaluate(row.shadow)
                _, shadow_root = root_candidate(shadow, row.shadow)
                # A changed face, inner failure, or contaminated history goes back
                # to the reference policy before any change of accepted coordinates.
                shadow_ready = (shadow.valid & shadow.support_known
                                & jnp.isfinite(shadow.energy) & jnp.isfinite(shadow.mapped_energy)
                                & jnp.all(jnp.isfinite(shadow.residual)))
                shadow_stop = stop._replace(maps=stop.maps + 1)

                def trust_iteration():
                    points = jnp.roll(row.points, -1, axis=0).at[-1].set(row.shadow)
                    residuals = jnp.roll(row.residuals, -1, axis=0).at[-1].set(shadow.residual)
                    count = jnp.minimum(row.history + 1, 6)
                    shadow_step = jax.lax.cond(
                        count > 1, lambda: _anderson(points, residuals, count, scale),
                        lambda: .5 * shadow.residual,
                    )
                    following = row.shadow + shadow_step
                    have_accel = jnp.all(jnp.isfinite(following))
                    following = jnp.where(have_accel, following, row.shadow)
                    acceleration = jnp.where(have_accel, following - row.value, 0.)
                    values = trust(row.value, scale, row.radius, acceleration, have_accel, context)
                    step, prediction, gradient, boundary, _, finite_step = values[:6]
                    scaled = base.residual / scale
                    merit = 0.5 * jnp.vdot(scaled, scaled).real
                    floor = eps * jnp.maximum(merit, jnp.finfo(dtype).tiny)
                    gradient_floor = eps * max(chart.size, 1) * jnp.maximum(jnp.linalg.norm(scaled), 1.)
                    ready = finite_step & (gradient > gradient_floor) & (prediction > floor)
                    handoff = shadow_stop._replace(maps=shadow_stop.maps + 1,
                                                   linear=row.linear + values[10])

                    def trials():
                        moves = jnp.concatenate((step[None], values[8], values[6]))
                        predictions = jnp.concatenate((prediction[None], values[9], values[7]))
                        different = jnp.any(moves != step, axis=1)
                        mask = jnp.concatenate((jnp.ones(1, bool), values[9] >= .1 * prediction,
                                                (values[7] > floor) & have_accel)) & different
                        mask = mask.at[0].set(True)

                        best = _select_trial(evaluate, row.value, scale, merit, base, moves, predictions, mask)
                        ratio, accept = best.ratio, best.accepted
                        trial = row.value + moves[best.index]
                        arrays = chart_arrays(chart, trial, context.base, context.metrics)
                        next_radius = jnp.where(ratio < .25, .25 * row.radius,
                            jnp.where((ratio > .75) & (boundary | (jnp.linalg.norm(moves[best.index]) >= row.radius)),
                                      jnp.minimum(2. * row.radius, ceiling), row.radius))
                        radius_floor = jnp.sqrt(eps) * jnp.maximum(jnp.linalg.norm(row.value), 1.)
                        stop_trials = handoff._replace(maps=handoff.maps + best.count, trials=row.trials + best.count)
                        advanced = stop_trials._replace(
                            arrays=jax.tree.map(lambda new, old: jnp.where(accept, new, old), arrays, row.arrays),
                            value=jnp.where(accept, trial, row.value), shadow=following,
                            points=points, residuals=residuals, history=count, radius=next_radius,
                            norm=norm, merit=jnp.where(accept, best.merit, merit), gradient=gradient,
                            inner=jnp.where(accept, best.inner, base.inner),
                            deltas=row.deltas.at[row.accepted].set(array_delta(row.arrays, arrays)),
                            energies=row.energies.at[row.accepted].set(best.energy),
                            accepted=row.accepted + accept, rejected=row.rejected + ~accept, advance=True,
                        )
                        return jax.lax.cond(best.support_known & (next_radius > radius_floor),
                                            lambda: advanced, lambda: stop_trials)
                    return jax.lax.cond(ready, trials, lambda: handoff)
                return jax.lax.cond(shadow_ready & ~shadow_root, trust_iteration, lambda: shadow_stop)
            return jax.lax.cond(base.valid & base.support_known & ~root, propose, lambda: stop)

        zeros = jnp.zeros((6, chart.size), dtype)
        trace = jnp.zeros(capacity, dtype)
        zero = jnp.int32(0)
        inf = jnp.asarray(jnp.inf, dtype)
        initial = Progress(current, value, value, zeros, zeros, zero, radius, inf, inf, inf, inner,
                           zero, zero, zero, zero, zero, trace, trace, True)
        return jax.lax.while_loop(lambda row: row.advance & (row.accepted + row.rejected < budget),
                                  iteration, initial)
    return run

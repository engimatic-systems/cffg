"""Host admission and accounting for certificate-gated adaptive proposals."""

from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np

from cffg.outer import OuterSolveResult
from cffg.outer_chart import OuterArrays, OuterChart, build_outer_chart, chart_arrays, chart_coordinates


@dataclass(frozen=True)
class ProposalAttempt:
    arrays: OuterArrays
    result: OuterSolveResult | None
    rounds: int
    accepted: int
    rejected: int
    maps: int
    compilations: int
    chart: OuterChart
    message_deltas: tuple[float, ...]
    free_energies: tuple[float, ...]


def certify_adaptive_proposal(attempt, outer_map, energy, infeasible, tolerance):
    """Repeat the original host certificate before accepting a compiled proposal."""
    if attempt.result is None:
        return attempt
    arrays = attempt.arrays
    mapped, inner, code, fact = outer_map(arrays)
    coordinates = chart_coordinates(attempt.chart, arrays)
    residual = chart_coordinates(attempt.chart, mapped) - coordinates
    root = float(jnp.max(jnp.abs(residual) / (1. + jnp.abs(coordinates)), initial=0.))
    current_energy, successor_energy = energy(arrays), energy(mapped)
    finite = all(np.isfinite(np.asarray(value)).all() for value in (*arrays, *mapped))
    valid = (finite and int(code) == 0 and int(fact.reason) == 0 and np.isfinite(float(inner))
             and not infeasible(arrays) and not infeasible(mapped)
             and attempt.chart.proper_reference and root <= tolerance
             and np.isfinite(current_energy) and np.isfinite(successor_energy)
             and abs(successor_energy - current_energy) <= tolerance * (1. + abs(current_energy)))
    solved = None
    if valid:
        solved = replace(attempt.result, outer_residual=root,
                         free_energies=(*attempt.result.free_energies[:-1], current_energy),
                         outer_map_evaluations=attempt.maps + 1)
    return replace(attempt, result=solved, maps=attempt.maps + 1)


def try_adaptive_proposals(structure, eliminated, provider, state, hard_blocks, budget, tolerance):
    """No speculation before proper startup, or in functional/discrete domains."""
    proposal_budget = min(budget - max(8, budget // 8), 128)
    if state.messages.size or state.ef_eta.size or proposal_budget <= 0:
        return None
    arrays = OuterArrays(state.messages, state.g_eta, state.g_lam, state.ef_eta)
    chart = build_outer_chart(structure, arrays, eliminated)
    if not chart.proper_reference:
        return None
    coordinates = chart_coordinates(chart, arrays)
    if not np.isfinite(np.asarray(coordinates)).all():
        return None
    plan = provider(chart, state, hard_blocks)
    if plan.proposal is None:
        return None
    proposal = plan.proposal(coordinates, plan.context, jnp.int32(proposal_budget),
                             jnp.asarray(tolerance, dtype=coordinates.dtype))
    rounds, count = int(proposal.rounds), int(proposal.count)
    deltas = tuple(float(value) for value in np.asarray(proposal.deltas)[:count])
    energies = tuple(float(value) for value in np.asarray(proposal.energies)[:count])
    retained = chart_arrays(chart, proposal.coordinates) if count else arrays
    maps = rounds
    solved = None
    if bool(proposal.certified):
        merit, gradient = plan.root_diagnostics(proposal.coordinates, jnp.ones_like(coordinates), plan.context)
        maps += 1
        if np.isfinite(float(merit)) and np.isfinite(float(gradient)):
            from cffg.runtime_types import OuterTerminationReason
            solved = OuterSolveResult(
                retained, OuterTerminationReason.ROOT_CONVERGED, rounds,
                deltas, energies,
                float(proposal.residual), float(merit), float(gradient), float(proposal.radius),
                int(proposal.accepted), int(proposal.rejected), float(proposal.inner_residual), 0,
                maps, max(rounds - 1, 0), 0, int(plan.compiled),
            )
    return ProposalAttempt(retained, solved, rounds, int(proposal.accepted), int(proposal.rejected),
                           maps, int(plan.compiled), chart, deltas, energies)

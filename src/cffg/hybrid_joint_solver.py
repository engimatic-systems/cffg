"""Bounded matrix-free numerical policy for the unbound complete D12 system.

Only accepted residual evaluations are linearized. The optimizer uses a fixed
while-loop state, no dense Jacobian, restart basis, or retained iteration tape.
A stationary nonzero residual is exhaustion, never a convergence certificate.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from cffg import hybrid_joint as joint
from cffg.hybrid_natural import Status, failure


class Outcome(IntEnum):
    CONVERGED = 0
    EXHAUSTED = 1
    DIVERGED = 2
    RUNNING = 3


class Reason(IntEnum):
    CERTIFIED = 0
    OUTER_BUDGET = 1
    TRIAL_BUDGET = 2
    INVALID_INITIALIZATION = 3
    INVALID_ACCEPTED = 4
    INVALID_DERIVATIVE = 5
    STATIONARY_RESIDUAL = 6


class Rejection(IntEnum):
    NONE = 0
    NUMERICAL = 1
    XI_BOUNDARY = 2
    INSUFFICIENT_DECREASE = 3


@dataclass(frozen=True)
class Budgets:
    """Explicit finite caps. Changes used in checkpoints must be predeclared."""

    outer: int = 200
    trials: int = 24
    linear: int = 128
    backtrack: float = .5
    armijo: float = .0001

    def __post_init__(self):
        if any(type(n) is not int or n < 0 for n in (self.outer, self.trials, self.linear)):
            raise ValueError("iteration budgets must be nonnegative integers")
        if not 0 < self.backtrack < 1 or not 0 < self.armijo < 1:
            raise ValueError("backtracking and Armijo fractions must be in (0,1)")


class LinearState(NamedTuple):
    iteration: jax.Array
    step: jax.Array
    residual: jax.Array
    normal_residual: jax.Array
    direction: jax.Array
    done: jax.Array
    finite: jax.Array


class TrialState(NamedTuple):
    trial: jax.Array
    coordinate: jax.Array
    evaluation: joint.Evaluation
    accepted: jax.Array
    evaluations: jax.Array
    rejection_counts: jax.Array
    last_rejection: jax.Array


class State(NamedTuple):
    coordinate: jax.Array
    evaluation: joint.Evaluation
    iterations: jax.Array
    evaluations: jax.Array
    rejected_trials: jax.Array
    linear_iterations: jax.Array
    gauss_newton_steps: jax.Array
    relative_energy_change: jax.Array
    rejection_counts: jax.Array
    last_rejection: jax.Array
    status: jax.Array
    reason: jax.Array


def tolerance(dtype):
    return 64 * jnp.finfo(dtype).eps ** .75


def _integer(value):
    return jnp.asarray(value, jnp.int32)


def cgls(residual, forward, transpose, size, budget):
    """Capped least squares in N coordinates with M residuals, no restart basis.

    CGLS stores three N vectors and one M vector. Its stopping condition only
    ends the linear subproblem; nonlinear convergence uses all certificates.
    """
    rhs = -residual
    normal = transpose(rhs)
    normal_norm = jnp.linalg.norm(normal)
    target = jnp.sqrt(jnp.finfo(residual.dtype).eps) * normal_norm
    finite = jnp.all(jnp.isfinite(normal)) & jnp.all(jnp.isfinite(rhs)) & jnp.isfinite(normal_norm)
    state = LinearState(_integer(0), jnp.zeros((size,), residual.dtype), rhs, normal, normal,
                        normal_norm <= target, finite)

    def body(state):
        image = forward(state.direction)
        curvature = jnp.vdot(image, image).real
        gamma = jnp.vdot(state.normal_residual, state.normal_residual).real
        # A zero image has no resolvable least-squares curvature; do not ridge it.
        resolved = (curvature > 0) & jnp.isfinite(curvature)
        alpha = gamma / jnp.where(resolved, curvature, 1)
        step = state.step + alpha * state.direction
        residual = state.residual - alpha * image
        normal = transpose(residual)
        beta = jnp.vdot(normal, normal).real / jnp.where(gamma > 0, gamma, 1)
        direction = normal + beta * state.direction
        finite = (jnp.all(jnp.isfinite(step)) & jnp.all(jnp.isfinite(residual))
                  & jnp.all(jnp.isfinite(normal)) & jnp.all(jnp.isfinite(direction)))
        done = ~resolved | (jnp.linalg.norm(normal) <= target)
        return LinearState(state.iteration + 1,
                           jnp.where(resolved & finite, step, state.step),
                           jnp.where(resolved & finite, residual, state.residual),
                           jnp.where(resolved & finite, normal, state.normal_residual),
                           jnp.where(resolved & finite, direction, state.direction), done,
                           state.finite & finite & jnp.all(jnp.isfinite(image)))

    return jax.lax.while_loop(lambda s: (s.iteration < budget) & ~s.done & s.finite, body, state)


def gauss_newton(coordinate, parameters, sites, vector, evaluation, budgets):
    """Matrix-free complete-residual linear step at a proper accepted point."""
    coordinate_scales = 1 + abs(vector)
    residual, forward, transpose = joint.linearize(coordinate, parameters, sites, vector,
                                                 evaluation.scales, coordinate_scales)
    result = cgls(residual, forward, transpose, coordinate.size, budgets.linear)
    return coordinate_scales * result.step, result


def _merit(evaluation, scales):
    scaled = evaluation.residual / scales
    return jnp.vdot(scaled, scaled).real / 2


def line_search(coordinate, parameters, sites, vector, evaluation, direction, budgets):
    """Reject improper/invalid trials; every trial uses the accepted scales."""
    xi, delta = vector[coordinate.message_size:], direction[coordinate.message_size:]
    fractions = jnp.where(delta < 0, -xi / jnp.where(delta < 0, delta, -1), 1)
    boundary = jnp.minimum(1, jnp.min(fractions, initial=1))
    merit = _merit(evaluation, evaluation.scales)
    initial = TrialState(_integer(0), vector, evaluation, jnp.asarray(False), _integer(0),
                         jnp.zeros((len(Rejection),), jnp.int32), _integer(Rejection.NONE))

    def body(state):
        fraction = boundary * budgets.backtrack ** state.trial
        proposal = vector + fraction * direction
        candidate = joint.evaluate(coordinate, parameters, sites, proposal)
        trial_merit = _merit(candidate, evaluation.scales)
        numerical = (candidate.status == 0) & jnp.isfinite(trial_merit)
        feasible = (fraction > 0) & jnp.all(proposal[coordinate.message_size:] >= 0)
        moving = jnp.any(proposal != vector)
        accepted = numerical & feasible & moving & (trial_merit <= (1 - budgets.armijo * fraction) * merit)
        reason = jnp.where(~feasible, _integer(Rejection.XI_BOUNDARY),
                           jnp.where(~numerical, _integer(Rejection.NUMERICAL),
                                     _integer(Rejection.INSUFFICIENT_DECREASE)))
        return TrialState(state.trial + 1, jnp.where(accepted, proposal, state.coordinate),
                          jax.tree.map(lambda new, old: jnp.where(accepted, new, old), candidate, state.evaluation),
                          accepted, state.evaluations + 1,
                          state.rejection_counts.at[reason].add((~accepted).astype(jnp.int32)),
                          jnp.where(accepted, state.last_rejection, reason))

    return jax.lax.while_loop(lambda s: (s.trial < budgets.trials) & ~s.accepted, body, initial)


def initial_state(coordinate, parameters, sites, initial_xi=0., *, perturb=False):
    vector, value = joint.initialize(coordinate, parameters, sites, initial_xi, perturb=perturb)
    value = value._replace(status=value.status | failure(~jnp.isfinite(_merit(value, value.scales)), Status.NONFINITE))
    valid = value.status == 0
    return State(vector, value, _integer(0), _integer(1), _integer(0), _integer(0), _integer(0),
                 jnp.asarray(jnp.inf, vector.dtype), jnp.zeros((len(Rejection),), jnp.int32),
                 _integer(Rejection.NONE), jnp.where(valid, _integer(Outcome.RUNNING), _integer(Outcome.DIVERGED)),
                 jnp.where(valid, _integer(Reason.OUTER_BUDGET), _integer(Reason.INVALID_INITIALIZATION)))


def iteration(coordinate, parameters, sites, state, budgets=Budgets()):
    """One inspectable accepted/trial iteration; no optimizer-history derivative.

    Call only for a RUNNING state with a proper evaluation. Both proposal
    strategies operate in coordinate space, while merit lives in residual space.
    """
    target = coordinate.pack(state.evaluation.local.targets, state.evaluation.auxiliary.target)
    direction = target - state.coordinate
    first = line_search(coordinate, parameters, sites, state.coordinate, state.evaluation, direction, budgets)

    def fallback(_):
        direction, linear = gauss_newton(coordinate, parameters, sites, state.coordinate, state.evaluation, budgets)
        trial = line_search(coordinate, parameters, sites, state.coordinate, state.evaluation, direction, budgets)
        return trial, linear.iteration, linear.finite, jnp.any(direction != 0)

    second, linear_count, linear_finite, moving = jax.lax.cond(
        first.accepted,
        lambda _: (first, _integer(0), jnp.asarray(True), jnp.asarray(True)), fallback, operand=None)
    fallback_used = ~first.accepted
    accepted = second.accepted
    added_evaluations = first.evaluations + jnp.where(fallback_used, second.evaluations, 0)
    added_counts = first.rejection_counts + jnp.where(fallback_used, second.rejection_counts, 0)
    difference = abs(second.evaluation.local.free_energy - state.evaluation.local.free_energy)
    difference /= 1 + abs(second.evaluation.local.free_energy)
    valid = ((second.evaluation.status == 0) & jnp.isfinite(difference)
             & jnp.isfinite(_merit(second.evaluation, second.evaluation.scales)))
    status = jnp.where(accepted, _integer(Outcome.RUNNING), _integer(Outcome.EXHAUSTED))
    status = jnp.where(~linear_finite | ~valid, _integer(Outcome.DIVERGED), status)
    reason = jnp.where(moving, _integer(Reason.TRIAL_BUDGET), _integer(Reason.STATIONARY_RESIDUAL))
    reason = jnp.where(~linear_finite, _integer(Reason.INVALID_DERIVATIVE), reason)
    reason = jnp.where(~valid, _integer(Reason.INVALID_ACCEPTED), reason)
    return State(second.coordinate, second.evaluation, state.iterations + accepted.astype(jnp.int32),
                 state.evaluations + added_evaluations, state.rejected_trials + jnp.sum(added_counts),
                 state.linear_iterations + linear_count, state.gauss_newton_steps + fallback_used.astype(jnp.int32),
                 jnp.where(accepted, difference, state.relative_energy_change), state.rejection_counts + added_counts,
                 second.last_rejection, status, reason)


def certify(coordinate, parameters, sites, state):
    """Recompute all outputs at the returned coordinate, with truthful status."""
    value = joint.evaluate(coordinate, parameters, sites, state.coordinate)
    numeric = value.status | state.evaluation.status
    numeric |= failure(~jnp.isfinite(_merit(value, value.scales)), Status.NONFINITE)
    energy_change = abs(value.local.free_energy - state.evaluation.local.free_energy) / (1 + abs(value.local.free_energy))
    valid = (numeric == 0) & jnp.isfinite(energy_change)
    passes = value.certificates.maximum() <= tolerance(state.coordinate.dtype)
    # Unchanged evaluation establishes energy stability only after all other
    # complete certificates pass. A stationary nonzero residual cannot use it.
    change = jnp.where(passes, energy_change, state.relative_energy_change)
    passes &= change <= tolerance(state.coordinate.dtype)
    status = jnp.where(passes & valid, _integer(Outcome.CONVERGED), _integer(Outcome.EXHAUSTED))
    status = jnp.where(~valid | (state.status == Outcome.DIVERGED), _integer(Outcome.DIVERGED), status)
    reason = jnp.where(status == Outcome.CONVERGED, _integer(Reason.CERTIFIED), state.reason)
    reason = jnp.where(~valid & (state.status != Outcome.DIVERGED), _integer(Reason.INVALID_ACCEPTED), reason)
    return state._replace(evaluation=value._replace(status=numeric), evaluations=state.evaluations + 1,
                          relative_energy_change=change, status=status, reason=reason)


def solve(coordinate, parameters, sites, initial_xi=0., *, perturb=False, budgets=Budgets()):
    """Joint bounded solve; host/JIT/vmap share this same numerical program.

    CONVERGED requires every dynamic maximum certificate, proper active beliefs,
    finite primal/dual objectives and stable free energy at one returned point.
    EXHAUSTED and DIVERGED carry distinct reasons and rejected-trial counts.
    """
    state = initial_state(coordinate, parameters, sites, initial_xi, perturb=perturb)
    tau = tolerance(state.coordinate.dtype)

    def condition(state):
        return ((state.status == Outcome.RUNNING) & (state.iterations < budgets.outer)
                & (state.evaluation.certificates.maximum() > tau))

    state = jax.lax.while_loop(condition, lambda s: iteration(coordinate, parameters, sites, s, budgets), state)
    state = state._replace(reason=jnp.where(state.status == Outcome.RUNNING, _integer(Reason.OUTER_BUDGET), state.reason))
    return certify(coordinate, parameters, sites, state)

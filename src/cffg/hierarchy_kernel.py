"""Exact log-domain region messages and Bethe terms for one categorical child tick."""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def _finite_logs(x):
    # Inaccessible atoms have zero expectation. Mask their log before multiplication.
    return jnp.where(jnp.isneginf(x), 0, x)


def initial_cost(log_conditional):
    zeros = jnp.zeros_like(log_conditional)
    return jnp.stack((-_finite_logs(log_conditional), zeros, zeros, zeros), axis=-1)


@jax.jit
def child_message(frontier, old_cavity, old_cost, transition, emission, observation):
    """Eliminate the old fast state for each regime, excluding *all* parent mass.

    Costs retain, conditional on the endpoint, E[-log initial], E[-log B],
    E[-log A], and the negative conditional entropy of the eliminated path.
    These expectations allow a smoothed path ledger without storing a growing
    trajectory. Missing emissions integrate over their unobserved outcome.
    """
    pair = transition + frontier[:, None, :]
    pred = logsumexp(pair, axis=2)
    log_reverse = pair - _finite_logs(pred)[:, :, None]
    reverse = jnp.exp(log_reverse)
    missing = observation < 0
    log_likelihood = jnp.where(missing, jnp.zeros_like(emission[0]), emission[observation])
    new_frontier = pred + log_likelihood
    costs = jnp.einsum("rji,rik->rjk", reverse, old_cost)
    transition_cost = -jnp.sum(reverse * _finite_logs(transition), axis=2)
    emission_entropy = -jnp.sum(jnp.exp(emission) * _finite_logs(emission), axis=0)
    emission_cost = jnp.where(missing, emission_entropy, -_finite_logs(log_likelihood))
    entropy = jnp.sum(reverse * _finite_logs(log_reverse), axis=2) - jnp.where(missing, emission_entropy, 0)
    costs = costs + jnp.stack((jnp.zeros_like(pred), transition_cost,
                               jnp.broadcast_to(emission_cost, pred.shape), entropy), axis=-1)
    scale = logsumexp(new_frontier, axis=1)
    # Per-regime scaling keeps a rare event history out of subsequent softmax
    # differences. Keep its true likelihood separately, including exact zeros.
    conditional = new_frontier - _finite_logs(scale)[:, None]
    finite = jnp.isfinite(costs).all() & jnp.isfinite(reverse).all()
    return conditional, costs, reverse, old_cavity + scale, finite


@jax.jit
def parent_message(prior, cavity, frontier, cost, reverse):
    shift = _finite_logs(jnp.max(jnp.where(jnp.isfinite(prior), cavity, -jnp.inf)))
    weights = prior + (cavity - shift)
    centered = logsumexp(weights)
    log_z = shift + centered
    log_joint = (weights - _finite_logs(centered))[:, None] + frontier
    joint = jnp.exp(log_joint)
    log_regime = logsumexp(log_joint, axis=1)
    regime = jnp.exp(log_regime)
    terms = jnp.einsum("rx,rxk->k", joint, cost)
    # Two regions (slow prior, fast trajectory) share the current regime.
    # The child path entropy includes the endpoint entropy; add H(R) once.
    parent = jnp.sum(regime * (_finite_logs(log_regime) - _finite_logs(prior)))
    child_endpoint = jnp.sum(joint * _finite_logs(log_joint))
    separator = -jnp.sum(regime * _finite_logs(log_regime))
    ledger = jnp.concatenate((parent[None], terms, child_endpoint[None], separator[None]))
    return log_z, log_joint, ledger, joint[:, :, None] * reverse


@jax.jit
def rollover(log_joint, transition):
    """Move the regime exactly once and preserve its correlation with the child."""
    pair = transition[:, :, None] + log_joint[None, :, :]
    next_joint = logsumexp(pair, axis=1)
    prior = logsumexp(next_joint, axis=1)
    new_child = next_joint - _finite_logs(prior)[:, None]
    # A conditional on an unreachable regime is not identified by the joint.
    # Use the vacuous child there; the exact-zero parent mass still excludes it.
    conditional = jnp.where(jnp.isneginf(prior)[:, None], -jnp.log(jnp.asarray(
        log_joint.shape[1], log_joint.dtype)), new_child)
    parent_stat = jnp.exp(logsumexp(pair, axis=2))
    return prior, conditional, initial_cost(conditional), parent_stat

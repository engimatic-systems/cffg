"""P-substituted hybrid Gaussian-mixture GFE rules.

The composite implements DERIVATIONS.md D5. A discrete selector carries the
epistemic coordinate, Gaussian state messages retain the D3 assumed-density
rules, and the substituted output is represented by its moment-matched
Gaussian. The only approximation is the collapsed-mixture entropy: it upper
bounds the true mixture entropy, so the reported node-local GFE lower-bounds
the exact substituted objective.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from cffg import gauss
from cffg.compiled import NEG
from cffg.rules import (
    GfeSolveResult,
    _gfe_guard_incoming,
    _simplex_solve_raw,
)


_LOG2PI = jnp.log(2.0 * jnp.pi)


class HybridGfeDiagnostics(NamedTuple):
    """Per-member D5 terms and local-solve diagnostics."""

    rho: jax.Array
    total: jax.Array
    mi_regime_surrogate: jax.Array
    mi_regime_lower: jax.Array
    mi_regime_upper: jax.Array
    mi_state: jax.Array
    h_regime: jax.Array
    h_state: jax.Array
    goal_risk: jax.Array
    residual: jax.Array
    status: jax.Array


def _entropy_covariance(covariance: jax.Array) -> jax.Array:
    dimension = covariance.shape[-1]
    return 0.5 * (
        gauss._pd_logdet(covariance)
        + dimension * (_LOG2PI + 1.0)
    )


def _entropy_precision(precision: jax.Array) -> jax.Array:
    dimension = precision.shape[-1]
    return 0.5 * (
        dimension * (_LOG2PI + 1.0)
        - gauss._pd_logdet(precision)
    )


def _discrete_entropy(probabilities: jax.Array) -> jax.Array:
    active = probabilities > 0
    # Keep reverse derivatives finite on the fixed support without adding mass.
    log_probability = jnp.log(jnp.where(active, probabilities, 1.0))
    return -jnp.sum(
        jnp.where(
            active,
            probabilities * log_probability,
            0.0,
        )
    )


def _collapse(
    rho: jax.Array, component_means: jax.Array, component_covariances: jax.Array
) -> tuple[jax.Array, jax.Array]:
    mean = jnp.einsum("k,ki->i", rho, component_means)
    second = jnp.einsum(
        "k,kij->ij",
        rho,
        component_covariances
        + jnp.einsum("ki,kj->kij", component_means, component_means),
    )
    covariance = second - jnp.outer(mean, mean)
    return mean, 0.5 * (covariance + covariance.T)


def _goal_costs(
    component_means: jax.Array,
    component_covariances: jax.Array,
    goal_mean: jax.Array,
    goal_precision: jax.Array,
) -> jax.Array:
    error = component_means - goal_mean
    return 0.5 * (
        jnp.einsum("ki,ij,kj->k", error, goal_precision, error)
        + jnp.einsum("ij,kji->k", goal_precision, component_covariances)
    )


def _log_potential(
    rho: jax.Array,
    component_means: jax.Array,
    component_covariances: jax.Array,
    emission_entropies: jax.Array,
    state_entropies: jax.Array,
    goal_costs: jax.Array,
) -> jax.Array:
    mean, covariance = _collapse(
        rho, component_means, component_covariances
    )
    inverse = gauss._pd_inverse(covariance)
    derivative = component_covariances + jnp.einsum(
        "ki,kj->kij", component_means, component_means
    )
    derivative = (
        derivative
        - jnp.einsum("ki,j->kij", component_means, mean)
        - jnp.einsum("i,kj->kij", mean, component_means)
    )
    entropy_gradient = 0.5 * jnp.einsum("ij,kji->k", inverse, derivative)
    return (
        entropy_gradient
        - emission_entropies
        + state_entropies
        - goal_costs
    )


@jax.custom_jvp
def _implicit_solution(
    component_means,
    component_covariances,
    emission_entropies,
    state_entropies,
    goal_costs,
    log_incoming,
    solution,
):
    return solution


@_implicit_solution.defjvp
def _implicit_solution_jvp(primals, tangents):
    (
        component_means,
        component_covariances,
        emission_entropies,
        state_entropies,
        goal_costs,
        log_incoming,
        solution,
    ) = primals
    (
        d_means,
        d_covariances,
        d_emission_entropies,
        d_state_entropies,
        d_goal_costs,
        d_log_incoming,
        _,
    ) = tangents
    dtype = jnp.result_type(
        component_means,
        component_covariances,
        emission_entropies,
        state_entropies,
        goal_costs,
        log_incoming,
    )
    if log_incoming.shape[0] == 1:
        return solution, jnp.zeros_like(solution)

    def center(value):
        return value - value.mean()

    def unpack(reduced):
        return center(
            jnp.concatenate([reduced, jnp.zeros((1,), dtype=dtype)])
        )

    def stationary(
        reduced,
        means,
        covariances,
        emission_h,
        state_h,
        costs,
        incoming,
    ):
        rho = jax.nn.softmax(unpack(reduced))
        potential = _log_potential(
            rho, means, covariances, emission_h, state_h, costs
        )
        return (
            center(unpack(reduced))
            - center(potential + _gfe_guard_incoming(incoming))
        )[:-1]

    stopped = jax.lax.stop_gradient(solution)
    logits = center(
        _log_potential(
            stopped,
            component_means,
            component_covariances,
            emission_entropies,
            state_entropies,
            goal_costs,
        )
        + _gfe_guard_incoming(log_incoming)
    )
    reduced = logits[:-1] - logits[-1]
    jacobian = jax.jacfwd(stationary, argnums=0)(
        reduced,
        component_means,
        component_covariances,
        emission_entropies,
        state_entropies,
        goal_costs,
        log_incoming,
    )
    _, parameter_tangent = jax.jvp(
        lambda means, covariances, emission_h, state_h, costs, incoming: stationary(
            reduced,
            means,
            covariances,
            emission_h,
            state_h,
            costs,
            incoming,
        ),
        (
            component_means,
            component_covariances,
            emission_entropies,
            state_entropies,
            goal_costs,
            log_incoming,
        ),
        (
            d_means,
            d_covariances,
            d_emission_entropies,
            d_state_entropies,
            d_goal_costs,
            d_log_incoming,
        ),
    )
    d_reduced = jnp.linalg.solve(jacobian, -parameter_tangent)
    _, d_solution = jax.jvp(
        lambda value: jax.nn.softmax(unpack(value)),
        (reduced,),
        (d_reduced,),
    )
    return solution, d_solution


def _solve(
    component_means,
    component_covariances,
    emission_entropies,
    state_entropies,
    goal_costs,
    log_incoming,
) -> GfeSolveResult:
    dtype = jnp.result_type(
        component_means,
        component_covariances,
        emission_entropies,
        state_entropies,
        goal_costs,
        log_incoming,
    )
    values = tuple(
        jnp.asarray(value, dtype=dtype)
        for value in (
            component_means,
            component_covariances,
            emission_entropies,
            state_entropies,
            goal_costs,
            log_incoming,
        )
    )
    means, covariances, emission_h, state_h, costs, incoming = values

    def potential(rho):
        return _log_potential(
            rho, means, covariances, emission_h, state_h, costs
        )

    solved = _simplex_solve_raw(potential, incoming)
    solution = _implicit_solution(
        means,
        covariances,
        emission_h,
        state_h,
        costs,
        incoming,
        jax.lax.stop_gradient(solved.solution),
    )
    support = incoming != jnp.asarray(NEG, dtype=incoming.dtype)
    supported = jnp.where(support, solution, 0.0)
    total = supported.sum()
    solution = supported / jnp.where(total > 0, total, 1.0)
    return GfeSolveResult(
        solution, solved.status, solved.iterations, solved.residual
    )


def _pairwise_entropy_lower(
    rho: jax.Array,
    component_means: jax.Array,
    component_covariances: jax.Array,
) -> jax.Array:
    dimension = component_means.shape[-1]
    pair_covariance = (
        component_covariances[:, None]
        + component_covariances[None, :]
    )
    difference = component_means[:, None] - component_means[None, :]
    inverse = gauss._pd_inverse(pair_covariance)
    log_density = -0.5 * (
        dimension * _LOG2PI
        + gauss._pd_logdet(pair_covariance)
        + jnp.einsum("kti,ktij,ktj->kt", difference, inverse, difference)
    )
    log_weights = jnp.where(rho > 0, jnp.log(rho), -jnp.inf)
    mixtures = logsumexp(log_weights[None, :] + log_density, axis=1)
    return -jnp.sum(jnp.where(rho > 0, rho * mixtures, 0.0))


def _terms(
    rho,
    component_means,
    component_covariances,
    emission_entropies,
    state_entropies,
    goal_mean,
    goal_precision,
    *,
    include_bounds,
):
    _, collapse_covariance = _collapse(
        rho, component_means, component_covariances
    )
    collapse_entropy = _entropy_covariance(collapse_covariance)
    component_entropies = jax.vmap(_entropy_covariance)(
        component_covariances
    )
    conditional_entropy = rho @ component_entropies
    regime_entropy = _discrete_entropy(rho)
    state_entropy = rho @ state_entropies
    emission_entropy = rho @ emission_entropies
    goal_costs = _goal_costs(
        component_means,
        component_covariances,
        goal_mean,
        goal_precision,
    )
    dimension = goal_mean.shape[-1]
    goal_constant = 0.5 * (
        dimension * _LOG2PI - gauss._pd_logdet(goal_precision)
    )
    goal_risk = goal_constant + rho @ goal_costs
    mi_regime = jnp.maximum(collapse_entropy - conditional_entropy, 0.0)
    mi_state = rho @ (component_entropies - emission_entropies)
    if include_bounds:
        pairwise = _pairwise_entropy_lower(
            rho, component_means, component_covariances
        )
        mi_lower = jnp.maximum(pairwise - conditional_entropy, 0.0)
    else:
        mi_lower = jnp.asarray(0.0, dtype=rho.dtype)
    mi_upper = jnp.minimum(mi_regime, regime_entropy)
    total = (
        -collapse_entropy
        + emission_entropy
        - regime_entropy
        - state_entropy
        + goal_risk
    )
    return (
        total,
        mi_regime,
        mi_lower,
        mi_upper,
        mi_state,
        regime_entropy,
        state_entropy,
        goal_risk,
    )


def _linear_components(eta, precision, matrix, offset, noise_precision):
    proper = gauss._pd_ok(precision) & jnp.all(gauss._pd_ok(noise_precision))
    dimension = precision.shape[-1]
    safe_precision = jnp.where(
        proper,
        precision,
        jnp.eye(dimension, dtype=precision.dtype),
    )
    covariance = gauss._pd_inverse(safe_precision)
    mean = covariance @ eta
    noise_covariance = gauss._pd_inverse(noise_precision)
    component_means = jnp.einsum("kij,j->ki", matrix, mean) + offset
    component_covariances = (
        jnp.einsum("kij,jl,kml->kim", matrix, covariance, matrix)
        + noise_covariance
    )
    emission_entropies = jax.vmap(_entropy_precision)(noise_precision)
    state_entropy = _entropy_precision(safe_precision)
    state_entropies = jnp.full(
        (matrix.shape[0],), state_entropy, dtype=eta.dtype
    )
    return (
        component_means,
        component_covariances,
        emission_entropies,
        state_entropies,
        proper,
    )


def _prior_components(mean, noise_precision):
    covariance = gauss._pd_inverse(noise_precision)
    emission_entropies = jax.vmap(_entropy_precision)(noise_precision)
    state_entropies = jnp.zeros_like(emission_entropies)
    proper = jnp.all(gauss._pd_ok(noise_precision))
    return mean, covariance, emission_entropies, state_entropies, proper


def _evaluate(
    log_incoming,
    component_means,
    component_covariances,
    emission_entropies,
    state_entropies,
    goal_mean,
    goal_precision,
    proper,
    *,
    include_bounds=False,
):
    costs = _goal_costs(
        component_means,
        component_covariances,
        goal_mean,
        goal_precision,
    )
    solved = _solve(
        component_means,
        component_covariances,
        emission_entropies,
        state_entropies,
        costs,
        log_incoming,
    )
    rho = solved.solution
    potential = _log_potential(
        rho,
        component_means,
        component_covariances,
        emission_entropies,
        state_entropies,
        costs,
    )
    selector = jax.nn.log_softmax(potential)
    out_mean, out_covariance = _collapse(
        rho, component_means, component_covariances
    )
    out_precision = gauss._pd_inverse(out_covariance)
    out_eta = out_precision @ out_mean
    terms = _terms(
        rho,
        component_means,
        component_covariances,
        emission_entropies,
        state_entropies,
        goal_mean,
        goal_precision,
        include_bounds=include_bounds,
    )
    finite = (
        proper
        & gauss._pd_ok(goal_precision)
        & jnp.all(gauss._pd_ok(component_covariances))
        & gauss._pd_ok(out_covariance)
        & jnp.all(jnp.isfinite(out_eta))
        & jnp.all(jnp.isfinite(out_precision))
        & jnp.all(jnp.isfinite(jnp.asarray(terms)))
    )
    status = jnp.where(finite, solved.status, 2)
    residual = jnp.where(finite, solved.residual, jnp.inf)
    terms = (
        jnp.where(finite, terms[0], jnp.inf),
        *terms[1:],
    )
    return selector, out_eta, out_precision, terms, rho, residual, status


def linear_update(
    log_incoming,
    eta,
    precision,
    matrix,
    offset,
    noise_precision,
    goal_mean,
    goal_precision,
):
    """Batched D5 update for ``gaussian_mixture_linear`` composites."""

    def one(le, e, lam, a, b, w, g, gw):
        vacuous = jnp.all(e == 0) & jnp.all(lam == 0)

        def transient(_):
            out_dim = b.shape[-1]
            return (
                jax.nn.log_softmax(jnp.zeros_like(le)),
                jnp.zeros((out_dim,), dtype=e.dtype),
                jnp.zeros((out_dim, out_dim), dtype=lam.dtype),
                jnp.zeros_like(e),
                jnp.zeros_like(lam),
                jnp.asarray(0.0, dtype=e.dtype),
                jnp.asarray(0),
            )

        def evaluate(_):
            components = _linear_components(e, lam, a, b, w)
            selector, out_e, out_l, _, _, residual, status = _evaluate(
                le,
                *components[:-1],
                g,
                gw,
                components[-1],
                include_bounds=False,
            )
            return (
                selector,
                out_e,
                out_l,
                # The substituted output cavity is exterior and therefore
                # vacuous; D3's backward collapse is exactly
                # (eta, Lambda) = 0.
                jnp.zeros_like(e),
                jnp.zeros_like(lam),
                residual,
                status,
            )

        return jax.lax.cond(vacuous, transient, evaluate, operand=None)

    return jax.vmap(one)(
        log_incoming,
        eta,
        precision,
        matrix,
        offset,
        noise_precision,
        goal_mean,
        goal_precision,
    )


def prior_update(
    log_incoming,
    mean,
    noise_precision,
    goal_mean,
    goal_precision,
):
    """Batched D5 update for ``gaussian_mixture`` composites."""

    def one(le, m, w, g, gw):
        components = _prior_components(m, w)
        selector, out_e, out_l, _, _, residual, status = _evaluate(
            le, *components[:-1], g, gw, components[-1], include_bounds=False
        )
        return selector, out_e, out_l, residual, status

    return jax.vmap(one)(
        log_incoming, mean, noise_precision, goal_mean, goal_precision
    )


def _linear_diagnostics(
    log_incoming,
    eta,
    precision,
    matrix,
    offset,
    noise_precision,
    goal_mean,
    goal_precision,
    *,
    include_bounds,
) -> HybridGfeDiagnostics:
    """Batched direct ledger terms and entropy bounds for linear composites."""

    def one(le, e, lam, a, b, w, g, gw):
        components = _linear_components(e, lam, a, b, w)
        _, _, _, terms, rho, residual, status = _evaluate(
            le,
            *components[:-1],
            g,
            gw,
            components[-1],
            include_bounds=include_bounds,
        )
        return HybridGfeDiagnostics(rho, *terms, residual, status)

    return jax.vmap(one)(
        log_incoming,
        eta,
        precision,
        matrix,
        offset,
        noise_precision,
        goal_mean,
        goal_precision,
    )


def linear_diagnostics(*args) -> HybridGfeDiagnostics:
    """D5 terms plus the O(K²) exact-mixture entropy sandwich."""
    return _linear_diagnostics(*args, include_bounds=True)


def _prior_diagnostics(
    log_incoming,
    mean,
    noise_precision,
    goal_mean,
    goal_precision,
    *,
    include_bounds,
) -> HybridGfeDiagnostics:
    """Batched direct ledger terms and entropy bounds for mixture priors."""

    def one(le, m, w, g, gw):
        components = _prior_components(m, w)
        _, _, _, terms, rho, residual, status = _evaluate(
            le,
            *components[:-1],
            g,
            gw,
            components[-1],
            include_bounds=include_bounds,
        )
        return HybridGfeDiagnostics(rho, *terms, residual, status)

    return jax.vmap(one)(
        log_incoming, mean, noise_precision, goal_mean, goal_precision
    )


def prior_diagnostics(*args) -> HybridGfeDiagnostics:
    """D5 terms plus the O(K²) exact-mixture entropy sandwich."""
    return _prior_diagnostics(*args, include_bounds=True)


def linear_energy(*args) -> jax.Array:
    return _linear_diagnostics(*args, include_bounds=False).total


def prior_energy(*args) -> jax.Array:
    return _prior_diagnostics(*args, include_bounds=False).total

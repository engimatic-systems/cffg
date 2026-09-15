"""Total energy derivatives of certified adaptive inference fixed points.

For R(u,theta)=0 and the fully recomputed reported energy E(u,theta), solve
R_u.T lambda=E_u and return E_theta-R_theta.T lambda. Gaussian chart metrics
are held fixed as a coordinate choice; approximation coefficients are never
frozen. The implicit-function interpretation requires a locally isolated,
differentiable root. Numerical certificates check the current root, original
adjoint residual and excessive adjoint amplification; they are not a theorem
of global uniqueness for a nonlinear model.
"""

from dataclasses import dataclass
from enum import StrEnum

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres
import numpy as np

from cffg.fixed_point import RootProgram, root_context
from cffg.outer import OuterArrays, build_outer_chart, chart_coordinates
from cffg.runtime_types import RunStatus
from cffg.trainable import NeuralParameters


class GradientStatus(StrEnum):
    CONVERGED = "converged"
    UNCERTIFIED = "uncertified_inference"
    UNSUPPORTED = "unsupported_domain"
    NONFINITE = "nonfinite"
    ADJOINT_BUDGET = "adjoint_budget_exhausted"
    ILL_CONDITIONED = "excessive_adjoint_amplification"


@dataclass(frozen=True)
class FixedPointGradient:
    status: GradientStatus
    message: str
    value: float = float("nan")
    gradient: jax.Array | None = None
    weights: jax.Array | None = None
    parameters: NeuralParameters | None = None
    root_residual: float = float("inf")
    adjoint_residual: float = float("inf")
    adjoint_gain: float = float("inf")

    def require_valid(self):
        if self.status is not GradientStatus.CONVERGED:
            raise ValueError(f"fixed-point gradient {self.status.value}: {self.message}")
        return self.gradient


def _admit(engine, result):
    if result.status is not RunStatus.CONVERGED:
        return FixedPointGradient(GradientStatus.UNCERTIFIED, "inference must report converged")
    if engine.arena.semiring != "sum" or engine.arena.mu3_groups or len(engine.arena.delta_pairs):
        return FixedPointGradient(GradientStatus.UNSUPPORTED,
                                  "requires a deterministic smooth sum-product graph")
    if not engine._state_is_finite(result.state) or any(
        not np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(result.state.params)
    ):
        return FixedPointGradient(GradientStatus.NONFINITE, "inference state or parameters are non-finite")
    return None


class FixedPointDifferentiator:
    """Reusable matrix-free derivative in a fixed reference coordinate chart.

    Only the coordinate metric is reused. Observations, parameters, messages,
    local approximation coefficients and both derivative operators stay dynamic.
    Each call independently certifies the supplied run before exposing a gradient.
    """

    def __init__(self, engine, reference, *, factors=None, tolerance=None,
                 max_iterations=20, krylov_dimension=32):
        refused = _admit(engine, reference)
        if refused is not None:
            raise ValueError(refused.message)
        for name, value in (("adjoint iterations", max_iterations), ("Krylov dimension", krylov_dimension)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.engine = engine
        self.factors = None if factors is None else tuple(factors)
        self.parameters = NeuralParameters.from_engine(engine, reference.state.params, self.factors)
        self.dtype = np.asarray(self.parameters.extract(reference.state.params)).dtype
        eps = np.finfo(self.dtype).eps
        if tolerance is not None and (not np.isfinite(tolerance) or tolerance <= 0):
            raise ValueError("gradient tolerance must be finite and positive")
        self.tolerance = max(64 * eps, np.sqrt(eps) if tolerance is None else float(tolerance))
        state = reference.state
        arrays = OuterArrays(state.messages, state.g_eta, state.g_lam, state.ef_eta)
        chart = build_outer_chart(engine.arena, arrays, engine._outer_eliminated)
        if not chart.proper_reference:
            raise ValueError("no proper fixed-point reference")
        self.chart = chart
        program = RootProgram(chart, engine._iter_step, engine._reconstruct)
        parameters, tau = self.parameters, self.tolerance

        def residual(value, theta, context):
            context = context._replace(params=parameters.inject(theta, context.params))
            return program.residual(value, context)

        def energy(value, theta, context):
            context = context._replace(params=parameters.inject(theta, context.params))
            return program.energy(value, context, engine._ledger.belief_kernel)

        @jax.jit
        def certificate(coordinates, weights, context):
            mapped, local_residual, local_status = program.update(coordinates, context)
            mapped_coordinates = chart_coordinates(chart, mapped)
            root = mapped_coordinates - coordinates
            root_size = jnp.max(jnp.abs(root) / (1 + jnp.abs(coordinates)), initial=0.)
            return (energy(coordinates, weights, context), energy(mapped_coordinates, weights, context),
                    root_size, local_residual, local_status, mapped.messages)

        @jax.jit
        def differentiate(coordinates, weights, context):
            energy_u, energy_theta = jax.grad(energy, argnums=(0, 1))(coordinates, weights, context)
            _, transpose = jax.vjp(lambda u: residual(u, weights, context), coordinates)
            operator = lambda vector: transpose(vector)[0]
            if chart.size:
                adjoint, _ = gmres(operator, energy_u, tol=tau, atol=0.,
                                   restart=min(krylov_dimension, chart.size), maxiter=max_iterations,
                                   solve_method="incremental")
                # Certify the original system: the solver's info flag is insufficient.
                error = operator(adjoint) - energy_u
                scale = jnp.maximum(jnp.linalg.norm(energy_u), jnp.finfo(weights.dtype).tiny)
                linear_residual = jnp.linalg.norm(error) / scale
                gain = jnp.linalg.norm(adjoint) / scale
            else:
                adjoint = jnp.zeros_like(coordinates)
                linear_residual = gain = jnp.asarray(0., weights.dtype)
            _, parameter_transpose = jax.vjp(lambda theta: residual(coordinates, theta, context), weights)
            return energy_theta - parameter_transpose(adjoint)[0], linear_residual, gain

        self._certificate = certificate
        self._differentiate = differentiate

    def __call__(self, result):
        refused = _admit(self.engine, result)
        if refused is not None:
            return refused
        state, chart, parameters = result.state, self.chart, self.parameters
        try:
            current = NeuralParameters.from_engine(self.engine, state.params, self.factors)
        except (ValueError, KeyError) as error:
            return FixedPointGradient(GradientStatus.UNSUPPORTED, str(error))
        weights = parameters.extract(state.params)
        if current != parameters or np.asarray(weights).dtype != self.dtype or chart.support_changed(state.messages):
            return FixedPointGradient(GradientStatus.UNCERTIFIED, "parameter layout, dtype or exact support changed")
        arrays = OuterArrays(state.messages, state.g_eta, state.g_lam, state.ef_eta)
        context = root_context(chart, state)._replace(base=arrays._replace(ef_eta=state.ef_eta.reshape(-1)))
        coordinates = chart_coordinates(chart, arrays)
        value, next_value, root, local, status, messages = self._certificate(coordinates, weights, context)
        value, next_value, root, local = map(float, (value, next_value, root, local))
        facts = dict(value=value, weights=weights, parameters=parameters, root_residual=root)
        if not all(np.isfinite(v) for v in (value, next_value, root, local)) or int(status):
            return FixedPointGradient(GradientStatus.NONFINITE, "current residual, local rule or energy is invalid", **facts)
        tau = self.tolerance
        if chart.support_changed(messages) or root > tau or abs(next_value - value) > tau * (1 + abs(value)):
            return FixedPointGradient(GradientStatus.UNCERTIFIED,
                                      "current weights and observations do not satisfy the root certificate", **facts)
        gradient, residual, gain = self._differentiate(coordinates, weights, context)
        residual, gain = float(residual), float(gain)
        facts.update(adjoint_residual=residual, adjoint_gain=gain)
        if not np.isfinite(residual) or not np.isfinite(gain) or not bool(jnp.all(jnp.isfinite(gradient))):
            return FixedPointGradient(GradientStatus.NONFINITE, "adjoint or total derivative is non-finite", **facts)
        if residual > tau:
            return FixedPointGradient(GradientStatus.ADJOINT_BUDGET, "original adjoint residual exceeds tolerance", **facts)
        if gain > 1 / np.sqrt(np.finfo(self.dtype).eps):
            return FixedPointGradient(GradientStatus.ILL_CONDITIONED, "adjoint amplification exceeds dtype resolution", **facts)
        return FixedPointGradient(GradientStatus.CONVERGED, "root and adjoint residuals certified",
                                  gradient=gradient, **facts)


def fixed_point_gradient(engine, result, *, factors=None, tolerance=None, max_iterations=20,
                         krylov_dimension=32):
    """Differentiate one certified run without changing its parameters.

    Use FixedPointDifferentiator to reuse compilation across training steps.
    Stochastic mu3 estimates and discrete delta decisions are outside this API;
    admitted smooth GFE factors use their complete graph residual and ledger.
    """
    refused = _admit(engine, result)
    if refused is not None:
        return refused
    if max_iterations <= 0:
        return FixedPointGradient(GradientStatus.ADJOINT_BUDGET, "positive adjoint iteration budget required")
    try:
        derivative = FixedPointDifferentiator(engine, result, factors=factors, tolerance=tolerance,
                                             max_iterations=max_iterations, krylov_dimension=krylov_dimension)
    except np.linalg.LinAlgError:
        return FixedPointGradient(GradientStatus.UNCERTIFIED, "invalid reference covariance")
    return derivative(result)

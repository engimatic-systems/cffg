"""Pure D10 structured coordinates for recurrent switching chains.

The authority keeps one categorical path belief and one Gaussian Markov
trajectory belief.  It is deliberately independent of compiler and engine
dispatch.  Categorical pair rows are source regimes and columns are
destination regimes.  Gaussian lag moments are oriented as
``E[x_t x_(t-1).T]``.

The recurrence auxiliary is explicit.  A Gaussian coordinate minimizes the
expanded Jaakkola--Jordan bound while ``xi`` and its Pólya--Gamma mean are
fixed.  The collapsed bound is evaluated only after that auxiliary is
reoptimized at a complete sweep boundary.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from cffg import gauss
from cffg._structured_hybrid_chain import (
    categorical_coordinate,
    conditioned_categorical_coordinate,
    recurrence_auxiliary,
)
from cffg._structured_hybrid_gaussian import (
    expected_dynamics_log_potential,
    gaussian_coordinate,
)
from cffg._structured_hybrid_objective import (
    dynamics_statistics,
    recurrence_statistics,
    structured_hybrid_objective,
)
from cffg._structured_hybrid_types import (
    CategoricalTrajectory,
    DynamicsStatistics,
    GaussianTrajectory,
    RecurrenceAuxiliary,
    RecurrenceStatistics,
    StructuredHybridBelief,
    StructuredHybridModel,
    StructuredHybridSweep,
    SWEEP_DIVERGED,
    SWEEP_FINITE,
    model_arrays,
    model_shape,
)


def _recurrence_gaussian_terms(
    pair,
    auxiliary: RecurrenceAuxiliary,
    weights,
    offsets,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    card = pair.shape[1]
    dimension = weights.shape[-1]
    if card == 1:
        zeros_eta = jnp.zeros((pair.shape[0], dimension), dtype=pair.dtype)
        zeros_precision = jnp.zeros(
            (pair.shape[0], dimension, dimension), dtype=pair.dtype
        )
        return zeros_eta, zeros_precision, jnp.asarray(True)
    exposure = jnp.cumsum(pair[:, :, ::-1], axis=2)[:, :, ::-1][
        :, :, :-1
    ]
    signed = pair[:, :, :-1] - 0.5 * exposure
    precision_weight = exposure * auxiliary.omega
    linear_weight = signed - precision_weight * offsets[None, :, :]
    eta = jnp.einsum("til,ild->td", linear_weight, weights)
    precision = jnp.einsum(
        "til,ild,ile->tde", precision_weight, weights, weights
    )
    precision = 0.5 * (precision + precision.swapaxes(-1, -2))
    proper = (
        auxiliary.proper.all()
        & jnp.all(jnp.isfinite(eta))
        & jnp.all(jnp.isfinite(precision))
        & jnp.all(jax.vmap(gauss._psd_ok)(precision))
    )
    return eta, precision, proper


def structured_hybrid_initial(
    model: StructuredHybridModel,
) -> StructuredHybridBelief:
    """Construct a truth-free proper starting belief for D10 sweeps."""
    model = model_arrays(model)
    time, card, dimension = model_shape(model)
    categorical = categorical_coordinate(
        model.log_initial,
        model.log_unary,
        jnp.zeros((time - 1, card), dtype=model.log_unary.dtype),
        jnp.zeros((time - 1, card, card), dtype=model.log_unary.dtype),
    )
    gaussian = gaussian_coordinate(
        model.gaussian_eta,
        model.gaussian_precision,
        categorical.singleton,
        model.dynamics_matrix,
        model.dynamics_offset,
        model.dynamics_covariance,
        jnp.zeros((time - 1, dimension), dtype=model.gaussian_eta.dtype),
        jnp.zeros(
            (time - 1, dimension, dimension), dtype=model.gaussian_eta.dtype
        ),
    )
    return StructuredHybridBelief(categorical, gaussian)


def _sequence_auxiliary(
    gaussian: GaussianTrajectory,
    weights,
    offsets,
    fixed_xi=None,
) -> RecurrenceAuxiliary:
    if fixed_xi is None:
        return jax.vmap(
            lambda mean, covariance: recurrence_auxiliary(
                mean, covariance, weights, offsets
            )
        )(gaussian.mean[:-1], gaussian.covariance[:-1])
    return jax.vmap(
        lambda mean, covariance, xi: recurrence_auxiliary(
            mean, covariance, weights, offsets, fixed_xi=xi
        )
    )(gaussian.mean[:-1], gaussian.covariance[:-1], fixed_xi)


def _belief_residual(
    previous: StructuredHybridBelief,
    current: StructuredHybridBelief,
) -> jax.Array:
    pair_delta = jnp.abs(
        previous.categorical.pair - current.categorical.pair
    )
    lag_delta = jnp.abs(
        previous.gaussian.lag_covariance - current.gaussian.lag_covariance
    )
    values = (
        jnp.max(
            jnp.abs(
                previous.categorical.singleton
                - current.categorical.singleton
            )
        ),
        jnp.max(pair_delta, initial=0.0),
        jnp.max(jnp.abs(previous.gaussian.mean - current.gaussian.mean)),
        jnp.max(
            jnp.abs(
                previous.gaussian.covariance - current.gaussian.covariance
            )
        ),
        jnp.max(lag_delta, initial=0.0),
    )
    return jnp.max(jnp.stack(values))


def structured_hybrid_evaluate(
    belief: StructuredHybridBelief,
    model: StructuredHybridModel,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate the collapsed D10 objective at an installed belief."""
    model = model_arrays(model)
    model_shape(model)
    dynamics, dynamics_ok = expected_dynamics_log_potential(
        belief.gaussian.mean,
        belief.gaussian.covariance,
        belief.gaussian.lag_covariance,
        model.dynamics_matrix,
        model.dynamics_offset,
        model.dynamics_covariance,
    )
    auxiliary = _sequence_auxiliary(
        belief.gaussian, model.recurrence_weights, model.recurrence_offsets
    )
    objective = structured_hybrid_objective(
        belief.categorical,
        belief.gaussian,
        model.log_initial,
        model.log_unary,
        model.gaussian_eta,
        model.gaussian_precision,
        model.gaussian_log_constant,
        dynamics,
        auxiliary.log_potential,
    )
    proper = (
        belief.categorical.proper
        & belief.gaussian.proper
        & dynamics_ok
        & jnp.all(auxiliary.proper)
        & jnp.isfinite(objective)
    )
    return jnp.where(proper, objective, jnp.inf), proper


def structured_hybrid_sweep(
    belief: StructuredHybridBelief,
    model: StructuredHybridModel,
) -> StructuredHybridSweep:
    """Apply one expanded-bound coordinate sweep and reoptimize its auxiliary."""
    model = model_arrays(model)
    model_shape(model)
    old_dynamics, old_dynamics_ok = expected_dynamics_log_potential(
        belief.gaussian.mean,
        belief.gaussian.covariance,
        belief.gaussian.lag_covariance,
        model.dynamics_matrix,
        model.dynamics_offset,
        model.dynamics_covariance,
    )
    old_auxiliary = _sequence_auxiliary(
        belief.gaussian, model.recurrence_weights, model.recurrence_offsets
    )
    previous_objective = structured_hybrid_objective(
        belief.categorical,
        belief.gaussian,
        model.log_initial,
        model.log_unary,
        model.gaussian_eta,
        model.gaussian_precision,
        model.gaussian_log_constant,
        old_dynamics,
        old_auxiliary.log_potential,
    )
    categorical = categorical_coordinate(
        model.log_initial,
        model.log_unary,
        old_dynamics,
        old_auxiliary.log_potential,
    )
    categorical_objective = structured_hybrid_objective(
        categorical,
        belief.gaussian,
        model.log_initial,
        model.log_unary,
        model.gaussian_eta,
        model.gaussian_precision,
        model.gaussian_log_constant,
        old_dynamics,
        old_auxiliary.log_potential,
    )
    recurrence_eta, recurrence_precision, recurrence_message_ok = (
        _recurrence_gaussian_terms(
            categorical.pair,
            old_auxiliary,
            model.recurrence_weights,
            model.recurrence_offsets,
        )
    )
    gaussian = gaussian_coordinate(
        model.gaussian_eta,
        model.gaussian_precision,
        categorical.singleton,
        model.dynamics_matrix,
        model.dynamics_offset,
        model.dynamics_covariance,
        recurrence_eta,
        recurrence_precision,
    )
    dynamics, dynamics_ok = expected_dynamics_log_potential(
        gaussian.mean,
        gaussian.covariance,
        gaussian.lag_covariance,
        model.dynamics_matrix,
        model.dynamics_offset,
        model.dynamics_covariance,
    )
    fixed_auxiliary = _sequence_auxiliary(
        gaussian,
        model.recurrence_weights,
        model.recurrence_offsets,
        old_auxiliary.xi,
    )
    fixed_objective = structured_hybrid_objective(
        categorical,
        gaussian,
        model.log_initial,
        model.log_unary,
        model.gaussian_eta,
        model.gaussian_precision,
        model.gaussian_log_constant,
        dynamics,
        fixed_auxiliary.log_potential,
    )
    auxiliary = _sequence_auxiliary(
        gaussian, model.recurrence_weights, model.recurrence_offsets
    )
    objective = structured_hybrid_objective(
        categorical,
        gaussian,
        model.log_initial,
        model.log_unary,
        model.gaussian_eta,
        model.gaussian_precision,
        model.gaussian_log_constant,
        dynamics,
        auxiliary.log_potential,
    )
    current = StructuredHybridBelief(categorical, gaussian)
    residual = _belief_residual(belief, current)
    dynamics_stats = dynamics_statistics(categorical.singleton, gaussian)
    recurrence_stats = recurrence_statistics(
        categorical.pair,
        gaussian,
        model.recurrence_weights,
        model.recurrence_offsets,
    )
    dtype = gaussian.mean.dtype
    objectives = jnp.stack(
        [
            previous_objective,
            categorical_objective,
            fixed_objective,
            objective,
        ]
    )
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(objectives)))
    time, card = model.log_unary.shape
    dimension = model.gaussian_eta.shape[1]
    work = time * (card**2 + dimension**3)
    # Protect the monotonicity invariant against accumulated roundoff in the
    # two polynomial-time chain solves without masking a finite objective rise.
    tolerance = 256.0 * jnp.sqrt(work) * jnp.finfo(dtype).eps * scale
    monotone = (
        categorical_objective <= previous_objective + tolerance
    ) & (fixed_objective <= categorical_objective + tolerance) & (
        objective <= fixed_objective + tolerance
    )
    proper = (
        belief.categorical.proper
        & belief.gaussian.proper
        & old_dynamics_ok
        & old_auxiliary.proper.all()
        & categorical.proper
        & recurrence_message_ok
        & gaussian.proper
        & dynamics_ok
        & fixed_auxiliary.proper.all()
        & auxiliary.proper.all()
        & dynamics_stats.proper
        & recurrence_stats.proper
        & jnp.isfinite(residual)
        & monotone
    )
    status = jnp.where(proper, SWEEP_FINITE, SWEEP_DIVERGED)
    return StructuredHybridSweep(
        current,
        auxiliary,
        dynamics,
        dynamics_stats,
        recurrence_stats,
        previous_objective,
        categorical_objective,
        fixed_objective,
        objective,
        jnp.where(proper, residual, jnp.inf),
        status,
    )


__all__ = [
    "CategoricalTrajectory",
    "DynamicsStatistics",
    "GaussianTrajectory",
    "RecurrenceAuxiliary",
    "RecurrenceStatistics",
    "StructuredHybridBelief",
    "StructuredHybridModel",
    "StructuredHybridSweep",
    "SWEEP_DIVERGED",
    "SWEEP_FINITE",
    "categorical_coordinate",
    "conditioned_categorical_coordinate",
    "dynamics_statistics",
    "expected_dynamics_log_potential",
    "gaussian_coordinate",
    "recurrence_auxiliary",
    "recurrence_statistics",
    "structured_hybrid_initial",
    "structured_hybrid_evaluate",
    "structured_hybrid_objective",
    "structured_hybrid_sweep",
]

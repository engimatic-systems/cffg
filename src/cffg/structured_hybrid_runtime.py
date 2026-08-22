"""Derived runtime state and unfused dispatch for compiled D10 components."""

from __future__ import annotations

from functools import reduce

import jax
import jax.numpy as jnp
import numpy as np

from cffg import gauss
from cffg._structured_hybrid_objective import recurrence_statistics
from cffg._structured_hybrid_types import (
    CategoricalTrajectory,
    GaussianTrajectory,
    StructuredHybridBelief,
    StructuredHybridModel,
    SWEEP_DIVERGED,
    SWEEP_FINITE,
)
from cffg.compiler import Arena, NEG, StructuredHybridPlan
from cffg.structured_hybrid import (
    structured_hybrid_evaluate,
    structured_hybrid_initial,
    structured_hybrid_sweep,
)


def runtime_key(plan: StructuredHybridPlan, field: str) -> str:
    return f"d10:{plan.name}:{field}"


_key = runtime_key


def _parameter_rows(arena: Arena, params, members, role):
    addresses = tuple(arena.member_of[name].parameter(role) for name in members)
    rows = jnp.stack(
        [params[address.key][address.index] for address in addresses]
    )
    same = jnp.all(rows == rows[0])
    finite = jnp.all(jnp.isfinite(rows))
    return rows[0], same & finite


def _normal_constant(mean, precision):
    dimension = mean.shape[-1]
    return (
        0.5 * gauss._pd_logdet(precision)
        - 0.5 * dimension * jnp.log(jnp.asarray(2.0 * jnp.pi, mean.dtype))
        - 0.5 * jnp.einsum("i,ij,j->", mean, precision, mean)
    )


def component_model(
    arena: Arena,
    plan: StructuredHybridPlan,
    g_obs,
    g_mask,
    params,
) -> tuple[StructuredHybridModel, jax.Array]:
    """Materialize the pure D10 model from compiled parameter addresses."""
    topology = plan.topology
    time = len(topology.state_edges)
    card = topology.cardinality
    dimension = topology.dimension
    dtype = jnp.result_type(g_obs, *params.values(), jnp.float32)

    initial_log, initial_tied = _parameter_rows(
        arena, params, (topology.initial_regime_member,), "statistic"
    )
    initial_log = initial_log.astype(dtype)
    initial_log = jnp.where(
        initial_log == jnp.asarray(NEG, dtype=dtype), -jnp.inf, initial_log
    )
    initial_mean, initial_mean_ok = _parameter_rows(
        arena, params, (topology.initial_state_member,), "mean"
    )
    initial_precision, initial_precision_ok = _parameter_rows(
        arena, params, (topology.initial_state_member,), "cov"
    )
    dynamics_matrix, dynamics_matrix_ok = _parameter_rows(
        arena, params, topology.dynamics_members, "matrix"
    )
    dynamics_offset, dynamics_offset_ok = _parameter_rows(
        arena, params, topology.dynamics_members, "offset"
    )
    dynamics_precision, dynamics_precision_ok = _parameter_rows(
        arena, params, topology.dynamics_members, "cov"
    )
    recurrence_weights, recurrence_weights_ok = _parameter_rows(
        arena, params, topology.recurrence_members, "weights"
    )
    recurrence_offsets, recurrence_offsets_ok = _parameter_rows(
        arena, params, topology.recurrence_members, "offsets"
    )
    emission_matrix, emission_matrix_ok = _parameter_rows(
        arena, params, topology.emission_members, "matrix"
    )
    emission_offset, emission_offset_ok = _parameter_rows(
        arena, params, topology.emission_members, "offset"
    )
    emission_precision, emission_precision_ok = _parameter_rows(
        arena, params, topology.emission_members, "cov"
    )
    values = (
        initial_mean,
        initial_precision,
        dynamics_matrix,
        dynamics_offset,
        dynamics_precision,
        recurrence_weights,
        recurrence_offsets,
        emission_matrix,
        emission_offset,
        emission_precision,
    )
    values = tuple(value.astype(dtype) for value in values)
    (
        initial_mean,
        initial_precision,
        dynamics_matrix,
        dynamics_offset,
        dynamics_precision,
        recurrence_weights,
        recurrence_offsets,
        emission_matrix,
        emission_offset,
        emission_precision,
    ) = values

    observations = g_obs[
        arena.g_obs_indices(plan.observation_rows, emission_matrix.shape[0])
    ].astype(dtype)
    masks = g_mask[plan.observation_rows].astype(dtype)
    centered = observations - emission_offset[None, :]
    emission_eta = jnp.einsum(
        "ij,tj->ti", emission_matrix.T @ emission_precision, centered
    )
    emission_natural_precision = (
        emission_matrix.T @ emission_precision @ emission_matrix
    )
    emission_constant = (
        0.5 * gauss._pd_logdet(emission_precision)
        - 0.5
        * emission_matrix.shape[0]
        * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype))
        - 0.5 * jnp.einsum(
            "ti,ij,tj->t", centered, emission_precision, centered
        )
    )
    gaussian_eta = masks[:, None] * emission_eta
    gaussian_precision = masks[:, None, None] * emission_natural_precision
    gaussian_constant = masks * emission_constant
    gaussian_eta = gaussian_eta.at[0].add(initial_precision @ initial_mean)
    gaussian_precision = gaussian_precision.at[0].add(initial_precision)
    gaussian_constant = gaussian_constant.at[0].add(
        _normal_constant(initial_mean, initial_precision)
    )

    dynamics_ok = jax.vmap(gauss._pd_ok)(dynamics_precision)
    safe_dynamics = jnp.where(
        dynamics_ok[:, None, None],
        dynamics_precision,
        jnp.broadcast_to(
            jnp.eye(dimension, dtype=dtype), dynamics_precision.shape
        ),
    )
    dynamics_covariance = jax.vmap(gauss._pd_inverse)(safe_dynamics)
    parameter_flags = (
        initial_tied,
        initial_mean_ok,
        initial_precision_ok,
        dynamics_matrix_ok,
        dynamics_offset_ok,
        dynamics_precision_ok,
        recurrence_weights_ok,
        recurrence_offsets_ok,
        emission_matrix_ok,
        emission_offset_ok,
        emission_precision_ok,
    )
    input_ok = reduce(jnp.logical_and, parameter_flags, jnp.asarray(True))
    input_ok &= gauss._pd_ok(initial_precision)
    input_ok &= jnp.all(dynamics_ok) & gauss._pd_ok(emission_precision)
    input_ok &= jnp.all(jnp.isfinite(observations))
    input_ok &= jnp.all(jnp.isfinite(masks))
    input_ok &= jnp.all((masks == 0) | (masks == 1))
    input_ok &= jnp.any(jnp.isfinite(initial_log))

    model = StructuredHybridModel(
        initial_log,
        jnp.zeros((time, card), dtype=dtype),
        gaussian_eta,
        gaussian_precision,
        gaussian_constant,
        dynamics_matrix,
        dynamics_offset,
        dynamics_covariance,
        recurrence_weights,
        recurrence_offsets,
    )
    return model, input_ok


def _categorical_support(model: StructuredHybridModel):
    initial = jnp.isfinite(model.log_initial + model.log_unary[0])
    local = jnp.ones(
        (model.log_unary.shape[0] - 1, model.log_unary.shape[1], model.log_unary.shape[1]),
        dtype=bool,
    )

    def forward(previous, transition):
        current = jnp.any(previous[:, None] & transition, axis=0)
        return current, current

    _, tail = jax.lax.scan(forward, initial, local)
    forward_support = jnp.concatenate([initial[None, :], tail], axis=0)

    def backward(following, transition):
        current = jnp.any(transition & following[None, :], axis=1)
        return current, current

    _, head = jax.lax.scan(
        backward,
        jnp.ones_like(initial),
        local,
        reverse=True,
    )
    backward_support = jnp.concatenate(
        [head, jnp.ones_like(initial)[None, :]], axis=0
    )
    singleton = forward_support & backward_support
    pair = (
        forward_support[:-1, :, None]
        & local
        & backward_support[1:, None, :]
    )
    return singleton, pair


_BELIEF_FIELDS = (
    "cat_log_singleton",
    "cat_singleton",
    "cat_pair",
    "cat_entropy",
    "cat_log_normalizer",
    "cat_proper",
    "cat_support",
    "pair_support",
    "gaussian_mean",
    "gaussian_covariance",
    "gaussian_lag_covariance",
    "gaussian_second_moment",
    "gaussian_lag_second_moment",
    "gaussian_precision_diagonal",
    "gaussian_precision_lower",
    "gaussian_information",
    "gaussian_entropy",
    "gaussian_proper",
)


def _pack_belief(params, plan, belief, singleton_support, pair_support):
    categorical, gaussian = belief
    values = {
        "cat_log_singleton": jnp.where(
            jnp.isfinite(categorical.log_singleton),
            categorical.log_singleton,
            jnp.asarray(NEG, categorical.log_singleton.dtype),
        ),
        "cat_singleton": categorical.singleton,
        "cat_pair": categorical.pair,
        "cat_entropy": categorical.entropy,
        "cat_log_normalizer": categorical.log_normalizer,
        "cat_proper": categorical.proper,
        "cat_support": singleton_support,
        "pair_support": pair_support,
        "gaussian_mean": gaussian.mean,
        "gaussian_covariance": gaussian.covariance,
        "gaussian_lag_covariance": gaussian.lag_covariance,
        "gaussian_second_moment": gaussian.second_moment,
        "gaussian_lag_second_moment": gaussian.lag_second_moment,
        "gaussian_precision_diagonal": gaussian.precision_diagonal,
        "gaussian_precision_lower": gaussian.precision_lower,
        "gaussian_information": gaussian.information,
        "gaussian_entropy": gaussian.entropy,
        "gaussian_proper": gaussian.proper,
    }
    for field, value in values.items():
        params[_key(plan, field)] = value
    return params


def unpack_belief(params, plan: StructuredHybridPlan) -> StructuredHybridBelief:
    categorical = CategoricalTrajectory(
        params[_key(plan, "cat_log_singleton")],
        params[_key(plan, "cat_singleton")],
        params[_key(plan, "cat_pair")],
        params[_key(plan, "cat_log_normalizer")],
        params[_key(plan, "cat_entropy")],
        params[_key(plan, "cat_proper")],
    )
    gaussian = GaussianTrajectory(
        params[_key(plan, "gaussian_mean")],
        params[_key(plan, "gaussian_covariance")],
        params[_key(plan, "gaussian_lag_covariance")],
        params[_key(plan, "gaussian_second_moment")],
        params[_key(plan, "gaussian_lag_second_moment")],
        params[_key(plan, "gaussian_precision_diagonal")],
        params[_key(plan, "gaussian_precision_lower")],
        params[_key(plan, "gaussian_information")],
        params[_key(plan, "gaussian_entropy")],
        params[_key(plan, "gaussian_proper")],
    )
    return StructuredHybridBelief(categorical, gaussian)


def initialize_runtime(arena: Arena, g_obs, g_mask, params):
    """Install one finite initial coordinate per derived D10 plan."""
    out = dict(params)
    for plan in arena.structured_hybrid_plans:
        model, input_ok = component_model(arena, plan, g_obs, g_mask, out)
        belief = structured_hybrid_initial(model)
        objective, objective_ok = structured_hybrid_evaluate(belief, model)
        singleton_support, pair_support = _categorical_support(model)
        proper = input_ok & objective_ok
        out = _pack_belief(out, plan, belief, singleton_support, pair_support)
        dtype = belief.gaussian.mean.dtype
        out[_key(plan, "objective")] = jnp.where(
            proper, objective, jnp.finfo(dtype).max
        )
        out[_key(plan, "categorical_residual")] = jnp.asarray(0.0, dtype)
        out[_key(plan, "gaussian_residual")] = jnp.asarray(0.0, dtype)
        out[_key(plan, "objective_residual")] = jnp.asarray(0.0, dtype)
        out[_key(plan, "status")] = jnp.where(
            proper, SWEEP_FINITE, SWEEP_DIVERGED
        )
        out[_key(plan, "run_status")] = jnp.where(proper, 1, SWEEP_DIVERGED)
    return out


def _max_abs(*arrays):
    values = [jnp.max(jnp.abs(value), initial=0.0) for value in arrays]
    return jnp.max(jnp.stack(values), initial=0.0)


def component_step(arena: Arena, plan: StructuredHybridPlan, g_obs, g_mask, params):
    """Run one unfused D10 sweep and install only a finite coordinate."""
    model, input_ok = component_model(arena, plan, g_obs, g_mask, params)
    previous = unpack_belief(params, plan)
    sweep = structured_hybrid_sweep(previous, model)
    categorical_residual = _max_abs(
        sweep.belief.categorical.singleton - previous.categorical.singleton,
        sweep.belief.categorical.pair - previous.categorical.pair,
    )
    gaussian_residual = _max_abs(
        sweep.belief.gaussian.mean - previous.gaussian.mean,
        sweep.belief.gaussian.covariance - previous.gaussian.covariance,
        sweep.belief.gaussian.lag_covariance - previous.gaussian.lag_covariance,
    )
    objective_residual = jnp.abs(
        sweep.objective - params[_key(plan, "objective")]
    )
    proper = (
        input_ok
        & (sweep.status == SWEEP_FINITE)
        & jnp.isfinite(categorical_residual)
        & jnp.isfinite(gaussian_residual)
        & jnp.isfinite(objective_residual)
        & jnp.isfinite(sweep.objective)
    )
    support, pair_support = _categorical_support(model)
    candidate = _pack_belief(
        dict(params), plan, sweep.belief, support, pair_support
    )
    out = dict(params)
    for field in _BELIEF_FIELDS:
        key = _key(plan, field)
        out[key] = jnp.where(proper, candidate[key], params[key])
    dtype = sweep.belief.gaussian.mean.dtype
    maximum = jnp.finfo(dtype).max
    out[_key(plan, "objective")] = jnp.where(
        proper, sweep.objective, params[_key(plan, "objective")]
    )
    out[_key(plan, "categorical_residual")] = jnp.where(
        proper, categorical_residual, maximum
    )
    out[_key(plan, "gaussian_residual")] = jnp.where(
        proper, gaussian_residual, maximum
    )
    out[_key(plan, "objective_residual")] = jnp.where(
        proper, objective_residual, maximum
    )
    out[_key(plan, "status")] = jnp.where(
        proper, SWEEP_FINITE, SWEEP_DIVERGED
    )
    out[_key(plan, "run_status")] = jnp.where(proper, 1, SWEEP_DIVERGED)
    return (
        out,
        out[_key(plan, "categorical_residual")],
        out[_key(plan, "gaussian_residual")],
        out[_key(plan, "objective_residual")],
        out[_key(plan, "objective")],
        out[_key(plan, "status")],
    )


def component_objective(
    arena: Arena,
    plan: StructuredHybridPlan,
    g_obs,
    g_mask,
    params,
):
    """Evaluate the installed coordinate under the current model and data."""
    model, input_ok = component_model(arena, plan, g_obs, g_mask, params)
    objective, belief_ok = structured_hybrid_evaluate(
        unpack_belief(params, plan), model
    )
    return jnp.where(input_ok & belief_ok, objective, jnp.inf)


def component_learning_readouts(
    arena: Arena,
    plan: StructuredHybridPlan,
    g_obs,
    g_mask,
    params,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Return factor-local D10 statistics from one installed coordinate."""
    belief = unpack_belief(params, plan)
    categorical, gaussian = belief
    prefix = lambda field: _key(plan, field)
    runtime_ok = bool(
        int(np.asarray(params[prefix("status")])) == SWEEP_FINITE
        and int(np.asarray(params[prefix("run_status")])) == 0
        and bool(np.asarray(categorical.proper))
        and bool(np.asarray(gaussian.proper))
    )
    topology = plan.topology
    singleton = np.asarray(categorical.singleton)
    pair = np.asarray(categorical.pair)
    mean = np.asarray(gaussian.mean)
    covariance = np.asarray(gaussian.covariance)
    lag_covariance = np.asarray(gaussian.lag_covariance)
    second = np.asarray(gaussian.second_moment)
    lag_second = np.asarray(gaussian.lag_second_moment)
    dtype = np.result_type(singleton, mean)
    dimension = topology.dimension
    # Probability normalization error grows with categorical additions, never
    # with the unrelated scale of continuous moments.
    probability_tolerance = (
        64.0 * max(1, topology.cardinality) * np.finfo(dtype).eps
    )
    moment_factor = 64.0 * max(1, dimension) * np.finfo(dtype).eps
    psd_factor = 64.0 * max(1, dimension) ** 2 * np.finfo(dtype).eps
    shapes_ok = (
        singleton.shape == (len(topology.state_edges), topology.cardinality)
        and pair.shape == (
            len(topology.recurrence_members),
            topology.cardinality,
            topology.cardinality,
        )
        and mean.shape == (len(topology.state_edges), dimension)
        and covariance.shape == (len(topology.state_edges), dimension, dimension)
        and second.shape == (len(topology.state_edges), dimension, dimension)
        and lag_covariance.shape == (
            len(topology.recurrence_members), dimension, dimension
        )
        and lag_second.shape == (
            len(topology.recurrence_members), dimension, dimension
        )
    )
    finite_ok = shapes_ok and all(
        np.all(np.isfinite(value))
        for value in (
            singleton,
            pair,
            mean,
            covariance,
            lag_covariance,
            second,
            lag_second,
        )
    )
    probability_ok = False
    gaussian_ok = False
    if finite_ok:
        probability_ok = bool(
            np.all(singleton >= 0.0)
            and np.all(pair >= 0.0)
            and np.allclose(
                singleton.sum(axis=1),
                1.0,
                atol=probability_tolerance,
                rtol=0.0,
            )
            and np.allclose(
                pair.sum(axis=(1, 2)),
                1.0,
                atol=probability_tolerance,
                rtol=0.0,
            )
            and np.allclose(
                pair.sum(axis=2),
                singleton[:-1],
                atol=probability_tolerance,
                rtol=0.0,
            )
            and np.allclose(
                pair.sum(axis=1),
                singleton[1:],
                atol=probability_tolerance,
                rtol=0.0,
            )
        )

        def locally_consistent(actual, expected, *terms):
            local_scale = np.maximum(1.0, np.maximum(np.abs(actual), np.abs(expected)))
            for term in terms:
                local_scale = np.maximum(local_scale, np.abs(term))
            return bool(
                np.all(np.abs(actual - expected) <= moment_factor * local_scale)
            )

        mean_outer = np.einsum("ti,tj->tij", mean, mean)
        lag_outer = np.einsum("ti,tj->tij", mean[1:], mean[:-1])
        expected_second = covariance + mean_outer
        expected_lag = lag_covariance + lag_outer
        covariance_scales = np.maximum(
            1.0,
            np.max(np.abs(covariance), axis=(1, 2)),
        )
        gaussian_ok = bool(
            locally_consistent(
                covariance,
                covariance.swapaxes(1, 2),
            )
            and locally_consistent(
                second,
                expected_second,
                covariance,
                mean_outer,
            )
            and locally_consistent(
                lag_second,
                expected_lag,
                lag_covariance,
                lag_outer,
            )
            and np.all(
                np.linalg.eigvalsh(covariance)
                >= -psd_factor * covariance_scales[:, None]
            )
        )
    coordinate_ok = finite_ok and probability_ok and gaussian_ok
    if not runtime_ok or not coordinate_ok:
        raise ValueError(
            "D10 learning requires one proper, consistent, converged installed "
            "coordinate"
        )
    objective = component_objective(arena, plan, g_obs, g_mask, params)
    if not np.isfinite(np.asarray(objective)):
        raise ValueError(
            "D10 learning requires one proper, consistent, converged installed "
            "coordinate"
        )
    feature_second = np.zeros(
        (len(mean) - 1, dimension + 1, dimension + 1), dtype=dtype
    )
    feature_second[:, :-1, :-1] = second[:-1]
    feature_second[:, :-1, -1] = mean[:-1]
    feature_second[:, -1, :-1] = mean[:-1]
    feature_second[:, -1, -1] = 1.0
    feature_mean = np.concatenate(
        [mean[:-1], np.ones((len(mean) - 1, 1), dtype=dtype)], axis=1
    )

    gaussian_rows: dict[str, dict[str, np.ndarray]] = {}
    for index, name in enumerate(topology.dynamics_members):
        responsibility = singleton[index + 1]
        cross = np.concatenate(
            [lag_second[index], mean[index + 1, :, None]], axis=1
        )
        gaussian_rows[name] = {
            "responsibility": responsibility.copy(),
            "Syy": responsibility[:, None, None] * second[index + 1],
            "Syx": responsibility[:, None, None] * cross,
            "Sxx": responsibility[:, None, None] * feature_second[index],
            "proper": np.asarray(True),
        }

    emission_address = arena.member_of[
        topology.emission_members[0]
    ].parameter("matrix")
    observation_dimension = np.asarray(
        params[emission_address.key][emission_address.index]
    ).shape[0]
    observations = np.asarray(g_obs)[
        arena.g_obs_indices(plan.observation_rows, observation_dimension)
    ]
    masks = np.asarray(g_mask)[plan.observation_rows]
    for index, name in enumerate(topology.emission_members):
        augmented_second = np.block(
            [
                [second[index], mean[index, :, None]],
                [mean[index, None, :], np.ones((1, 1), dtype=dtype)],
            ]
        )
        observation = observations[index]
        gaussian_rows[name] = {
            "Syy": np.outer(observation, observation),
            "Syx": np.concatenate(
                [np.outer(observation, mean[index]), observation[:, None]],
                axis=1,
            ),
            "Sxx": augmented_second,
            "mask": np.asarray(masks[index]),
            "proper": np.asarray(True),
        }

    weights, weights_ok = _parameter_rows(
        arena, params, topology.recurrence_members, "weights"
    )
    offsets, offsets_ok = _parameter_rows(
        arena, params, topology.recurrence_members, "offsets"
    )
    recurrence = recurrence_statistics(
        categorical.pair, gaussian, weights, offsets
    )
    recurrence_ok = bool(
        np.asarray(recurrence.proper & weights_ok & offsets_ok)
    )
    recurrence_rows: dict[str, dict[str, np.ndarray]] = {}
    for index, name in enumerate(topology.recurrence_members):
        exposure = np.asarray(recurrence.exposure[index])
        omega = np.asarray(recurrence.omega[index])
        signed = np.asarray(recurrence.signed_success[index])
        recurrence_rows[name] = {
            "omega": omega,
            "exposure": exposure,
            "signed_success": signed,
            "linear": signed[..., None] * feature_mean[index],
            "quadratic": (
                exposure[..., None, None]
                * omega[..., None, None]
                * feature_second[index]
            ),
            "proper": np.asarray(recurrence_ok),
            "probability_ok": np.asarray(recurrence_ok),
            "drive_ok": np.asarray(recurrence_ok),
            "parameter_ok": np.asarray(
                bool(np.asarray(weights_ok & offsets_ok))
            ),
            "second_ok": np.asarray(recurrence_ok),
            "finite_output_ok": np.asarray(recurrence_ok),
        }

    return {
        "node_beliefs": {
            topology.initial_regime_member: singleton[0].copy()
        },
        "gaussian_node_stats": gaussian_rows,
        "recurrence_weight_stats": recurrence_rows,
    }


def make_step(arena: Arena):
    if len(arena.structured_hybrid_plans) != 1:
        raise ValueError("D10 runtime requires exactly one whole-template component")
    plan = arena.structured_hybrid_plans[0]

    def step(g_obs, g_mask, params):
        return component_step(arena, plan, g_obs, g_mask, params)

    return step


__all__ = [
    "component_model",
    "component_learning_readouts",
    "component_objective",
    "component_step",
    "initialize_runtime",
    "make_step",
    "runtime_key",
    "unpack_belief",
]

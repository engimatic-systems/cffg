"""Learned D10 statistics, installation, and hostile-coordinate boundaries."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from cffg import conj
from cffg.compiler import compile_template
from cffg.core.compose import scoped
from cffg.engine import Engine, RunStatus, _replace
from cffg.learn import FAMILIES, Learnables
from cffg.models.rslds import RSLDSLearning, rslds
from cffg.models.rslds_fit import (
    _attempt_evidence,
    _run_inner,
    vague_rslds_fit_priors,
)
from cffg.structured_hybrid_runtime import runtime_key


def _model(steps=5, card=3, dimension=2, *, dtype=np.float64):
    matrices = np.stack(
        [
            np.asarray([[0.62, 0.11], [-0.07, 0.51]], dtype=dtype)
            + 0.04 * index * np.eye(dimension, dtype=dtype)
            for index in range(card)
        ]
    )
    offsets = np.linspace(-0.2, 0.25, card * dimension, dtype=dtype).reshape(
        card, dimension
    )
    covariance = np.stack(
        [(0.35 + 0.08 * index) * np.eye(dimension, dtype=dtype) for index in range(card)]
    )
    weights = np.linspace(
        -0.35, 0.4, card * (card - 1) * dimension, dtype=dtype
    ).reshape(card, card - 1, dimension)
    recurrence_offsets = np.linspace(
        -0.25, 0.3, card * (card - 1), dtype=dtype
    ).reshape(card, card - 1)
    template, arrays = rslds(
        steps,
        matrices,
        offsets,
        covariance,
        weights,
        recurrence_offsets,
        np.asarray([[1.0, 0.2], [-0.15, 0.8]], dtype=dtype),
        np.asarray([[0.45, 0.04], [0.04, 0.55]], dtype=dtype),
        np.asarray([0.1, -0.2], dtype=dtype),
        np.asarray([[0.9, 0.1], [0.1, 0.8]], dtype=dtype),
        np.full(card, 1.0 / card, dtype=dtype),
        emission_offset=np.asarray([0.05, -0.1], dtype=dtype),
    )
    return template, arrays


def _converged(dtype=np.float64, *, steps=5, card=3):
    template, arrays = _model(steps, card, dtype=dtype)
    engine = Engine(compile_template(template, arrays=arrays))
    values = np.linspace(-0.4, 0.5, steps * 2, dtype=dtype).reshape(steps, 2)
    state = engine.bind_data(
        engine.init_state(), {f"y{index}": row for index, row in enumerate(values)}
    )
    result = engine.run(state, 80, tol=2e-5)
    assert result.status is RunStatus.CONVERGED
    return engine, result, values


def _manual_records(engine, state, observations):
    plan = engine.arena.structured_hybrid_plans[0]
    belief = next(iter(engine.structured_hybrid_beliefs(state).values()))
    singleton = belief["singleton_probability"]
    pair = belief["pair_probability"]
    mean = belief["gaussian_mean"]
    second = belief["gaussian_second_moment"]
    lag = belief["gaussian_lag_second_moment"]
    dimension = mean.shape[1]
    feature_second = np.zeros((len(mean) - 1, dimension + 1, dimension + 1))
    feature_second[:, :-1, :-1] = second[:-1]
    feature_second[:, :-1, -1] = mean[:-1]
    feature_second[:, -1, :-1] = mean[:-1]
    feature_second[:, -1, -1] = 1.0
    feature_mean = np.concatenate([mean[:-1], np.ones((len(mean) - 1, 1))], axis=1)

    gaussian = {}
    for index, name in enumerate(plan.topology.dynamics_members):
        responsibility = singleton[index + 1]
        cross = np.concatenate([lag[index], mean[index + 1, :, None]], axis=1)
        gaussian[name] = {
            "responsibility": responsibility,
            "Syy": responsibility[:, None, None] * second[index + 1],
            "Syx": responsibility[:, None, None] * cross,
            "Sxx": responsibility[:, None, None] * feature_second[index],
            "proper": np.asarray(True),
        }
    for index, name in enumerate(plan.topology.emission_members):
        augmented = np.block(
            [[second[index], mean[index, :, None]], [mean[index, None, :], np.ones((1, 1))]]
        )
        y = observations[index]
        gaussian[name] = {
            "Syy": np.outer(y, y),
            "Syx": np.concatenate([np.outer(y, mean[index]), y[:, None]], axis=1),
            "Sxx": augmented,
            "mask": np.asarray(1.0),
            "proper": np.asarray(True),
        }

    runtime = engine.learning_readout(state, "recurrence_weight_stats")
    recurrence = {}
    for index, name in enumerate(plan.topology.recurrence_members):
        exposure = np.cumsum(pair[index, :, ::-1], axis=1)[:, ::-1,][..., :-1]
        signed = pair[index, :, :-1] - 0.5 * exposure
        omega = runtime[name]["omega"]
        recurrence[name] = {
            "linear": signed[..., None] * feature_mean[index],
            "quadratic": exposure[..., None, None] * omega[..., None, None] * feature_second[index],
            "proper": np.asarray(True),
        }
    return plan, gaussian, recurrence


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_d10_readouts_and_pooled_naturals_match_independent_coordinate(dtype):
    engine, result, observations = _converged(dtype)
    plan, manual_gaussian, manual_recurrence = _manual_records(
        engine, result.state, observations
    )
    actual_gaussian = engine.learning_readout(result.state, "gaussian_node_stats")
    actual_recurrence = engine.learning_readout(
        result.state, "recurrence_weight_stats"
    )
    actual_nodes = engine.learning_readout(result.state, "node_beliefs")
    belief = next(iter(engine.structured_hybrid_beliefs(result.state).values()))
    np.testing.assert_allclose(
        actual_nodes[plan.topology.initial_regime_member],
        belief["singleton_probability"][0],
    )
    for name, expected in manual_gaussian.items():
        for key in expected:
            np.testing.assert_allclose(actual_gaussian[name][key], expected[key])
    for name, expected in manual_recurrence.items():
        for key in ("linear", "quadratic", "proper"):
            np.testing.assert_allclose(actual_recurrence[name][key], expected[key])

    priors = vague_rslds_fit_priors(3, 2, 2).learning_priors()
    learning = Learnables(engine, priors=priors)
    before = {
        name: {key: value.copy() for key, value in rows.items()}
        for name, rows in learning.initial_params.items()
    }
    learning.replace(engine, result)
    for name, family_name, records in (
        ("dynamics", "component_mniw", manual_gaussian),
        ("recurrence", "pg_weights", manual_recurrence),
        ("emission", "mniw", manual_gaussian),
    ):
        learn_plan = learning.plans[name]
        family = FAMILIES[family_name]
        selected = [family.stats(learn_plan.ftype, records[source]) for source in learn_plan.sources]
        expected = family.fold(before[name], family.pool(selected))
        for key, value in expected.items():
            np.testing.assert_allclose(learning.params[name][key], value)

    accumulated = Learnables(engine, priors=priors)
    accumulated.accumulate(engine, result)
    accumulated.accumulate(engine, result)
    np.testing.assert_allclose(
        accumulated.params["initial_regime"]["alpha"],
        before["initial_regime"]["alpha"]
        + 2.0 * belief["singleton_probability"][0],
    )
    for name, family_name, records in (
        ("dynamics", "component_mniw", manual_gaussian),
        ("recurrence", "pg_weights", manual_recurrence),
        ("emission", "mniw", manual_gaussian),
    ):
        learn_plan = accumulated.plans[name]
        family = FAMILIES[family_name]
        selected = [
            family.stats(learn_plan.ftype, records[source])
            for source in learn_plan.sources
        ]
        doubled = {
            key: 2.0 * value
            for key, value in family.pool(selected).items()
        }
        expected = family.fold(before[name], doubled)
        for key, value in expected.items():
            np.testing.assert_allclose(accumulated.params[name][key], value)


def test_scoped_tied_sources_and_atomic_write_use_d10_then_mark_it_stale():
    template, arrays = _model()
    template = scoped(template, "renamed")
    arrays = {f"renamed/{name}": value for name, value in arrays.items()}
    engine = Engine(compile_template(template, arrays=arrays))
    assert len(engine.arena.structured_hybrid_plans) == 1
    priors = {
        f"renamed/{name}": value
        for name, value in vague_rslds_fit_priors(3, 2, 2).learning_priors().items()
    }
    learning = Learnables(engine, priors=priors)
    state = engine.init_state()
    written = learning.write(engine, state.params)
    plan = engine.arena.structured_hybrid_plans[0]
    assert int(written[runtime_key(plan, "status")]) == 3
    refreshed = engine.reset_structured_hybrid(_replace(state, params=written))
    assert int(refreshed.params[runtime_key(plan, "status")]) == 0
    unchanged = learning.write(engine, refreshed.params, only=())
    assert int(unchanged[runtime_key(plan, "status")]) == 0


def test_d10_learning_rejects_inconsistent_or_nonfinite_installed_moments():
    engine, result, _observations = _converged()
    plan = engine.arena.structured_hybrid_plans[0]
    for field, mutate in (
        ("cat_pair", lambda value: value.at[0, 0, 0].add(0.2)),
        ("cat_pair", lambda value: value[..., :-1]),
        ("gaussian_covariance", lambda value: value.at[0, 0, 1].add(0.2)),
        ("gaussian_mean", lambda value: value.at[0, 0].set(jnp.nan)),
    ):
        params = dict(result.state.params)
        key = runtime_key(plan, field)
        params[key] = mutate(params[key])
        changed = _replace(result.state, params=params)
        with pytest.raises(ValueError, match="proper, consistent"):
            engine.learning_readout(changed, "gaussian_node_stats")


def test_float32_pair_consistency_is_independent_of_gaussian_moment_scale():
    engine, result, _observations = _converged(np.float32)
    plan = engine.arena.structured_hybrid_plans[0]
    params = dict(result.state.params)
    key = runtime_key(plan, "cat_pair")
    params[key] = params[key].at[0, 0, 0].add(np.float32(1.0e-3))

    with pytest.raises(ValueError, match="proper, consistent"):
        engine.learning_readout(
            _replace(result.state, params=params),
            "gaussian_node_stats",
        )


def test_exact_zero_d10_source_and_component_mass_retains_prior_bits():
    engine, result, _observations = _converged()
    plan = engine.arena.structured_hybrid_plans[0]
    params = dict(result.state.params)
    singleton = jnp.zeros_like(params[runtime_key(plan, "cat_singleton")])
    singleton = singleton.at[:, 0].set(1.0)
    pair = jnp.zeros_like(params[runtime_key(plan, "cat_pair")])
    pair = pair.at[:, 0, 0].set(1.0)
    params[runtime_key(plan, "cat_singleton")] = singleton
    params[runtime_key(plan, "cat_pair")] = pair
    params[runtime_key(plan, "cat_log_singleton")] = jnp.where(
        singleton > 0,
        jnp.asarray(0.0, singleton.dtype),
        jnp.asarray(-1.0e30, singleton.dtype),
    )
    params[runtime_key(plan, "cat_entropy")] = jnp.asarray(0.0, singleton.dtype)
    state = _replace(result.state, params=params)
    oracle_result = dataclasses.replace(result, state=state)
    priors = vague_rslds_fit_priors(3, 2, 2).learning_priors()
    learning = Learnables(engine, priors=priors)
    initial = {
        name: {key: value.copy() for key, value in rows.items()}
        for name, rows in learning.params.items()
    }
    learning.replace(engine, oracle_result)
    np.testing.assert_array_equal(
        learning.params["initial_regime"]["alpha"][1:],
        initial["initial_regime"]["alpha"][1:],
    )
    for key in learning.params["dynamics"]:
        np.testing.assert_array_equal(
            learning.params["dynamics"][key][1:],
            initial["dynamics"][key][1:],
        )
    for key in learning.params["recurrence"]:
        np.testing.assert_array_equal(
            learning.params["recurrence"][key][1:],
            initial["recurrence"][key][1:],
        )


def test_all_masked_d10_emissions_retain_asymmetric_prior_bits():
    template, arrays = _model()
    engine = Engine(compile_template(template, arrays=arrays))
    result = engine.run(engine.init_state(), 80, tol=2e-5)
    result.require_converged()
    priors = vague_rslds_fit_priors(3, 2, 2).learning_priors()
    priors["emission"] = conj.MNIW(
        np.asarray([[0.3, -0.2, 0.1], [-0.4, 0.25, 0.15]], dtype=np.float32),
        np.asarray(
            [[1.7, 0.2, -0.1], [0.2, 1.4, 0.15], [-0.1, 0.15, 1.2]],
            dtype=np.float32,
        ),
        np.asarray([[1.8, 0.35], [0.35, 1.3]], dtype=np.float32),
        6.0,
    )
    learning = Learnables(engine, priors=priors)
    before = {key: value.copy() for key, value in learning.params["emission"].items()}

    stats = engine.learning_readout(result.state, "gaussian_node_stats")
    plan = engine.arena.structured_hybrid_plans[0]
    assert all(stats[name]["mask"] == 0.0 for name in plan.topology.emission_members)
    learning.replace(engine, result, only=("emission",))
    for key, value in before.items():
        np.testing.assert_array_equal(learning.params["emission"][key], value)

    invalid = {
        "Syy": np.zeros((2, 2)),
        "Syx": np.ones((2, 3)),
        "Sxx": np.zeros((3, 3)),
        "n": np.asarray(0.0),
    }
    with pytest.raises(ValueError, match="zero-occurrence.*nonzero"):
        FAMILIES["mniw"].fold(before, invalid)


def test_d10_retry_boundary_rejects_damping_and_install_rebuilds_coordinate():
    engine, result, _observations = _converged()
    priors = vague_rslds_fit_priors(3, 2, 2)
    learning = RSLDSLearning(
        initial_alpha=np.ones(3),
        dynamics=priors.dynamics,
        recurrence=priors.recurrence,
    )
    fresh = engine.init_state()
    installed = learning.install(engine, fresh)
    plan = engine.arena.structured_hybrid_plans[0]
    assert int(installed.params[runtime_key(plan, "status")]) == 0
    with pytest.raises(ValueError, match="do not accept damping"):
        engine.run(installed, 1, damp=0.5)
    exhausted = engine.run(installed, 0, tol=1e-8)
    assert exhausted.status is RunStatus.BUDGET_EXHAUSTED
    assert _attempt_evidence(engine, exhausted, 0.0).gaussian_beliefs_proper
    _bounded, attempts = _run_inner(
        engine,
        installed,
        max_iterations=1,
        tolerance=1e-8,
        damping=0.0,
        max_retries=4,
    )
    assert len(attempts) == 1
    assert result.status is RunStatus.CONVERGED


def test_d10_k8_t100_readout_is_linear_or_pair_quadratic_not_path_shaped():
    engine, result, _observations = _converged(np.float32, steps=100, card=8)
    plan = engine.arena.structured_hybrid_plans[0]
    belief = next(iter(engine.structured_hybrid_beliefs(result.state).values()))
    assert belief["singleton_probability"].shape == (100, 8)
    assert belief["pair_probability"].shape == (99, 8, 8)
    stats = engine.learning_readout(result.state, "gaussian_node_stats")
    assert len([name for name in stats if name in plan.topology.dynamics_members]) == 99

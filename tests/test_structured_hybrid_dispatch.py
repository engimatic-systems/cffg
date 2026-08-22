"""Production topology, dispatch, status, and readout tests for D10."""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from cffg.colony import Colony
from cffg.compiler import compile_template
from cffg.engine import Engine, RunStatus, _replace
from cffg.models.rslds import rslds
from cffg.structured_hybrid_topology import classify_structured_hybrid


def _parameters(card=2, dimension=1, observation_dimension=1, *, dtype=np.float64):
    matrices = np.stack(
        [
            (0.65 + 0.2 * index / max(card - 1, 1))
            * np.eye(dimension, dtype=dtype)
            for index in range(card)
        ]
    )
    dynamics_offset = np.linspace(-0.1, 0.1, card, dtype=dtype)[:, None]
    dynamics_offset = np.broadcast_to(
        dynamics_offset, (card, dimension)
    ).copy()
    dynamics_covariance = np.stack(
        [
            (0.25 + 0.05 * index) * np.eye(dimension, dtype=dtype)
            for index in range(card)
        ]
    )
    weights = np.zeros((card, card - 1, dimension), dtype=dtype)
    for source in range(card):
        for stick in range(card - 1):
            weights[source, stick] = (
                0.15 * (source + 1) - 0.08 * stick
            )
    offsets = np.linspace(
        -0.25, 0.25, card * (card - 1), dtype=dtype
    ).reshape(card, card - 1)
    emission = np.zeros((observation_dimension, dimension), dtype=dtype)
    for index in range(min(observation_dimension, dimension)):
        emission[index, index] = 1.0
    if not np.any(emission):
        emission[:] = 1.0 / np.sqrt(dimension)
    return {
        "dynamics_matrix": matrices,
        "dynamics_offset": dynamics_offset,
        "dynamics_covariance": dynamics_covariance,
        "recurrence_weights": weights,
        "recurrence_offsets": offsets,
        "emission_matrix": emission,
        "emission_covariance": 0.4
        * np.eye(observation_dimension, dtype=dtype),
        "initial_state_mean": np.zeros(dimension, dtype=dtype),
        "initial_state_covariance": np.eye(dimension, dtype=dtype),
        "initial_regime_probability": np.full(card, 1.0 / card, dtype=dtype),
    }


def _model(steps=4, card=2, dimension=1, observation_dimension=1, *, dtype=np.float64):
    template, arrays = rslds(
        steps,
        **_parameters(card, dimension, observation_dimension, dtype=dtype),
    )
    return dataclasses.replace(template, learnables={}), arrays


def _observations(steps, observation_dimension=1, dtype=np.float64):
    values = np.linspace(-0.3, 0.35, steps * observation_dimension, dtype=dtype)
    return {
        f"y{step}": values.reshape(steps, observation_dimension)[step]
        for step in range(steps)
    }


def _run(schedule, *, steps=4, card=2, dimension=1, dtype=np.float64):
    template, arrays = _model(steps, card, dimension, dtype=dtype)
    engine = Engine(compile_template(template, arrays=arrays), schedule=schedule)
    state = engine.bind_data(
        engine.init_state(), _observations(steps, dtype=dtype)
    )
    result = engine.run(state, 40, tol=1e-8)
    return engine, result


def _only_belief(engine, state):
    return next(iter(engine.structured_hybrid_beliefs(state).values()))


def test_compiler_derives_only_the_exact_inference_boundary():
    template, arrays = _model(4, 3, 2, 2)
    arena = compile_template(template, arrays=arrays)
    assert len(arena.structured_hybrid_plans) == 1
    plan = arena.structured_hybrid_plans[0]
    assert plan.topology.recurrence_members == (
        "switch1",
        "switch2",
        "switch3",
    )
    assert plan.topology.dynamics_members == (
        "dynamics1",
        "dynamics2",
        "dynamics3",
    )
    assert plan.topology.cardinality == 3
    assert plan.topology.dimension == 2

    learned, _ = rslds(4, **_parameters(3, 2, 2))
    learned_arena = compile_template(learned, arrays=arrays)
    assert not learned_arena.structured_hybrid_plans
    assert "learning retains D3/D9" in learned_arena.structured_hybrid_reason


def test_topology_rejects_self_loop_parallel_untied_and_extra_attachment():
    template, _ = _model(4)
    self_loop = dataclasses.replace(
        template,
        factors={
            **template.factors,
            "switch1": dataclasses.replace(
                template.factors["switch1"],
                edges={
                    **template.factors["switch1"].edges,
                    "out": template.factors["switch1"].edges["in"],
                },
            ),
        },
    )
    assert "self-loop" in classify_structured_hybrid(self_loop).reason

    parallel = dataclasses.replace(
        template,
        factors={
            **template.factors,
            "switch2": dataclasses.replace(
                template.factors["switch2"],
                edges={
                    **template.factors["switch2"].edges,
                    "in": template.factors["switch1"].edges["in"],
                    "out": template.factors["switch1"].edges["out"],
                },
            ),
        },
    )
    assert "parallel or ambiguous" in classify_structured_hybrid(parallel).reason

    untied = dataclasses.replace(
        template,
        factors={
            **template.factors,
            "emission1": dataclasses.replace(
                template.factors["emission1"],
                param_blobs={
                    **template.factors["emission1"].param_blobs,
                    "matrix": "different_emission_matrix",
                },
            ),
        },
    )
    assert "emission parameters are not tied" in classify_structured_hybrid(untied).reason

    extra = dataclasses.replace(
        template,
        factors={
            **template.factors,
            "extra_prior": dataclasses.replace(
                template.factors["z0_prior"], name="extra_prior"
            ),
        },
    )
    assert "initial law" in classify_structured_hybrid(extra).reason


def test_flood_and_scheduled_dispatch_install_the_same_k2_coordinate():
    flood_engine, flood = _run("flood")
    scheduled_engine, scheduled = _run("auto")
    assert flood.status is RunStatus.CONVERGED
    assert scheduled.status is RunStatus.CONVERGED
    assert scheduled_engine.schedule.kind == "mixed"
    left = _only_belief(flood_engine, flood.state)
    right = _only_belief(scheduled_engine, scheduled.state)
    for key in (
        "singleton_probability",
        "pair_probability",
        "gaussian_mean",
        "gaussian_covariance",
        "gaussian_lag_covariance",
        "gaussian_lag_second_moment",
    ):
        np.testing.assert_allclose(left[key], right[key], rtol=0, atol=0)
    assert float(left["objective"]) == flood_engine.bfe(flood.state)
    assert float(right["objective"]) == scheduled_engine.bfe(scheduled.state)


def test_k3_readout_has_consistent_source_and_destination_axes():
    engine, result = _run("auto", steps=5, card=3, dimension=2)
    assert result.status is RunStatus.CONVERGED
    belief = _only_belief(engine, result.state)
    pair = belief["pair_probability"]
    singleton = belief["singleton_probability"]
    np.testing.assert_allclose(pair.sum(axis=2), singleton[:-1], atol=1e-10)
    np.testing.assert_allclose(pair.sum(axis=1), singleton[1:], atol=1e-10)
    assert pair.shape == (4, 3, 3)
    assert belief["gaussian_lag_second_moment"].shape == (4, 2, 2)


def test_status_distinguishes_budget_convergence_and_divergence():
    template, arrays = _model(5)
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.bind_data(engine.init_state(), _observations(5))
    exhausted = engine.run(state, 1, tol=1e-12)
    assert exhausted.status is RunStatus.BUDGET_EXHAUSTED
    assert np.isfinite(exhausted.free_energies[-1])
    converged = engine.run(state, 40, tol=1e-8)
    assert converged.status is RunStatus.CONVERGED

    plan = engine.arena.structured_hybrid_plans[0]
    address = engine.arena.member_of[
        plan.topology.dynamics_members[0]
    ].parameter("cov")
    bad_params = dict(state.params)
    bad_params[address.key] = bad_params[address.key].at[:, 0, 0].set(-1.0)
    diverged = engine.run(_replace(state, params=bad_params), 3, tol=1e-8)
    assert diverged.status is RunStatus.DIVERGED
    assert diverged.iterations == 1
    assert not bool(_only_belief(engine, diverged.state)["proper"])

    nonfinite = dict(state.params)
    nonfinite[address.key] = nonfinite[address.key].at[0, 0, 0].set(jnp.nan)
    failed = engine.run(_replace(state, params=nonfinite), 3, tol=1e-8)
    assert failed.status is RunStatus.DIVERGED
    assert failed.iterations == 0


def test_exact_zero_support_and_float32_remain_literal():
    template, arrays = _model(4, dtype=np.float32)
    arrays["initial_regime_probability"] = np.array([1.0, 0.0], np.float32)
    engine = Engine(compile_template(template, arrays=arrays))
    state = jax.tree_util.tree_map(
        lambda value: (
            value.astype(jnp.float32)
            if jnp.issubdtype(value.dtype, jnp.inexact)
            else value
        ),
        engine.init_state(),
    )
    state = engine.bind_data(state, _observations(4, dtype=np.float32))
    result = engine.run(state, 50, tol=1e-5)
    assert result.status is RunStatus.CONVERGED
    belief = _only_belief(engine, result.state)
    assert belief["singleton_probability"].dtype == np.float32
    assert not belief["singleton_support"][0, 1]
    assert not belief["pair_support"][0, 1].any()
    assert belief["singleton_probability"][0, 1] == 0.0


def test_jitted_single_and_vmapped_component_steps_match():
    template, arrays = _model(4)
    engine = Engine(compile_template(template, arrays=arrays))
    first = engine.bind_data(engine.init_state(), _observations(4))
    second = engine.bind_data(
        engine.init_state(),
        {name: -value for name, value in _observations(4).items()},
    )

    def one(state):
        params, cat_r, gauss_r, objective_r, objective, status = (
            engine._structured_step(state.g_obs, state.g_mask, state.params)
        )
        return (
            _replace(state, params=params),
            cat_r,
            gauss_r,
            objective_r,
            objective,
            status,
        )

    batched = jax.tree_util.tree_map(lambda *rows: jnp.stack(rows), first, second)
    vectorized = eqx.filter_vmap(one)(batched)
    singles = [one(first), one(second)]
    for batch_value, single_values in zip(vectorized[1:], zip(*(row[1:] for row in singles))):
        np.testing.assert_allclose(
            np.asarray(batch_value), np.asarray(single_values), rtol=0, atol=0
        )
    for index, single in enumerate(singles):
        batched_state = jax.tree_util.tree_map(
            lambda leaf: leaf[index], vectorized[0]
        )
        for field in ("cat_pair", "gaussian_mean", "objective"):
            plan = engine.arena.structured_hybrid_plans[0]
            key = f"d10:{plan.name}:{field}"
            np.testing.assert_allclose(
                batched_state.params[key], single[0].params[key], rtol=0, atol=0
            )


def test_colony_vectorization_matches_sequential_status_and_coordinate():
    template, arrays = _model(4)
    colony = Colony()
    first = colony.spawn(template, arrays, observations=_observations(4))
    changed = {name: value.copy() for name, value in arrays.items()}
    changed["recurrence_offsets"] = changed["recurrence_offsets"] + 0.1
    second = colony.spawn(
        template,
        changed,
        observations={name: -value for name, value in _observations(4).items()},
    )
    results = colony.run(40, tol=1e-8)
    for instance in (first, second):
        assert results[instance].status is RunStatus.CONVERGED
        engine = colony.engines[colony.instances[instance].key]
        belief = _only_belief(engine, results[instance].state)
        assert bool(belief["proper"])


def test_k8_t100_uses_only_polynomial_trajectory_storage():
    template, arrays = _model(100, 8, 2)
    engine = Engine(compile_template(template, arrays=arrays), schedule="auto")
    result = engine.run(engine.init_state(), 1, tol=None)
    assert result.status is RunStatus.BUDGET_EXHAUSTED
    belief = _only_belief(engine, result.state)
    assert belief["singleton_probability"].shape == (100, 8)
    assert belief["pair_probability"].shape == (99, 8, 8)
    assert belief["gaussian_covariance"].shape == (100, 2, 2)
    assert belief["gaussian_lag_covariance"].shape == (99, 2, 2)
    assert all(
        value.shape != (8,) * 100
        for value in result.state.params.values()
    )

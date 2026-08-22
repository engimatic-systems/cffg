"""Production topology, dispatch, status, and readout tests for D10."""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg._structured_hybrid_types import StructuredHybridModel
from cffg.colony import Colony
from cffg.compiler import compile_template
from cffg.core.ir import VariableSpec
from cffg.engine import Engine, RunStatus, _replace
from cffg.models.rslds import rslds
from cffg.params import set_params
from cffg.structured_hybrid import (
    structured_hybrid_initial,
    structured_hybrid_sweep,
)
from cffg.structured_hybrid_runtime import component_model
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


def _manual_model(arrays, observations, steps, card):
    dtype = np.result_type(*arrays.values(), np.float32)
    pi = np.asarray(arrays["initial_regime_probability"], dtype=dtype)
    log_initial = np.full(card, -np.inf, dtype=dtype)
    positive = pi > 0
    log_initial[positive] = np.log(pi[positive])
    emission = np.asarray(arrays["emission_matrix"], dtype=dtype)
    emission_offset = np.asarray(arrays["emission_offset"], dtype=dtype)
    emission_precision = np.linalg.inv(arrays["emission_covariance"]).astype(dtype)
    dimension = emission.shape[1]
    gaussian_eta = np.zeros((steps, dimension), dtype=dtype)
    gaussian_precision = np.zeros((steps, dimension, dimension), dtype=dtype)
    gaussian_constant = np.zeros(steps, dtype=dtype)
    for step in range(steps):
        value = observations[f"y{step}"]
        if value is None:
            continue
        centered = np.asarray(value, dtype=dtype) - emission_offset
        gaussian_eta[step] = emission.T @ emission_precision @ centered
        gaussian_precision[step] = emission.T @ emission_precision @ emission
        gaussian_constant[step] = (
            0.5 * np.linalg.slogdet(emission_precision)[1]
            - 0.5 * emission.shape[0] * np.log(dtype.type(2.0 * np.pi))
            - 0.5 * centered @ emission_precision @ centered
        )
    initial_mean = np.asarray(arrays["initial_state_mean"], dtype=dtype)
    initial_precision = np.linalg.inv(
        arrays["initial_state_covariance"]
    ).astype(dtype)
    gaussian_eta[0] += initial_precision @ initial_mean
    gaussian_precision[0] += initial_precision
    gaussian_constant[0] += (
        0.5 * np.linalg.slogdet(initial_precision)[1]
        - 0.5 * dimension * np.log(dtype.type(2.0 * np.pi))
        - 0.5 * initial_mean @ initial_precision @ initial_mean
    )
    return StructuredHybridModel(
        log_initial,
        np.zeros((steps, card), dtype=dtype),
        gaussian_eta,
        gaussian_precision,
        gaussian_constant,
        np.asarray(arrays["dynamics_matrix"], dtype=dtype),
        np.asarray(arrays["dynamics_offset"], dtype=dtype),
        np.asarray(arrays["dynamics_covariance"], dtype=dtype),
        np.asarray(arrays["recurrence_weights"], dtype=dtype),
        np.asarray(arrays["recurrence_offsets"], dtype=dtype),
    )


def _cyclic_equality_model():
    template, arrays = _model(4)
    variables = dict(template.variables)
    for name in ("x0_ring1", "x0_ring2", "x0_ring3"):
        variables[name] = VariableSpec(name, dim=1)
    factors = {
        name: factor
        for name, factor in template.factors.items()
        if name not in ("x0_split", "x0_rest_split")
    }
    prototype = template.factors["x0_split"]
    rings = (
        ("x0", "x0_rest", "x0_ring1"),
        ("x0_switch", "x0_ring1", "x0_ring2"),
        ("x0_dynamics", "x0_ring2", "x0_ring3"),
        ("x0_emission", "x0_ring3", "x0_rest"),
    )
    for index, edges in enumerate(rings):
        name = f"ring{index}"
        factors[name] = dataclasses.replace(
            prototype,
            name=name,
            edges=dict(zip(("a", "b", "c"), edges)),
        )
    return (
        dataclasses.replace(template, variables=variables, factors=factors),
        arrays,
    )


def _collapsed_state_model():
    template, arrays = _model(2)
    variables = dict(template.variables)
    variables["x0_merge"] = VariableSpec("x0_merge", dim=1)
    variables.pop("y1")
    factors = {
        name: factor
        for name, factor in template.factors.items()
        if name not in ("x0_split", "x0_rest_split", "emission1")
    }
    prototype = template.factors["x0_split"]
    trees = (
        ("x0", "x0_switch", "x0_rest"),
        ("x0_rest", "x0_dynamics", "x0_merge"),
        ("x0_merge", "x0_emission", "x1"),
    )
    for index, edges in enumerate(trees):
        name = f"merged{index}"
        factors[name] = dataclasses.replace(
            prototype,
            name=name,
            edges=dict(zip(("a", "b", "c"), edges)),
        )
    ports = dict(template.ports)
    ports.pop("y1")
    return dataclasses.replace(
        template,
        variables=variables,
        factors=factors,
        ports=ports,
    ), arrays


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

    renamed = dataclasses.replace(
        template,
        factors={
            f"renamed_{len(template.factors) - index:02d}": dataclasses.replace(
                factor,
                name=f"renamed_{len(template.factors) - index:02d}",
            )
            for index, factor in enumerate(template.factors.values())
        },
    )
    assert classify_structured_hybrid(renamed).topology is not None


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

    equality_cycle, cycle_arrays = _cyclic_equality_model()
    rejected = classify_structured_hybrid(equality_cycle)
    assert rejected.topology is None
    assert "equality closure is cyclic" in rejected.reason
    cycle_arena = compile_template(equality_cycle, arrays=cycle_arrays)
    assert not cycle_arena.structured_hybrid_plans

    collapsed, collapsed_arrays = _collapsed_state_model()
    rejected = classify_structured_hybrid(collapsed)
    assert rejected.topology is None
    assert "input and output share one closure" in rejected.reason
    collapsed_arena = compile_template(collapsed, arrays=collapsed_arrays)
    assert not collapsed_arena.structured_hybrid_plans


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
    np.testing.assert_allclose(
        float(left["objective"]), flood_engine.bfe(flood.state), rtol=1e-14
    )
    np.testing.assert_allclose(
        float(right["objective"]), scheduled_engine.bfe(scheduled.state), rtol=1e-14
    )


@pytest.mark.parametrize(
    ("card", "dimension", "observation_dimension"),
    ((2, 1, 1), (3, 2, 2)),
)
def test_production_extraction_and_dispatch_match_the_pure_authority(
    card, dimension, observation_dimension
):
    steps = 4
    template, arrays = _model(
        steps, card, dimension, observation_dimension
    )
    observations = _observations(steps, observation_dimension)
    if card == 3:
        observations["y2"] = None
    engine = Engine(compile_template(template, arrays=arrays), schedule="auto")
    state = engine.bind_data(engine.init_state(), observations)
    expected_model = _manual_model(arrays, observations, steps, card)
    actual_model, model_ok = component_model(
        engine.arena,
        engine.arena.structured_hybrid_plans[0],
        state.g_obs,
        state.g_mask,
        state.params,
    )
    assert bool(model_ok)
    for actual, expected in zip(actual_model, expected_model, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)

    result = engine.run(state, 40, tol=1e-8)
    expected_belief = structured_hybrid_initial(expected_model)
    expected_sweep = None
    for _ in range(result.iterations):
        expected_sweep = structured_hybrid_sweep(expected_belief, expected_model)
        expected_belief = expected_sweep.belief
    assert expected_sweep is not None
    actual = _only_belief(engine, result.state)
    expected = expected_sweep.belief
    for key, value in (
        ("singleton_probability", expected.categorical.singleton),
        ("pair_probability", expected.categorical.pair),
        ("gaussian_mean", expected.gaussian.mean),
        ("gaussian_covariance", expected.gaussian.covariance),
        ("gaussian_lag_covariance", expected.gaussian.lag_covariance),
    ):
        np.testing.assert_allclose(actual[key], value, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(
        actual["objective"], expected_sweep.objective, rtol=3e-13, atol=3e-13
    )


def test_bfe_evaluates_the_installed_coordinate_under_current_parameters():
    engine, result = _run("auto")
    original = engine.bfe(result.state)
    unchanged = set_params(engine, result.state.params, {})
    assert engine.bfe(_replace(result.state, params=unchanged)) == original
    offsets = np.full((2, 1), 1.75)
    params = set_params(
        engine,
        result.state.params,
        {"switch1": {"offsets": offsets}},
    )
    changed = _replace(result.state, params=params)
    for action in (
        lambda: engine.bfe(changed),
        lambda: engine.structured_hybrid_beliefs(changed),
        lambda: engine.step(changed),
        lambda: engine.run(changed, 1, tol=1e-8),
    ):
        with pytest.raises(RuntimeError, match="reset_structured_hybrid"):
            action()

    refreshed = engine.reset_structured_hybrid(changed)
    assert np.isfinite(engine.bfe(refreshed))
    assert engine.bfe(refreshed) != original
    assert bool(_only_belief(engine, refreshed)["proper"])

    plan = engine.arena.structured_hybrid_plans[0]
    address = engine.arena.member_of[
        plan.topology.dynamics_members[0]
    ].parameter("cov")
    bad_params = dict(result.state.params)
    bad_params[address.key] = bad_params[address.key].at[0, 0, 0].set(jnp.nan)
    nonfinite = _replace(result.state, params=bad_params)
    assert np.isinf(engine.bfe(nonfinite))
    assert not bool(_only_belief(engine, nonfinite)["proper"])


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


def test_rebinding_nonfinite_observation_as_missing_restores_d10_state():
    template, arrays = _model(4)
    engine = Engine(compile_template(template, arrays=arrays))
    observations = _observations(4)
    observations["y0"] = np.asarray([np.nan])
    invalid = engine.bind_data(engine.init_state(), observations)
    assert engine.run(invalid, 2, tol=1e-8).status is RunStatus.DIVERGED

    missing = engine.bind_data(invalid, {"y0": None})
    result = engine.run(missing, 40, tol=1e-8)
    assert result.status is RunStatus.CONVERGED
    assert np.isfinite(engine.bfe(result.state))
    assert bool(_only_belief(engine, result.state)["proper"])


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
    starts = {
        iid: colony.instances[iid].state
        for iid in (first, second)
    }
    engine = colony.engines[colony.instances[first].key]
    expected = {
        iid: engine.run(state, 40, tol=1e-8)
        for iid, state in starts.items()
    }
    results = colony.run(40, tol=1e-8)
    for instance in (first, second):
        assert results[instance].status is RunStatus.CONVERGED
        assert results[instance].status is expected[instance].status
        assert results[instance].iterations == expected[instance].iterations
        belief = _only_belief(engine, results[instance].state)
        sequential = _only_belief(engine, expected[instance].state)
        for key in (
            "singleton_probability",
            "pair_probability",
            "gaussian_mean",
            "gaussian_covariance",
            "gaussian_lag_covariance",
            "objective",
        ):
            np.testing.assert_allclose(
                belief[key], sequential[key], rtol=3e-14, atol=3e-14
            )
        assert bool(belief["proper"])


def test_colony_mixed_batch_preserves_divergence_across_zero_budget_run():
    template, arrays = _model(4)
    colony = Colony()
    instances = [
        colony.spawn(template, arrays, observations=_observations(4))
        for _ in range(2)
    ]
    engine = colony.engines[colony.instances[instances[0]].key]
    plan = engine.arena.structured_hybrid_plans[0]
    address = engine.arena.member_of[
        plan.topology.dynamics_members[0]
    ].parameter("cov")
    state = colony.instances[instances[0]].state
    params = dict(state.params)
    params[address.key] = params[address.key].at[:, 0, 0].set(-1.0)
    colony.instances[instances[0]].state = _replace(state, params=params)

    first = colony.run(1, tol=1e-8)
    assert first[instances[0]].status is RunStatus.DIVERGED
    assert first[instances[1]].status is RunStatus.BUDGET_EXHAUSTED
    assert all(result.iterations == 1 for result in first.values())

    repeated = colony.run(0, tol=1e-8)
    assert repeated[instances[0]].status is RunStatus.DIVERGED
    assert repeated[instances[1]].status is RunStatus.BUDGET_EXHAUSTED
    assert all(result.iterations == 0 for result in repeated.values())
    for iid in instances:
        state = repeated[iid].state
        assert engine.run(state, 0, tol=1e-8).status is repeated[iid].status
    assert (
        _only_belief(engine, repeated[instances[0]].state)["status"]
        == RunStatus.DIVERGED.value
    )


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

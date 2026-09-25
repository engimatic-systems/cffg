"""Functional Part-II mu3 through the rolling-agent trial lifecycle."""

from __future__ import annotations

import functools
import json
from dataclasses import fields, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.agent import RollingAgent
from cffg.compiler import _log_of, compile_template
from cffg.core.serialize import from_json, to_json
from cffg.engine import Engine, InferenceError, RunResult, RunStatus, State, _replace
from cffg.learn import Learnables
from cffg.models.gridworld import build_gridworld
from cffg.models.rollout import build_rollout
from cffg.models.tmaze import build_tmaze


ALPHA0 = np.array([[2.2, 1.5], [1.8, 2.5]])
D0 = np.array([0.6, 0.4])


def binary_builder(*, past: int, steps: int, **kwargs):
    return build_rollout(
        "binary-mu3-agent",
        past=past,
        steps=steps,
        d=D0,
        A=np.array([[0.85, 0.20], [0.15, 0.80]]),
        c=np.array([0.75, 0.25]),
        e=np.array([0.5, 0.5]),
        Bs=np.array(
            [
                [[0.90, 0.10], [0.10, 0.90]],
                [[0.20, 0.80], [0.80, 0.20]],
            ]
        ),
        gfe=True,
        **kwargs,
    )


def zero_goal_builder(*, past: int, steps: int, **kwargs):
    template, arrays = binary_builder(past=past, steps=steps, **kwargs)
    arrays["c"] = np.array([1.0, 0.0])
    return template, arrays


def make_agent(
    *,
    key=jax.random.PRNGKey(7),
    samples: int = 128,
    min_ess: float = 0.01,
    horizon: int = 1,
    refresh: str = "trial",
    n_sweeps: int = 100,
    select: str = "map",
    seed: int = 0,
) -> RollingAgent:
    return RollingAgent(
        binary_builder,
        horizon=horizon,
        learn_a=True,
        initial_counts=ALPHA0,
        mu3=True,
        mu3_key=key,
        mu3_samples=samples,
        mu3_min_ess_fraction=min_ess,
        mu3_refresh=refresh,
        n_sweeps=n_sweeps,
        select=select,
        seed=seed,
    )


def make_weight_collapse_agent(*, key=jax.random.PRNGKey(5)) -> RollingAgent:
    return RollingAgent(
        zero_goal_builder,
        horizon=1,
        learn_a=True,
        initial_counts=ALPHA0,
        mu3=True,
        mu3_key=key,
        mu3_samples=64,
        mu3_min_ess_fraction=0.01,
        n_sweeps=100,
    )


def direct_state(engine: Engine, learnables: Learnables, belief: np.ndarray) -> State:
    """Independent parameter plumbing: deliberately not RollingAgent._with_params."""
    state = engine.init_state()
    params = learnables.write(engine, dict(state.params))
    address = engine.arena.member_of["prior"].parameter("statistic")
    params[address.key] = params[address.key].at[address.index].set(
        jnp.asarray(_log_of(belief))
    )
    return _replace(state, params=params)


def assert_result_exact(actual: RunResult, expected: RunResult) -> None:
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual.state),
        jax.tree_util.tree_leaves(expected.state),
        strict=True,
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)
    for field in fields(RunResult):
        # Kernel-cache compilation count depends on whether the otherwise
        # identical direct or agent run reaches the shared Engine first.
        if field.name == "failure":
            assert actual.failure == expected.failure
        elif field.name not in {"state", "solver_compilations"}:
            np.testing.assert_equal(
                getattr(actual, field.name), getattr(expected, field.name)
            )


def assert_restored_result(actual: RunResult, expected: RunResult) -> None:
    """Check every persisted field; run-local provenance is intentionally absent."""
    assert actual.failure is None
    assert_result_exact(actual, replace(expected, failure=None))


def checkpoint_arrays(path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def write_checkpoint(path, arrays: dict[str, np.ndarray]) -> None:
    arrays["checkpoint_fields"] = np.asarray(json.dumps(sorted(arrays)))
    np.savez(path, **arrays)


def used_agent_snapshot(agent: RollingAgent):
    return (
        agent._planning,
        agent.last_filter_result,
        agent.belief.copy(),
        agent.counts.copy(),
        agent.tick,
        agent.episode_tick,
        json.dumps(agent.rng.bit_generator.state, sort_keys=True),
    )


def assert_used_agent_unchanged(agent: RollingAgent, snapshot) -> None:
    planning, filter_result, belief, counts, tick, episode_tick, rng = snapshot
    assert agent._planning is planning
    assert agent.last_filter_result is filter_result
    np.testing.assert_array_equal(agent.belief, belief)
    np.testing.assert_array_equal(agent.counts, counts)
    assert agent.tick == tick
    assert agent.episode_tick == episode_tick
    assert json.dumps(agent.rng.bit_generator.state, sort_keys=True) == rng


def direct_plan(
    agent: RollingAgent,
    learnables: Learnables,
    belief: np.ndarray,
    key,
) -> tuple[RunResult, jax.Array, int]:
    state = direct_state(agent.plan, learnables, belief)
    prepared, next_key = agent.plan.prepare_mu3(state, key)
    result = agent.plan.run(prepared, agent.n_sweeps, tol=1e-10)
    action_q = np.asarray(agent.plan.marginals(result.state)["u1"])
    return result, next_key, int(np.argmax(action_q))


def direct_observe(
    agent: RollingAgent,
    learnables: Learnables,
    belief: np.ndarray,
    action: int,
    observation: int,
) -> tuple[RunResult, np.ndarray]:
    state = direct_state(agent.filter, learnables, belief)
    state = agent.filter.bind_data(state, {"act1": action, "obs1": observation})
    result = agent.filter.run(state, 60, tol=1e-12)
    result.require_converged()
    next_belief = np.asarray(agent.filter.marginals(result.state)["zt1"])
    next_belief /= next_belief.sum()
    learnables.accumulate(agent.filter, result, only=("A",))
    return result, next_belief


def test_metadata_key_and_fixed_epoch_are_explicit():
    agent = make_agent(key=None)
    (plan,) = agent.plan.arena.mu3_groups
    assert agent.functional_mu3
    assert plan.evaluator == "cffg.mu3_is.v1"
    assert not agent.filter.arena.mu3_groups
    with pytest.raises(ValueError, match="explicit JAX key"):
        agent.act()

    analytic = RollingAgent(
        binary_builder,
        horizon=1,
        learn_a=True,
        initial_counts=ALPHA0,
        n_sweeps=100,
    )
    assert not analytic.functional_mu3
    assert 0 <= analytic.act() < 2

    declared = RollingAgent(
        functools.partial(binary_builder, mu3=True),
        horizon=1,
        learn_a=True,
        initial_counts=ALPHA0,
        mu3_key=jax.random.PRNGKey(4),
    )
    assert declared.functional_mu3
    assert not declared.filter.arena.mu3_groups

    def strict_analytic_builder(
        *, past, steps, learned_a=False, learned_c=False, learn=()
    ):
        return binary_builder(
            past=past,
            steps=steps,
            learned_a=learned_a,
            learned_c=learned_c,
            learn=learn,
        )

    strict = RollingAgent(strict_analytic_builder, horizon=1, n_sweeps=100)
    assert not strict.functional_mu3


@pytest.mark.parametrize(
    ("builder", "kwargs"),
    ((build_gridworld, {"k": 3}), (build_tmaze, {})),
)
def test_public_rollout_wrappers_preserve_serialized_evaluator(builder, kwargs):
    template, arrays = builder(
        past=0, steps=2, learned_a=True, mu3=True, learn=("A",), **kwargs
    )
    restored = from_json(to_json(template))
    (plan,) = compile_template(restored, arrays=arrays).mu3_groups
    assert plan.evaluator == "cffg.mu3_is.v1"
    assert len(plan.rows) == 2


@pytest.mark.slow
def test_trial_loop_matches_direct_engines_and_checkpoint_replays(tmp_path):
    key0 = jax.random.PRNGKey(7)
    agent = make_agent(key=key0)
    learnables = Learnables(agent.plan, priors={"A": ALPHA0})
    belief = D0.copy()

    before = learnables.params["A"]["alpha"].sum()
    direct_filter, belief = direct_observe(
        agent, learnables, belief, action=0, observation=1
    )
    agent.observe(0, 1)
    assert_result_exact(agent.last_filter_result, direct_filter)
    np.testing.assert_array_equal(agent.belief, belief)
    np.testing.assert_array_equal(agent.counts, learnables.params["A"]["alpha"])
    assert learnables.params["A"]["alpha"].sum() - before == pytest.approx(1.0)

    direct1, key1, expected_action = direct_plan(agent, learnables, belief, key0)
    action = agent.act()
    assert action == expected_action
    assert_result_exact(agent.last_plan_result, direct1)
    np.testing.assert_array_equal(agent.planning_key_cursor, key1)
    assert agent.planning_epoch == 1
    samples1 = np.asarray(
        agent.last_plan_result.state.params[
            agent.plan.arena.mu3_groups[0].name + ":samples"
        ]
    ).copy()

    # Replanning the unchanged trial reuses the converged epoch and no key.
    assert agent.act() == action
    assert agent.planning_epoch == 1
    np.testing.assert_array_equal(agent.planning_key_cursor, key1)

    path = tmp_path / "functional-agent.npz"
    agent.checkpoint(path)
    restored = make_agent(key=jax.random.PRNGKey(999))
    restored.observe(1, 0)
    assert restored.last_filter_result is not None
    restored.restore(path)
    assert restored.last_filter_result is None
    assert_result_exact(restored.last_plan_result, agent.last_plan_result)
    np.testing.assert_array_equal(restored.planning_key_cursor, key1)
    assert restored.act() == action
    assert restored.planning_epoch == 1

    before = learnables.params["A"]["alpha"].sum()
    direct_filter, belief = direct_observe(
        agent, learnables, belief, action, observation=1
    )
    agent.observe(action, 1)
    restored.observe(action, 1)
    assert_result_exact(agent.last_filter_result, direct_filter)
    assert_result_exact(restored.last_filter_result, direct_filter)
    np.testing.assert_array_equal(agent.belief, belief)
    np.testing.assert_array_equal(restored.belief, belief)
    np.testing.assert_array_equal(agent.counts, learnables.params["A"]["alpha"])
    np.testing.assert_array_equal(restored.counts, agent.counts)
    assert learnables.params["A"]["alpha"].sum() - before == pytest.approx(1.0)

    counts = agent.counts.copy()
    agent.reset_episode()
    restored.reset_episode()
    np.testing.assert_array_equal(agent.counts, counts)
    assert agent.episode_tick == restored.episode_tick == 0

    direct2, key2, expected_action2 = direct_plan(agent, learnables, belief, key1)
    action2 = agent.act()
    assert action2 == restored.act() == expected_action2
    assert_result_exact(agent.last_plan_result, direct2)
    assert_result_exact(restored.last_plan_result, direct2)
    np.testing.assert_array_equal(agent.planning_key_cursor, key2)
    np.testing.assert_array_equal(restored.planning_key_cursor, key2)
    plan = agent.plan.arena.mu3_groups[0]
    base2 = np.asarray(agent.last_plan_result.state.params[plan.name + ":base_alpha"])
    np.testing.assert_array_equal(base2, agent.counts)
    assert not np.array_equal(base2, ALPHA0)
    assert not np.array_equal(
        samples1, agent.last_plan_result.state.params[plan.name + ":samples"]
    )


@pytest.mark.slow
def test_functional_checkpoint_has_one_exact_outer_and_runtime_manifest(tmp_path):
    source = make_agent()
    source.act()
    path = tmp_path / "functional-envelope.npz"
    source.checkpoint(path)

    arrays = checkpoint_arrays(path)
    assert int(arrays["checkpoint_version"]) == 1
    assert str(arrays["agent:mode"]) == "functional_mu3"
    assert int(arrays["mu3_samples"]) == source.plan.mu3_samples
    assert float(arrays["mu3_min_ess_fraction"]) == source.plan.mu3_min_ess_fraction
    assert str(arrays["mu3_refresh"]) == source._planning.refresh_policy
    assert json.loads(str(arrays["checkpoint_fields"])) == sorted(arrays)

    state_fields = {
        f"mu3:state:{field.name}" for field in fields(State) if field.name != "params"
    }
    parameter_fields = {
        f"mu3:state:param:{name}" for name in source.last_plan_result.state.params
    }
    assert {name for name in arrays if name.startswith("mu3:")} == {
        "mu3:metadata",
        "mu3:key",
        "mu3:state:metadata",
        *state_fields,
        *parameter_fields,
    }


@pytest.mark.slow
def test_agent_version_1_functional_checkpoint_migrates_runtime(tmp_path):
    source = make_agent()
    action = source.act()
    current = tmp_path / "current-functional.npz"
    source.checkpoint(current)
    arrays = checkpoint_arrays(current)
    arrays["agent:version"] = np.asarray(1)
    for field in (
        "checkpoint_version",
        "checkpoint_fields",
        "plan_structure_hash",
        "filter_structure_hash",
        "plan_parameter_hash",
        "filter_parameter_hash",
        "horizon",
        "damp",
        "n_sweeps",
        "select",
        "jax_enable_x64",
        "mu3_samples",
        "mu3_min_ess_fraction",
        "mu3_refresh",
    ):
        del arrays[field]
    legacy = tmp_path / "agent-version-1-functional.npz"
    np.savez(legacy, **arrays)

    restored = make_agent(key=jax.random.PRNGKey(999))
    restored.observe(1, 0)
    restored.restore(legacy)

    assert_result_exact(restored.last_plan_result, source.last_plan_result)
    np.testing.assert_array_equal(
        restored.planning_key_cursor, source.planning_key_cursor
    )
    np.testing.assert_array_equal(restored.belief, source.belief)
    np.testing.assert_array_equal(restored.counts, source.counts)
    assert restored.tick == source.tick
    assert restored.episode_tick == source.episode_tick
    assert restored.planning_epoch == source.planning_epoch
    assert restored.act() == action


@pytest.mark.parametrize(
    "mismatch",
    (pytest.param("samples", marks=pytest.mark.slow), pytest.param("min_ess", marks=pytest.mark.slow),
     pytest.param("refresh", marks=pytest.mark.slow), "mode"),
)
def test_functional_checkpoint_configuration_mismatch_is_atomic(
    tmp_path, mismatch
):
    source = make_agent()
    source.act()
    path = tmp_path / f"functional-{mismatch}.npz"
    source.checkpoint(path)

    if mismatch == "samples":
        target = make_agent(samples=64)
        message = "runtime configuration mismatch"
    elif mismatch == "min_ess":
        target = make_agent(min_ess=0.02)
        message = "runtime configuration mismatch"
    elif mismatch == "refresh":
        target = make_agent(refresh="posterior")
        message = "runtime configuration mismatch"
    else:
        target = RollingAgent(
            binary_builder,
            horizon=1,
            learn_a=True,
            initial_counts=ALPHA0,
            n_sweeps=100,
        )
        message = "cannot restore"
    action = target.act()
    target.observe(action, 0)
    snapshot = used_agent_snapshot(target)

    with pytest.raises(ValueError, match=message):
        target.restore(path)
    assert_used_agent_unchanged(target, snapshot)


@pytest.mark.parametrize(
    "corruption",
    (
        pytest.param("extra_array", marks=pytest.mark.slow),
        pytest.param("missing_key", marks=pytest.mark.slow),
        pytest.param("unclaimed_key", marks=pytest.mark.slow),
        pytest.param("null_cache", marks=pytest.mark.slow),
        "nonfinite_state",
        pytest.param("nonfinite_trace", marks=pytest.mark.slow),
        pytest.param("nonfinite_outer", marks=pytest.mark.slow),
        pytest.param("invalid_outer", marks=pytest.mark.slow),
        pytest.param("invalid_weights", marks=pytest.mark.slow),
        pytest.param("invalid_counter", marks=pytest.mark.slow),
        pytest.param("invalid_cache", marks=pytest.mark.slow),
        pytest.param("extra_metadata", marks=pytest.mark.slow),
        pytest.param("missing_state", marks=pytest.mark.slow),
    ),
)
def test_functional_checkpoint_runtime_schema_corruption_is_atomic(
    tmp_path, corruption
):
    source = make_agent()
    source.act()
    good = tmp_path / "functional-good.npz"
    source.checkpoint(good)
    arrays = checkpoint_arrays(good)

    if corruption == "extra_array":
        arrays["mu3:future"] = np.asarray(1)
    elif corruption == "missing_key":
        del arrays["mu3:key"]
    elif corruption == "unclaimed_key":
        metadata = json.loads(str(arrays["mu3:metadata"]))
        metadata["has_key"] = False
        del metadata["key_impl"]
        del metadata["typed_key"]
        arrays["mu3:metadata"] = np.asarray(json.dumps(metadata))
        del arrays["mu3:key"]
    elif corruption == "null_cache":
        metadata = json.loads(str(arrays["mu3:metadata"]))
        metadata["cache_config"] = None
        arrays["mu3:metadata"] = np.asarray(json.dumps(metadata))
    elif corruption == "nonfinite_state":
        arrays["mu3:state:messages"] = np.full_like(
            arrays["mu3:state:messages"], np.nan
        )
    elif corruption == "nonfinite_trace":
        metadata = json.loads(str(arrays["mu3:metadata"]))
        metadata["result"]["message_deltas"][0] = float("nan")
        arrays["mu3:metadata"] = np.asarray(json.dumps(metadata))
    elif corruption == "nonfinite_outer":
        metadata = json.loads(str(arrays["mu3:metadata"]))
        metadata["result"]["outer_merit"] = float("nan")
        arrays["mu3:metadata"] = np.asarray(json.dumps(metadata))
    elif corruption == "invalid_outer":
        metadata = json.loads(str(arrays["mu3:metadata"]))
        metadata["result"]["trust_radius"] = -1.0
        arrays["mu3:metadata"] = np.asarray(json.dumps(metadata))
    elif corruption == "invalid_weights":
        weight_field = next(name for name in arrays if name.endswith(":weights"))
        arrays[weight_field] = np.full_like(arrays[weight_field], -1.0)
    elif corruption == "invalid_counter":
        metadata = json.loads(str(arrays["mu3:metadata"]))
        metadata["result"]["iterations"] = -1
        arrays["mu3:metadata"] = np.asarray(json.dumps(metadata))
    elif corruption == "invalid_cache":
        metadata = json.loads(str(arrays["mu3:metadata"]))
        metadata["cache_config"]["tol"] = float("nan")
        arrays["mu3:metadata"] = np.asarray(json.dumps(metadata))
    elif corruption == "extra_metadata":
        metadata = json.loads(str(arrays["mu3:metadata"]))
        metadata["future"] = True
        arrays["mu3:metadata"] = np.asarray(json.dumps(metadata))
    else:
        state_field = next(
            name for name in arrays if name.startswith("mu3:state:param:")
        )
        del arrays[state_field]
    path = tmp_path / f"functional-{corruption}.npz"
    write_checkpoint(path, arrays)

    target = make_agent()
    action = target.act()
    target.observe(action, 0)
    snapshot = used_agent_snapshot(target)
    with pytest.raises((KeyError, ValueError), match="functional checkpoint"):
        target.restore(path)
    assert_used_agent_unchanged(target, snapshot)


def test_functional_checkpoint_cannot_promote_exhaustion_to_convergence(tmp_path):
    source = make_agent(n_sweeps=0)
    with pytest.raises(InferenceError) as exhausted:
        source.act()
    assert exhausted.value.result.status is RunStatus.BUDGET_EXHAUSTED
    good = tmp_path / "functional-exhausted.npz"
    source.checkpoint(good)
    arrays = checkpoint_arrays(good)
    metadata = json.loads(str(arrays["mu3:metadata"]))
    metadata["result"]["status"] = RunStatus.CONVERGED.value
    arrays["mu3:metadata"] = np.asarray(json.dumps(metadata))
    corrupt = tmp_path / "functional-promoted.npz"
    write_checkpoint(corrupt, arrays)

    target = make_agent(n_sweeps=0)
    target.observe(0, 1)
    snapshot = used_agent_snapshot(target)
    with pytest.raises(ValueError, match="solver diagnostics"):
        target.restore(corrupt)
    assert_used_agent_unchanged(target, snapshot)
    with pytest.raises(InferenceError):
        target.act()


def test_nonconverged_checkpoint_rejects_invalid_particles_before_mutation(tmp_path):
    source = make_agent(n_sweeps=0)
    with pytest.raises(InferenceError) as exhausted:
        source.act()
    assert exhausted.value.result.status is RunStatus.BUDGET_EXHAUSTED
    good = tmp_path / "functional-exhausted.npz"
    source.checkpoint(good)
    arrays = checkpoint_arrays(good)
    particle_field = next(name for name in arrays if name.endswith(":samples"))
    arrays[particle_field] = np.full_like(arrays[particle_field], -1.0)
    corrupt = tmp_path / "functional-exhausted-negative-particles.npz"
    write_checkpoint(corrupt, arrays)

    target = make_agent(n_sweeps=0)
    target.observe(0, 1)
    snapshot = used_agent_snapshot(target)
    with pytest.raises(ValueError, match="invalid particles or weights"):
        target.restore(corrupt)
    assert_used_agent_unchanged(target, snapshot)


def test_sampling_failure_returns_no_action_and_preserves_diagnostics():
    agent = make_agent(samples=32, min_ess=1.0)
    with pytest.raises(InferenceError) as exc:
        agent.act()
    assert exc.value.result.status is RunStatus.BUDGET_EXHAUSTED
    assert exc.value.result.mu3_statuses == (RunStatus.BUDGET_EXHAUSTED,)
    assert agent.planning_diagnostics

    calls = 0

    def environment(_action):
        nonlocal calls
        calls += 1
        return 0

    with pytest.raises(InferenceError):
        agent.step(environment)
    assert calls == 0

    inference_limited = make_agent(n_sweeps=0)
    with pytest.raises(InferenceError) as ordinary:
        inference_limited.act()
    assert ordinary.value.result.status is RunStatus.BUDGET_EXHAUSTED
    assert ordinary.value.result.mu3_statuses == (RunStatus.CONVERGED,)


def test_weight_collapse_diverges_and_never_executes_an_action():
    agent = make_weight_collapse_agent()
    calls = 0

    def environment(_action):
        nonlocal calls
        calls += 1
        return 0

    with pytest.raises(InferenceError) as exc:
        agent.step(environment)
    assert calls == 0
    assert exc.value.result.status is RunStatus.DIVERGED
    assert exc.value.result.mu3_statuses == (RunStatus.DIVERGED,)
    assert next(iter(agent.planning_diagnostics.values()))["status"] == "diverged"


def test_weight_collapse_checkpoint_replays_and_advances_exactly(tmp_path):
    source = make_weight_collapse_agent()
    with pytest.raises(InferenceError) as collapsed:
        source.act()
    source_result = collapsed.value.result
    assert source_result.status is RunStatus.DIVERGED
    assert source_result.mu3_statuses == (RunStatus.DIVERGED,)
    assert source_result.mu3_ess == (0.0,)
    source_key = np.asarray(source.planning_key_cursor).copy()
    source_epoch = source.planning_epoch
    path = tmp_path / "functional-weight-collapse.npz"
    source.checkpoint(path)

    restored = make_weight_collapse_agent(key=jax.random.PRNGKey(999))
    restored.restore(path)
    assert_restored_result(restored.last_plan_result, source_result)
    np.testing.assert_array_equal(restored.planning_key_cursor, source_key)
    assert restored.planning_epoch == source_epoch

    with pytest.raises(InferenceError) as replayed:
        restored.act()
    assert_restored_result(replayed.value.result, source_result)
    np.testing.assert_array_equal(restored.planning_key_cursor, source_key)
    assert restored.planning_epoch == source_epoch

    source.refresh_mu3()
    restored.refresh_mu3()
    with pytest.raises(InferenceError) as source_next:
        source.act()
    with pytest.raises(InferenceError) as restored_next:
        restored.act()
    assert_result_exact(restored_next.value.result, source_next.value.result)
    np.testing.assert_array_equal(
        restored.planning_key_cursor, source.planning_key_cursor
    )
    assert source.planning_epoch == restored.planning_epoch == source_epoch + 1


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("status", "invalid status"),
        ("ess", "inconsistent ESS"),
        ("moments", "inconsistent moments"),
        ("weights_ess", "inconsistent prepared state"),
    ),
)
def test_weight_collapse_checkpoint_rejects_invalid_failure_state(
    tmp_path, corruption, message
):
    source = make_weight_collapse_agent()
    with pytest.raises(InferenceError):
        source.act()
    good = tmp_path / "functional-weight-collapse.npz"
    source.checkpoint(good)
    arrays = checkpoint_arrays(good)

    if corruption == "status":
        field = next(name for name in arrays if name.endswith(":status"))
        arrays[field] = np.asarray(3, dtype=arrays[field].dtype)
    elif corruption == "ess":
        field = next(name for name in arrays if name.endswith(":ess"))
        arrays[field] = np.asarray(-1.0, dtype=arrays[field].dtype)
        metadata = json.loads(str(arrays["mu3:metadata"]))
        metadata["result"]["mu3_ess"][0] = -1.0
        arrays["mu3:metadata"] = np.asarray(json.dumps(metadata))
    elif corruption == "moments":
        field = next(name for name in arrays if name.endswith(":Abar"))
        arrays[field] = np.flip(arrays[field], axis=-2)
    else:
        weight_field = next(name for name in arrays if name.endswith(":weights"))
        weights = np.zeros_like(arrays[weight_field])
        weights[0] = np.asarray(1.0, dtype=weights.dtype)
        arrays[weight_field] = weights
        ess_field = next(name for name in arrays if name.endswith(":ess"))
        arrays[ess_field] = np.asarray(1.0, dtype=arrays[ess_field].dtype)
        metadata = json.loads(str(arrays["mu3:metadata"]))
        metadata["result"]["mu3_ess"][0] = 1.0
        arrays["mu3:metadata"] = np.asarray(json.dumps(metadata))
    corrupt = tmp_path / f"functional-weight-collapse-{corruption}.npz"
    write_checkpoint(corrupt, arrays)

    target = make_weight_collapse_agent()
    target.observe(0, 1)
    snapshot = used_agent_snapshot(target)
    with pytest.raises(ValueError, match=message):
        target.restore(corrupt)
    assert_used_agent_unchanged(target, snapshot)


@pytest.mark.slow
def test_posterior_refresh_policy_reuses_particles_for_context_only_changes():
    agent = make_agent(refresh="posterior")
    agent.act()
    first_result = agent.last_plan_result
    first_key = np.asarray(agent.planning_key_cursor).copy()
    plan = agent.plan.arena.mu3_groups[0]
    first_samples = np.asarray(
        first_result.state.params[plan.name + ":samples"]
    ).copy()

    agent.belief = np.array([0.35, 0.65])
    agent.act()
    assert agent.last_plan_result is not first_result
    assert agent.planning_epoch == 1
    np.testing.assert_array_equal(agent.planning_key_cursor, first_key)
    np.testing.assert_array_equal(
        agent.last_plan_result.state.params[plan.name + ":samples"], first_samples
    )


@pytest.mark.slow
def test_cached_epoch_is_invalidated_by_runtime_configuration_changes():
    agent = make_agent()
    agent.act()
    first_result = agent.last_plan_result
    first_key = np.asarray(agent.planning_key_cursor).copy()

    agent.n_sweeps += 1
    agent.act()
    assert agent.last_plan_result is not first_result
    assert agent.planning_epoch == 1
    np.testing.assert_array_equal(agent.planning_key_cursor, first_key)

    agent.plan.mu3_min_ess_fraction = 1.0
    with pytest.raises(InferenceError) as exc:
        agent.act()
    assert exc.value.result.status is RunStatus.BUDGET_EXHAUSTED
    assert exc.value.result.mu3_statuses == (RunStatus.BUDGET_EXHAUSTED,)
    assert agent.planning_epoch == 1
    np.testing.assert_array_equal(agent.planning_key_cursor, first_key)


@pytest.mark.slow
def test_typed_jax_key_checkpoint_preserves_representation(tmp_path):
    agent = make_agent(key=jax.random.key(12))
    agent.act()
    path = tmp_path / "typed-key.npz"
    agent.checkpoint(path)

    restored = make_agent(key=jax.random.key(99))
    restored.restore(path)
    assert str(restored.planning_key_cursor.dtype).startswith("key<")
    np.testing.assert_array_equal(
        jax.random.key_data(restored.planning_key_cursor),
        jax.random.key_data(agent.planning_key_cursor),
    )
    assert_result_exact(restored.last_plan_result, agent.last_plan_result)


@pytest.mark.slow
def test_checkpoint_drops_result_after_particle_budget_change(tmp_path):
    source = make_agent(samples=128)
    source.act()
    saved_cursor = np.asarray(source.planning_key_cursor).copy()
    source.plan.mu3_samples = 64
    path = tmp_path / "changed-budget.npz"
    source.checkpoint(path)

    restored = make_agent(samples=64)
    restored.restore(path)
    assert restored.last_plan_result is None
    np.testing.assert_array_equal(restored.planning_key_cursor, saved_cursor)
    restored.act()
    assert restored.planning_epoch == 2
    assert (
        restored.last_plan_result.state.params[
            restored.plan.arena.mu3_groups[0].name + ":samples"
        ].shape
        == (64, 2, 2)
    )


@pytest.mark.slow
def test_action_rng_is_separate_from_functional_key_cursor():
    first = make_agent(select="sample", seed=1)
    second = make_agent(select="sample", seed=99)
    before = json.dumps(first.rng.bit_generator.state, sort_keys=True)
    first.act()
    second.act()
    after = json.dumps(first.rng.bit_generator.state, sort_keys=True)

    assert before != after
    np.testing.assert_array_equal(
        first.planning_key_cursor, second.planning_key_cursor
    )
    key_after_plan = np.asarray(first.planning_key_cursor).copy()
    first.act()
    np.testing.assert_array_equal(first.planning_key_cursor, key_after_plan)
    assert json.dumps(first.rng.bit_generator.state, sort_keys=True) != after


def test_bad_base_and_legacy_functional_restore_fail_before_mutation(tmp_path):
    for value in (0.0, np.nan, np.inf):
        agent = make_agent()
        agent.learnables.params["A"]["alpha"][0, 0] = value
        with pytest.raises(ValueError, match="must be > 0|proper Dirichlet"):
            agent.act()

    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, belief=D0, tick=np.asarray(4), counts=ALPHA0)
    fresh = make_agent()
    before_belief = fresh.belief.copy()
    before_counts = fresh.counts.copy()
    with pytest.raises(ValueError, match="legacy checkpoint"):
        fresh.restore(legacy)
    np.testing.assert_array_equal(fresh.belief, before_belief)
    np.testing.assert_array_equal(fresh.counts, before_counts)
    assert fresh.tick == 0

    source = make_agent()
    source.act()
    valid = tmp_path / "valid.npz"
    source.checkpoint(valid)
    with np.load(valid) as stored:
        corrupt_payload = {key: stored[key].copy() for key in stored.files}
    learn_key = next(key for key in corrupt_payload if key.startswith("learn:"))
    corrupt_payload[learn_key] = corrupt_payload[learn_key] + 1.0
    from cffg.learning_checkpoint import array_sha256
    from cffg.learning_schema import PERSISTENT_MANIFEST_KEY

    manifest_key = f"learn:{PERSISTENT_MANIFEST_KEY}"
    manifest = json.loads(corrupt_payload[manifest_key].item())
    for descriptor in manifest["learnables"]:
        for row in descriptor["arrays"]:
            if row["role"] == "posterior":
                row["sha256"] = array_sha256(corrupt_payload[learn_key])
    corrupt_payload[manifest_key] = np.asarray(json.dumps(manifest))
    corrupt = tmp_path / "corrupt.npz"
    np.savez(corrupt, **corrupt_payload)

    target = make_agent()
    before_belief = target.belief.copy()
    before_counts = target.counts.copy()
    with pytest.raises(ValueError, match="planning state is stale"):
        target.restore(corrupt)
    np.testing.assert_array_equal(target.belief, before_belief)
    np.testing.assert_array_equal(target.counts, before_counts)
    assert target.tick == 0


@pytest.mark.slow
def test_float32_ten_slice_agent_keeps_one_shared_evaluator_finite():
    with jax.enable_x64(False):
        agent = make_agent(samples=32, horizon=10)
        (plan,) = agent.plan.arena.mu3_groups
        assert len(plan.rows) == len(plan.members) == 10
        action = agent.act()
        assert 0 <= action < 2
        for leaf in jax.tree_util.tree_leaves(agent.last_plan_result.state):
            assert bool(jnp.all(jnp.isfinite(leaf)))
        assert agent.last_plan_result.status is RunStatus.CONVERGED
        assert agent.last_plan_result.mu3_statuses == (RunStatus.CONVERGED,)

"""Initial and later failures share one truthful public result contract."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_compiler_contracts import model
from test_engine_contracts import (
    assert_result_equal,
    assert_state_equal,
    ordinary_case,
)

from cffg.colony import Colony
from cffg.compiled import NEG
from cffg.compiler import compile_template
from cffg.engine import Engine, RunStatus, _replace


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("schedule", ("auto", "flood"))
@pytest.mark.parametrize("budget", (0, 8))
@pytest.mark.parametrize("failure", ("nan", "infinity", "empty_support"))
def test_initial_failure_preserves_state_and_healthy_peer(
    dtype, schedule, budget, failure, monkeypatch
):
    colony = Colony()
    ids = []
    for index in range(2):
        template, arrays, observations = ordinary_case(index)
        ids.append(colony.spawn(template, arrays, observations=observations))
    key = colony.instances[ids[0]].key
    assert colony.instances[ids[1]].key == key
    engine = Engine(colony.engines[key].arena, schedule=schedule)
    colony.engines[key] = engine
    assert (engine._prime is not None) == (schedule == "auto")

    def cast(value):
        return (
            value.astype(dtype) if jnp.issubdtype(value.dtype, jnp.floating) else value
        )

    states = [jax.tree_util.tree_map(cast, colony.instances[iid].state) for iid in ids]
    if failure == "empty_support":
        messages = jnp.full_like(states[0].messages, NEG)
        expected_status = RunStatus.INFEASIBLE
    else:
        value = jnp.nan if failure == "nan" else jnp.inf
        messages = states[0].messages.at[-1].set(value)
        expected_status = RunStatus.DIVERGED
    states[0] = _replace(states[0], messages=messages)
    for iid, state in zip(ids, states, strict=True):
        colony.instances[iid].state = state
        assert state.messages.dtype == np.dtype(dtype)

    expected = [
        engine.run(state, budget, tol=1e-12, harden_last=budget) for state in states
    ]
    batch_sizes = []
    original_vmap = eqx.filter_vmap

    def observe_vmap(function, *args, **kwargs):
        mapped = original_vmap(function, *args, **kwargs)
        if function.__name__ != "one":
            return mapped

        def execute(state):
            batch_sizes.append(state.messages.shape[0])
            return mapped(state)

        return execute

    monkeypatch.setattr(eqx, "filter_vmap", observe_vmap)
    actual = colony.run(budget, tol=1e-12)

    rejected = actual[ids[0]]
    assert rejected.status is expected_status
    assert rejected.iterations == 0
    assert_state_equal(rejected.state, states[0])
    for iid, reference in zip(ids, expected, strict=True):
        assert_result_equal(actual[iid], reference)
        assert colony.instances[iid].state is actual[iid].state
    if budget:
        assert batch_sizes and set(batch_sizes) == {2}
        assert actual[ids[1]].status is RunStatus.CONVERGED
    else:
        assert not batch_sizes
        assert actual[ids[1]].status is RunStatus.BUDGET_EXHAUSTED


def test_all_rejected_instances_skip_priming_and_updates(monkeypatch):
    colony = Colony()
    ids = []
    for index in range(2):
        template, arrays, observations = ordinary_case(index)
        ids.append(colony.spawn(template, arrays, observations=observations))
    engine = colony.engines[colony.instances[ids[0]].key]
    assert engine._prime is not None
    states = [colony.instances[iid].state for iid in ids]
    states[0] = _replace(states[0], messages=states[0].messages.at[-1].set(jnp.nan))
    states[1] = _replace(states[1], messages=jnp.full_like(states[1].messages, NEG))
    expected = [engine.run(state, 8, harden_last=8) for state in states]
    for iid, state in zip(ids, states, strict=True):
        colony.instances[iid].state = state

    def forbidden(*args, **kwargs):
        raise AssertionError("Rejected inputs must not enter numerical execution")

    monkeypatch.setattr(engine, "_prime", forbidden)
    monkeypatch.setattr(engine, "_iter_step", forbidden)
    actual = colony.run(8)
    for iid, reference in zip(ids, expected, strict=True):
        assert_result_equal(actual[iid], reference)


def check_later_failure(
    monkeypatch, dtype, schedule, failure, *, size=2, representation="messages"
):
    colony = Colony()
    ids = []
    for i in range(size):
        if representation == "messages":
            template, arrays, observations = ordinary_case(i % 2)
        else:
            template, arrays = model("gaussian")
            observations = None
        ids.append(colony.spawn(template, arrays, observations=observations))
    key = colony.instances[ids[0]].key
    engine = Engine(colony.engines[key].arena, schedule=schedule)
    colony.engines[key] = engine
    counter_field = "messages" if representation == "messages" else "g_eta"
    states = []
    for i, iid in enumerate(ids):
        state = jax.tree_util.tree_map(
            lambda x: x.astype(dtype) if x.dtype.kind == "f" else x,
            colony.instances[iid].state,
        )
        state = _replace(state, messages=jnp.zeros_like(state.messages))
        origin_state = state
        if engine._prime is not None:
            msgs, eta, lam, ef_eta, *_ = engine._prime(
                state.messages,
                state.g_eta,
                state.g_lam,
                state.ef_eta,
                state.g_obs,
                state.g_mask,
                state.ef_obs,
                state.ef_mask,
                state.params,
            )
            origin_state = _replace(
                state, messages=msgs, g_eta=eta, g_lam=lam, ef_eta=ef_eta
            )
        state = _replace(
            state,
            params={
                **state.params,
                "test:failure": jnp.asarray(i == 0),
                "test:origin": getattr(origin_state, counter_field)[0],
            },
        )
        colony.instances[iid].state = state
        states.append(state)

    @jax.jit
    def step(
        messages, eta, lam, obs, mask, ef_eta, ef_obs, ef_mask, params, damp, harden
    ):
        blocks = {"messages": messages, "g_eta": eta, "g_lam": lam}
        blocks[counter_field] = blocks[counter_field].at[0].add(1)
        count = blocks[counter_field][0] - params["test:origin"]
        fails = params["test:failure"] & (count >= 2.5)
        if failure == "support":
            blocks["messages"] = jnp.where(
                fails, jnp.full_like(messages, NEG), blocks["messages"]
            )
        if failure == "state":
            values = blocks[representation]
            blocks[representation] = values.at[-1].set(
                jnp.where(fails, jnp.nan, values[-1])
            )
        code = jnp.asarray(0, jnp.int32)
        if failure in ("inner", "inner_budget"):
            code = jnp.where(fails, 2 if failure == "inner" else 1, 0)
        return (
            blocks["messages"],
            blocks["g_eta"],
            blocks["g_lam"],
            ef_eta,
            jnp.asarray(0.125, dtype),
            code,
        )

    def energy(state):
        count = float(getattr(state, counter_field)[0] - state.params["test:origin"])
        if failure == "energy" and state.params["test:failure"] and count >= 2.5:
            return float("nan")
        return count

    monkeypatch.setattr(engine, "_iter_step", step)
    monkeypatch.setattr(engine, "bfe", energy)
    expected = [engine.run(state, 5, harden_last=5) for state in states]
    batch_sizes = []
    original_vmap = eqx.filter_vmap

    def observe_vmap(function, *args, **kwargs):
        mapped = original_vmap(function, *args, **kwargs)

        def execute(state):
            batch_sizes.append(state.messages.shape[0])
            return mapped(state)

        return execute

    monkeypatch.setattr(eqx, "filter_vmap", observe_vmap)
    actual = colony.run(5)
    assert batch_sizes and set(batch_sizes) == {size}
    assert actual[ids[0]].iterations == 3
    assert len(actual[ids[0]].free_energies) == (3 if failure == "support" else 2)
    if failure == "inner_budget":
        assert actual[ids[0]].status is RunStatus.BUDGET_EXHAUSTED
        assert actual[ids[0]].residual == 0.125
    else:
        assert np.isinf(actual[ids[0]].residual)
    if failure == "state":
        assert not np.isfinite(actual[ids[0]].message_deltas[-1])
        assert not np.isfinite(expected[0].message_deltas[-1])
    for iid, reference in zip(ids, expected, strict=True):
        assert_result_equal(actual[iid], reference)
        assert colony.instances[iid].state is actual[iid].state
    for iid in ids[1:]:
        assert actual[iid].status is RunStatus.BUDGET_EXHAUSTED
        assert actual[iid].iterations == 5
        assert np.isfinite(actual[iid].residual)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("schedule", ("auto", "flood"))
@pytest.mark.parametrize(
    "failure", ("inner", "inner_budget", "energy", "support", "state")
)
def test_later_failure_reports_current_residual_and_preserves_healthy_peer(
    monkeypatch, dtype, schedule, failure
):
    check_later_failure(monkeypatch, dtype, schedule, failure)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("representation", ("g_eta", "g_lam"))
def test_nonfinite_changes_are_reported_for_each_continuous_representation(
    monkeypatch, dtype, representation
):
    check_later_failure(
        monkeypatch, dtype, "flood", "state", representation=representation
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("failure", ("inner", "state"))
def test_late_failure_is_isolated_in_a_larger_batch(monkeypatch, dtype, failure):
    check_later_failure(monkeypatch, dtype, "auto", failure, size=20)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("failure", ("objective", "undefined_ledger"))
def test_terminal_objective_and_undefined_ledger_failures_report_infinite_residual(
    monkeypatch, dtype, failure
):
    template, arrays = model("hmm" if failure == "objective" else "gaussian")
    colony = Colony()
    ids = [colony.spawn(template, arrays) for _ in range(2)]
    key = colony.instances[ids[0]].key
    engine = Engine(
        compile_template(
            template,
            arrays=arrays,
            semiring="max" if failure == "objective" else "sum",
        ),
        schedule="flood",
    )
    colony.engines[key] = engine
    for iid in ids:
        colony.instances[iid].state = jax.tree_util.tree_map(
            lambda x: x.astype(dtype) if x.dtype.kind == "f" else x,
            colony.instances[iid].state,
        )

    @jax.jit
    def step(
        messages, eta, lam, obs, mask, ef_eta, ef_obs, ef_mask, params, damp, harden
    ):
        return (
            messages,
            eta,
            lam,
            ef_eta,
            jnp.asarray(0.125, dtype),
            jnp.asarray(1 if failure == "undefined_ledger" else 0, jnp.int32),
        )

    monkeypatch.setattr(engine, "_iter_step", step)
    if failure == "objective":
        monkeypatch.setattr(engine, "objective", lambda state: float("nan"))
    expected = [
        engine.run(colony.instances[iid].state, 3, tol=1e-5, harden_last=3)
        for iid in ids
    ]
    actual = colony.run(3, tol=1e-5)
    for iid, reference in zip(ids, expected, strict=True):
        assert_result_equal(actual[iid], reference)
        assert actual[iid].status is RunStatus.DIVERGED
        assert np.isinf(actual[iid].residual)

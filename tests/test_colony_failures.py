"""Rejected initial states and healthy peers share one public result contract."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_engine_contracts import (
    assert_result_equal,
    assert_state_equal,
    ordinary_case,
)

from cffg.colony import Colony
from cffg.compiled import NEG
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

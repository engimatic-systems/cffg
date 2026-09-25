"""Runtime seams at compiler-track closing commit 76251b44fc467a976aa056556f8fb371fe5de7f4.

These are observable refactor contracts. Existing rule-family and schedule
oracles remain the mathematical authorities.
"""

from dataclasses import fields, replace
from enum import Enum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_schedule import vmp_unit_template

from cffg.colony import Colony
from cffg.compiler import compile_template
from cffg.engine import Engine, RunResult, RunStatus

SCHEDULES = {
    3: "6448208c3a9d8602eba9b8636ecb02d2eaf455c34cdac795d52a4db05af4624d",
    4: "e8142b96a3670da8c496a60b52f19d28ea73f756e9b7b81a00cdcf0ee764bb0a",
}
CASES = (
    (
        [0.6, 0.3, 0.1],
        [[0.7, 0.4, 0.1], [0.3, 0.6, 0.9]],
        0,
        (1.3862943611198901, 0.20556502096082985, 0.0),
        0.5978370007556206,
        (0.7636363636363637, 0.21818181818181817, 0.018181818181818205),
    ),
    (
        [0.15, 0.25, 0.6],
        [[0.2, 0.6, 0.8], [0.8, 0.4, 0.2]],
        1,
        (0.8472978603872034, 0.3655426769215776, 0.0),
        1.07880966137193,
        (0.35294117647058826, 0.2941176470588235, 0.35294117647058826),
    ),
    (
        [0.1, 0.2, 0.3, 0.4],
        [[0.85, 0.65, 0.35, 0.15], [0.15, 0.35, 0.65, 0.85]],
        0,
        (1.2039728043259361, 0.40736669360614053, 0.0),
        0.9675840262617056,
        (
            0.22368421052631587,
            0.3421052631578947,
            0.2763157894736841,
            0.15789473684210523,
        ),
    ),
)


def ordinary_case(index):
    prior, likelihood, observation, *_ = CASES[index]
    template = vmp_unit_template()
    size = len(prior)
    template.name = f"engine-contract-{size}"
    template.variables["z"] = replace(template.variables["z"], cardinality=size)
    return (
        template,
        {"d": np.array(prior), "A": np.array(likelihood)},
        {"px": observation},
    )


def spawn_cases():
    colony = Colony()
    ids = []
    for index in range(len(CASES)):
        template, arrays, observations = ordinary_case(index)
        ids.append(colony.spawn(template, arrays, observations=observations))
    return colony, ids


def assert_state_equal(actual, expected):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(
        expected
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        assert left.shape == right.shape
        assert left.dtype == right.dtype
        np.testing.assert_allclose(left, right, atol=1e-13, rtol=1e-13, equal_nan=True)


def assert_result_equal(actual, expected):
    for field in fields(RunResult):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if field.name == "state":
            assert_state_equal(left, right)
        elif isinstance(left, Enum):
            assert left is right, field.name
        elif field.name == "failure":
            assert left == right
        else:
            np.testing.assert_allclose(
                left, right, atol=1e-13, rtol=1e-13, equal_nan=True, err_msg=field.name
            )


@pytest.mark.parametrize("budget", [1, 20])
def test_iterative_colony_results_match_distinct_standalone_instances(
    budget, monkeypatch
):
    colony, ids = spawn_cases()
    assert len(colony.engines) == 2
    keys = [colony.instances[iid].key for iid in ids]
    assert keys[0] == keys[1] != keys[2]
    engines = dict(colony.engines)
    for engine in engines.values():
        assert engine.schedule.kind == "mixed"
        assert engine._propagate is None
        assert not engine.arena.gfe_groups
        assert not engine.arena.gfe_tm_groups
        assert not engine.arena.hybrid_gfe_groups
        assert not engine.arena.mu3_groups
        assert any(group.mode == "vmp" for group in engine.arena.groups)

    vectorized_batches = []
    original = eqx.filter_vmap

    def observe(function, *args, **kwargs):
        mapped = original(function, *args, **kwargs)
        if function.__name__ != "one":
            return mapped

        def call(state):
            vectorized_batches.append(state.messages.shape[0])
            return mapped(state)

        return call

    monkeypatch.setattr(eqx, "filter_vmap", observe)
    results = colony.run(budget, tol=1e-12)
    assert vectorized_batches and set(vectorized_batches) == {2}
    assert all(colony.engines[key] is engine for key, engine in engines.items())
    for index, iid in enumerate(ids):
        template, arrays, observations = ordinary_case(index)
        engine = Engine(compile_template(template, arrays=arrays))
        state = engine.bind_data(engine.init_state(), observations)
        solo = engine.run(state, budget, tol=1e-12, harden_last=budget)
        actual = results[iid]
        assert_result_equal(actual, solo)
        assert colony.instances[iid].state is actual.state
        assert colony.last_results[iid] is actual
        size = len(CASES[index][0])
        assert engine.schedule.fingerprint == SCHEDULES[size]
        assert actual.state.messages.shape == (2 * (size + 2),)
        assert actual.state.messages.dtype == np.dtype("float64")
        assert sum(
            value.nbytes for value in jax.tree_util.tree_leaves(actual.state)
        ) == (152 if size == 3 else 192)
        assert actual.status is (
            RunStatus.BUDGET_EXHAUSTED if budget == 1 else RunStatus.CONVERGED
        )
        assert actual.iterations == (1 if budget == 1 else 3)
        assert actual.inner_status is RunStatus.CONVERGED
        assert actual.inner_residual == 0.0
        assert actual.termination_reason.value == "not_run"
        assert actual.residual == (float("inf") if budget == 1 else 0.0)
        np.testing.assert_allclose(
            actual.message_deltas, CASES[index][3][: actual.iterations], atol=1e-13
        )
        np.testing.assert_allclose(
            actual.free_energies, [CASES[index][4]] * actual.iterations, atol=1e-13
        )
        beliefs = engine.marginals(actual.state)
        np.testing.assert_allclose(beliefs["z"], CASES[index][5], atol=1e-13)
        np.testing.assert_array_equal(beliefs["x"], np.eye(2)[observations["px"]])


def test_colony_binding_removal_and_spawn_preserve_other_instances_and_cache():
    colony, (first, second, different) = spawn_cases()
    colony.run(20, tol=1e-12)
    engines = dict(colony.engines)
    untouched = {iid: colony.instances[iid].state for iid in (second, different)}
    results = {iid: colony.last_results[iid] for iid in untouched}
    colony.bind(first, {"px": 1})
    colony.kill(first)
    assert first not in colony.instances and first not in colony.last_results
    template, arrays, observations = ordinary_case(0)
    replacement = colony.spawn(template, arrays, observations=observations)
    assert replacement > different
    assert colony.instances[replacement].key == colony.instances[second].key
    assert colony.engines.keys() == engines.keys()
    assert all(colony.engines[key] is engine for key, engine in engines.items())
    for iid, state in untouched.items():
        assert colony.instances[iid].state is state
        assert colony.last_results[iid] is results[iid]


@pytest.mark.parametrize(
    "code,status", [(1, RunStatus.BUDGET_EXHAUSTED), (2, RunStatus.DIVERGED)]
)
def test_jit_inner_status_codes_define_host_failure_results(code, status, monkeypatch):
    template, arrays, observations = ordinary_case(0)
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.bind_data(engine.init_state(), observations)

    @jax.jit
    def inner_result(
        messages,
        eta,
        precision,
        observations,
        mask,
        ef_eta,
        ef_observations,
        ef_mask,
        params,
        damp,
        harden,
    ):
        return (
            messages,
            eta,
            precision,
            ef_eta,
            jnp.asarray(0.125, messages.dtype),
            jnp.asarray(code, jnp.int32),
        )

    monkeypatch.setattr(engine, "_iter_step", inner_result)
    result = engine.run(state, 5, tol=1e-12)
    assert result.status is result.inner_status is status
    assert result.inner_residual == 0.125
    assert result.residual == (0.125 if code == 1 else float("inf"))
    assert result.iterations == 1
    assert result.message_deltas == (0.0,)
    assert result.free_energies == ()
    assert result.termination_reason.value == "not_run"

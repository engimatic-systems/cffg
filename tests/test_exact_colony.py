"""Exact schedules remain exact across irrelevant phases and Colony members."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_engine_contracts import assert_result_equal, assert_state_equal
from test_schedule import unit_categorical

from cffg import Engine, RunStatus, compile_template
from cffg.colony import Colony
from cffg.compiled import NEG
from cffg.core import (
    BlobRef,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.engine import _replace


def gaussian_goal_chain(length=2, mean=0.0):
    template = Template(name="gaussian_goal_chain")
    arrays = {
        "mean": np.array([mean]),
        "prior_cov": np.eye(1),
        "matrix": np.eye(1),
        "offset": np.zeros(1),
        "process_cov": 0.1 * np.eye(1),
        "goal_cov": np.eye(1),
    }
    template.blobs = {
        name: BlobRef(name, f"blobs/{name}.npy") for name in arrays
    }
    template.variables["s0"] = VariableSpec("s0", dim=1)
    template.factors["prior"] = FactorSpec(
        "prior", "gaussian", edges={"out": "s0"},
        param_blobs={"mean": "mean", "cov": "prior_cov"},
    )
    previous = "s0"
    observations = {}
    for index in range(1, length + 1):
        state, goal = f"s{index}", f"g{index}"
        template.variables[state] = VariableSpec(state, dim=1)
        template.variables[goal] = VariableSpec(goal, dim=1)
        transition = f"transition{index}"
        template.factors[transition] = FactorSpec(
            transition, "gaussian_linear", edges={"in": previous, "out": state},
            param_blobs={"matrix": "matrix", "offset": "offset", "cov": "process_cov"},
        )
        if index < length:
            previous, observed = f"s{index}c", f"s{index}o"
            for name in (previous, observed):
                template.variables[name] = VariableSpec(name, dim=1)
            equality = f"equality{index}"
            template.factors[equality] = FactorSpec(
                equality, "equality", edges={"a": state, "b": previous, "c": observed}
            )
        else:
            observed = state
        emission = f"goal{index}"
        template.factors[emission] = FactorSpec(
            emission, "gaussian_linear", edges={"in": observed, "out": goal},
            param_blobs={"matrix": "matrix", "offset": "offset", "cov": "goal_cov"},
        )
        template.ports[goal] = PortSpec(goal, goal, PortRole.DATA)
        observations[goal] = np.array([0.5 + mean])
    return template, arrays, observations


@pytest.mark.parametrize("enabled,length", [(False, 2), (True, 2), (True, 20)])
@pytest.mark.parametrize("tol", [None, 1e-6])
def test_irrelevant_hardening_preserves_exact_gaussian_result(enabled, length, tol):
    with jax.enable_x64(enabled):
        template, arrays, observations = gaussian_goal_chain(length)
        engine = Engine(compile_template(template, arrays=arrays))
        state = engine.bind_data(engine.init_state(), observations)
        assert not len(engine.arena.delta_pairs)
        reference = engine.run(state, 5, tol=tol)
        assert reference.status is RunStatus.CONVERGED
        assert reference.iterations == 1
        for harden_last in (0, 1, 5):
            actual = engine.run(state, 5, tol=tol, harden_last=harden_last)
            assert_result_equal(actual, reference)
        for invalid in (-1, 6):
            with pytest.raises(ValueError, match="harden_last"):
                engine.run(state, 5, harden_last=invalid)


@pytest.mark.parametrize("size", [1, 3])
@pytest.mark.parametrize("budget,damp", [(0, 0.0), (5, 0.0), (5, 0.5)])
def test_colony_matches_distinct_gaussian_engines(size, budget, damp):
    colony = Colony()
    references = {}
    for index in range(size):
        template, arrays, observations = gaussian_goal_chain(mean=0.25 * index)
        iid = colony.spawn(template, arrays, observations=observations)
        engine = Engine(compile_template(template, arrays=arrays))
        state = engine.bind_data(engine.init_state(), observations)
        references[iid] = engine.run(state, budget, damp=damp, tol=1e-6)
    assert len(colony.engines) == 1
    actual = colony.run(budget, damp=damp, tol=1e-6)
    for iid, reference in references.items():
        assert_result_equal(actual[iid], reference)
        assert colony.instances[iid].state is actual[iid].state
        assert colony.last_results[iid] is actual[iid]
        if budget and not damp:
            assert actual[iid].status is RunStatus.CONVERGED
            assert actual[iid].iterations == 1


@pytest.mark.parametrize("failure", ["nonfinite_gaussian", "empty_support"])
def test_exact_colony_preserves_failed_input_and_solves_healthy_peer(failure):
    if failure == "nonfinite_gaussian":
        template, arrays, observations = gaussian_goal_chain()
        status = RunStatus.DIVERGED
    else:
        template = unit_categorical()
        arrays, observations = {"p": np.array([0.2, 0.3, 0.5])}, {}
        status = RunStatus.INFEASIBLE
    colony = Colony()
    rejected, healthy = [
        colony.spawn(template, arrays, observations=observations) for _ in range(2)
    ]
    engine = colony.engines[colony.instances[rejected].key]
    initial = colony.instances[rejected].state
    if failure == "nonfinite_gaussian":
        initial = _replace(initial, g_eta=initial.g_eta.at[0].set(jnp.nan))
    else:
        initial = _replace(initial, messages=jnp.full_like(initial.messages, NEG))
    colony.instances[rejected].state = initial
    references = {
        iid: engine.run(instance.state, 5)
        for iid, instance in colony.instances.items()
    }
    actual = colony.run(5)
    for iid, reference in references.items():
        assert_result_equal(actual[iid], reference)
    assert actual[rejected].status is status
    assert actual[rejected].iterations == 0
    assert_state_equal(actual[rejected].state, initial)
    assert actual[healthy].status is RunStatus.CONVERGED
    assert actual[healthy].iterations == 1


def test_real_delta_phases_and_colony_full_hardening_remain_distinct(monkeypatch):
    template = unit_categorical([ConstraintSpec(ConstraintKind.DELTA, "x")])
    arrays = {"p": np.array([0.2, 0.3, 0.5])}
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.init_state()
    flags = []
    original = engine._iter_step

    def observe(*args):
        flags.append(args[-1])
        return original(*args)

    monkeypatch.setattr(engine, "_iter_step", observe)
    for harden_last in (0, 3, 6):
        flags.clear()
        result = engine.run(state, 6, harden_last=harden_last)
        assert result.status is RunStatus.BUDGET_EXHAUSTED
        assert flags == [False] * (6 - harden_last) + [True] * harden_last
        expected = [0.0, 0.0, 1.0] if harden_last else arrays["p"]
        np.testing.assert_allclose(engine.marginals(result.state)["x"], expected)
    for size in (1, 2):
        colony = Colony()
        ids = [colony.spawn(template, arrays) for _ in range(size)]
        for result in colony.run(6).values():
            assert_result_equal(result, engine.run(state, 6, harden_last=6))
        for iid in ids:
            np.testing.assert_array_equal(colony.marginals(iid)["x"], [0.0, 0.0, 1.0])

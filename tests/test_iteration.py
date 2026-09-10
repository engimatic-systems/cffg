"""Internal outcome ownership and the shared public result boundary."""

import ast
import inspect
from dataclasses import FrozenInstanceError, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_compiler_contracts import model
from test_engine_contracts import ordinary_case

from cffg import iteration, iteration_types, run_results
from cffg.colony import Colony
from cffg.compiled import NEG
from cffg.compiler import compile_template
from cffg.engine import Engine, OuterTerminationReason, RunStatus, _replace
from cffg.iteration_types import IterationOutcome, IterationReason


def make_engine(name="hmm"):
    template, arrays = model(name)
    return Engine(compile_template(template, arrays=arrays), schedule="flood")


def make_outcome(state, **changes):
    values = {
        "state": state,
        "reason": IterationReason.BUDGET,
        "message_deltas": [1.0, 0.0],
        "free_energies": [1.0, 1.0],
        "residual": 0.0,
        "inner_code": 0,
        "inner_residual": 0.0,
        "state_finite": True,
        "zero_support": False,
    }
    values.update(changes)
    return IterationOutcome(**values)


def test_outcome_traces_are_immutable_and_keep_host_precision():
    engine = make_engine()
    distinguishable = 1.0 + 2.0**-40
    incoming = np.asarray([distinguishable, 0.0])
    with jax.enable_x64(False):
        outcome = make_outcome(engine.init_state(), message_deltas=incoming)
        incoming[:] = 9
        exposed = np.asarray(outcome.message_deltas)
        exposed[:] = 7
        result = run_results.finalize_iteration(outcome)
    assert result.message_deltas == (distinguishable, 0.0)
    assert np.asarray(outcome.message_deltas).dtype == np.dtype(np.float64)
    with pytest.raises(FrozenInstanceError):
        outcome.residual = 2.0
    with pytest.raises(TypeError, match="read-only"):
        outcome.message_deltas[0] = 2.0


def test_numeric_termination_facts_are_fixed_shape_and_batch_under_jit():
    outcome = make_outcome(make_engine().init_state())
    rejected = replace(
        outcome,
        reason=IterationReason.INITIAL_NONFINITE,
        message_deltas=[],
        free_energies=[],
        residual=float("inf"),
        inner_code=2,
        inner_residual=float("inf"),
        state_finite=False,
        zero_support=None,
    )
    facts = (outcome.numerical_facts, rejected.numerical_facts)
    assert all(value.shape == () for item in facts for value in item)
    batch = jax.tree_util.tree_map(lambda *values: np.stack(values), *facts)
    transformed = jax.jit(jax.vmap(lambda value: value))(batch)
    np.testing.assert_array_equal(
        transformed.reason, [IterationReason.BUDGET, IterationReason.INITIAL_NONFINITE]
    )
    np.testing.assert_array_equal(transformed.state_finite, [1, 0])
    np.testing.assert_array_equal(transformed.zero_support, [0, -1])
    facts[0].reason[...] = IterationReason.UPDATE_NONFINITE
    assert outcome.reason is IterationReason.BUDGET


@pytest.mark.parametrize("fault", ("running", "nonfinite_delta", "no_fe_comparison"))
def test_public_finalizer_rejects_unfinished_or_unfounded_convergence(fault):
    changes = {"reason": IterationReason.FE_STATIONARY}
    if fault == "running":
        changes["reason"] = IterationReason.RUNNING
    elif fault == "nonfinite_delta":
        changes["message_deltas"] = [1.0, float("nan")]
    else:
        changes["free_energies"] = [1.0]
    outcome = make_outcome(make_engine().init_state(), **changes)
    with pytest.raises(ValueError, match="not terminal|lacks"):
        run_results.finalize_iteration(outcome)


@pytest.mark.parametrize("code", (1, 2))
def test_functional_rejection_precedes_counted_step_and_keeps_picard_tag(
    monkeypatch, code
):
    engine = make_engine("gfe")
    initial = engine.init_state()
    updated = _replace(
        initial, params={**initial.params, "test:updated": jnp.asarray(1)}
    )

    def functional(state):
        assert state is initial
        return updated, 0.0, code, 0.125

    def forbidden(*args, **kwargs):
        raise AssertionError("Functional rejection precedes the scheduled step and FE")

    monkeypatch.setattr(engine, "_update_mu3", functional)
    monkeypatch.setattr(engine, "_iter_step", forbidden)
    monkeypatch.setattr(engine, "bfe", forbidden)
    result = engine.run(initial, 3, harden_last=0, outer_solver="picard")
    assert result.state is updated
    assert result.status is (
        RunStatus.BUDGET_EXHAUSTED if code == 1 else RunStatus.DIVERGED
    )
    assert result.inner_status is result.status
    assert result.residual == result.inner_residual == 0.125
    assert result.iterations == 0
    assert result.message_deltas == result.free_energies == ()
    assert result.termination_reason is OuterTerminationReason.LEGACY_PICARD


@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("initial", ("valid", "nonfinite", "empty_support"))
def test_picard_initial_rejection_and_valid_zero_budget_have_distinct_tags(
    enabled, initial
):
    engine = make_engine("gfe")
    with jax.enable_x64(enabled):
        state = engine.init_state()
        if initial == "nonfinite":
            state = _replace(state, messages=state.messages.at[-1].set(jnp.nan))
        elif initial == "empty_support":
            state = _replace(state, messages=jnp.full_like(state.messages, NEG))
        result = engine.run(state, 0, outer_solver="picard")
    assert result.iterations == 0
    assert result.termination_reason is (
        OuterTerminationReason.LEGACY_PICARD
        if initial == "valid"
        else OuterTerminationReason.NOT_RUN
    )
    assert (
        result.status
        is {
            "valid": RunStatus.BUDGET_EXHAUSTED,
            "nonfinite": RunStatus.DIVERGED,
            "empty_support": RunStatus.INFEASIBLE,
        }[initial]
    )


def test_engine_and_actual_colony_batch_use_the_same_result_owner(monkeypatch):
    colony = Colony()
    ids = []
    for i in range(2):
        template, arrays, observations = ordinary_case(i)
        ids.append(colony.spawn(template, arrays, observations=observations))
    assert colony.instances[ids[0]].key == colony.instances[ids[1]].key
    engine = colony.engines[colony.instances[ids[0]].key]
    seen = []
    original = run_results.finalize_iteration

    def record(outcome, **kwargs):
        assert isinstance(outcome, IterationOutcome)
        seen.append(outcome)
        return original(outcome, **kwargs)

    monkeypatch.setattr(run_results, "finalize_iteration", record)
    engine.run(colony.instances[ids[0]].state, 3, harden_last=3)
    colony.run(3)
    assert len(seen) == 3


@pytest.mark.parametrize("module", (iteration, iteration_types))
def test_iteration_owners_do_not_import_the_facade_or_factor_rules(module):
    tree = ast.parse(inspect.getsource(module))
    imports = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not imports.intersection(
        {
            "cffg.engine",
            "cffg.compiler",
            "cffg.compiled",
            "cffg.rules",
            "cffg.gauss",
            "cffg.efrules",
            "cffg.recurrence",
        }
    )
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names.intersection({"Engine", "Arena", "FactorType", "RunStatus"})

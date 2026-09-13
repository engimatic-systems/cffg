"""Objective ownership, current-instance inputs and unchanged ledger gates."""

import ast
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_compiler_contracts import model
from test_constraint_data import constraint_model, examples
from test_engine_contracts import ordinary_case

from cffg import engine as engine_module
from cffg import ledger, ledger_terms
from cffg.compiler import compile_template
from cffg.engine import Engine, _replace


def engine_for(name):
    template, arrays = model(name)
    return Engine(compile_template(template, arrays=arrays), mu3_samples=16)


def evaluate(function, state):
    return function(
        state.messages,
        state.g_eta,
        state.g_lam,
        state.g_obs,
        state.g_mask,
        state.ef_eta,
        state.ef_obs,
        state.ef_mask,
        state.params,
    )


def test_raw_factory_keeps_its_existing_engine_alias():
    assert engine_module._make_bfe is ledger.make_belief_energy
    engine = engine_for("hmm")
    assert callable(engine._ledger.belief_kernel)
    assert not hasattr(engine._ledger, "run")


@pytest.mark.parametrize("gate", ("max", "unprepared", "empty", "valid"))
def test_host_gate_order_and_current_evaluator_are_preserved(monkeypatch, gate):
    template, arrays = model("hmm")
    engine = Engine(
        compile_template(
            template, arrays=arrays, semiring="max" if gate == "max" else "sum"
        )
    )
    state = engine.init_state()
    calls = []
    precise = 1.0 + 2.0**-40

    def ready(current):
        assert current is state
        calls.append("ready")
        if gate == "unprepared":
            raise RuntimeError("unprepared")

    def support(current):
        assert current is state
        calls.append("support")
        return gate == "empty"

    def raw(*values):
        calls.append("raw")
        assert values[-1] is state.params
        assert values[0] is state.messages
        return np.asarray(precise, dtype=np.float64)

    monkeypatch.setattr(engine, "_require_mu3_ready", ready)
    monkeypatch.setattr(engine, "_has_zero_support", support)
    monkeypatch.setattr(engine, "_bfe", raw)
    with jax.enable_x64(False):
        if gate in ("max", "unprepared"):
            with pytest.raises(RuntimeError, match="objective|unprepared"):
                engine.bfe(state)
        else:
            assert engine.bfe(state) == (float("inf") if gate == "empty" else precise)
    assert (
        calls
        == {
            "max": [],
            "unprepared": ["ready"],
            "empty": ["ready", "support"],
            "valid": ["ready", "support", "raw"],
        }[gate]
    )


@pytest.mark.parametrize("enabled", (False, True))
def test_raw_direct_jit_and_vmap_use_each_instances_current_values(enabled):
    engines, states = [], []
    with jax.enable_x64(enabled):
        for index in (0, 1):
            template, arrays, observations = ordinary_case(index)
            engine = Engine(compile_template(template, arrays=arrays))
            engines.append(engine)
            states.append(engine.bind_data(engine.init_state(), observations))
        raw = engines[0]._ledger.belief_kernel
        reference = jnp.asarray(
            [engine.bfe(state) for engine, state in zip(engines, states)]
        )
        direct = jnp.asarray([evaluate(raw, state) for state in states])
        compiled = jnp.asarray([evaluate(jax.jit(raw), state) for state in states])
        stacked = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *states)
        batched = jax.jit(jax.vmap(lambda state: evaluate(raw, state)))(stacked)
        tolerance = 8 * np.finfo(np.asarray(reference).dtype).eps
        for actual in (direct, compiled, batched):
            np.testing.assert_allclose(
                actual, reference, rtol=tolerance, atol=tolerance
            )
        assert not np.isclose(reference[0], reference[1])


@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("kind", ("exactly_k", "at_most_k", "table"))
def test_assignment_scoring_uses_current_parameters_and_decodes_only_at_facade(
    monkeypatch, enabled, kind
):
    engines, states = [], []
    with jax.enable_x64(enabled):
        for value in examples(kind):
            template, arrays = constraint_model(kind, value)
            engine = Engine(compile_template(template, arrays=arrays))
            engines.append(engine)
            states.append(engine.init_state())
        engine = engines[0]
        assignment = {"a": 0, "b": 1}
        assert engine._ledger.assignment_energy(assignment, states[0].params) == float(
            "inf"
        )
        assert engine._ledger.assignment_energy(assignment, states[1].params) == 0.0
        calls = []

        def decode(state):
            calls.append(state)
            return assignment

        monkeypatch.setattr(engine, "decode", decode)
        supplied = _replace(
            states[1], messages=jnp.full_like(states[1].messages, jnp.nan)
        )
        assert engine.objective(supplied) == 0.0
        assert len(calls) == 1 and calls[0] is supplied


@pytest.mark.parametrize("enabled", (False, True))
def test_two_slice_functional_ledger_charges_one_shared_kl_without_updating(
    monkeypatch, enabled
):
    engine = engine_for("mu3")
    assert len(engine.arena.mu3_groups) == 1
    assert sum(len(group.members) for group in engine.arena.gfe_groups) == 2

    def forbidden(*args, **kwargs):
        raise AssertionError("Objective evaluation must not update functional state")

    monkeypatch.setattr(engine, "_update_mu3", forbidden)
    with jax.enable_x64(enabled):
        state, _ = engine.prepare_mu3(engine.init_state(), jax.random.key(7))
        before = engine.bfe(state)
        key = engine.arena.mu3_groups[0].name + ":kl"
        changed = _replace(
            state, params={**state.params, key: state.params[key] + 0.125}
        )
        after = engine.bfe(changed)
        tolerance = 8 * np.finfo(np.asarray(state.messages).dtype).eps
        np.testing.assert_allclose(
            after - before, 0.125, rtol=tolerance, atol=tolerance
        )
        assert np.asarray(state.params[key]).shape == ()


@pytest.mark.parametrize("module", (ledger, ledger_terms))
def test_objective_owners_do_not_import_runtime_orchestration(module):
    tree = ast.parse(inspect.getsource(module))
    imports = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not imports.intersection(
        {
            "cffg.engine",
            "cffg.agent",
            "cffg.iteration",
            "cffg.exact_strategy",
            "cffg.outer_strategy",
            "cffg.run_results",
            "cffg.schedule_program",
        }
    )
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names.intersection({"Engine", "RunStatus", "RunResult"})

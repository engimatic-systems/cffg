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
from cffg.core import BlobRef, FactorSpec, Template, VariableSpec
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


def test_public_objective_contract_matches_the_two_prior_score():
    template = Template(
        "two_priors",
        variables={"x": VariableSpec("x", 2)},
        factors={name: FactorSpec(name, "categorical", {"out": "x"}, param_blobs={"statistic": "p"})
                 for name in ("a", "b")},
        blobs={"p": BlobRef("p", "p.npy")},
    )
    p = np.array([0.2, 0.8])
    engine = Engine(compile_template(template, arrays={"p": p}, semiring="max"))
    state = engine.run(engine.init_state(), 10).require_converged()
    assert engine.objective_contract["objective"] == "decoded_assignment_energy"
    assert engine.objective_contract["approximate"] is False
    assert engine.decode(state) == {"x": 1}
    np.testing.assert_allclose(engine.objective(state), -2 * np.log(p[1]))
    with pytest.raises(RuntimeError, match="Engine.objective"):
        engine.bfe(state)

    engine = Engine(compile_template(template, arrays={"p": p}, semiring="sum"))
    state = engine.run(engine.init_state(), 10).require_converged()
    assert engine.objective_contract["objective"] == "declared_bethe_or_generalised_energy"
    np.testing.assert_allclose(engine.bfe(state), -np.log(p @ p))


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


@pytest.mark.parametrize("enabled", (False, pytest.param(True, marks=pytest.mark.slow)))
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


def assert_reconciles(engine, state):
    breakdown = engine.fe_breakdown(state)
    scalar = engine.bfe(state)
    if np.isfinite(scalar):
        bound = (256 * np.finfo(state.messages.dtype).eps * max(1, len(breakdown.entries))
                 * max(1, abs(scalar)))
        np.testing.assert_allclose(breakdown.total, scalar, atol=bound, rtol=0)
        np.testing.assert_allclose(sum(row.total for row in breakdown.entries), scalar,
                                   atol=bound, rtol=0)
    else:
        assert not np.isfinite(breakdown.total)
        assert any(not np.isfinite(row.total) for row in breakdown.entries)
    return breakdown


@pytest.mark.parametrize("states,enabled", [(2, True), (80, False)])
def test_breakdown_charges_fixed_channel_negative_mutual_information(states, enabled):
    from test_gfe import make_model

    # The input prior is uniform. Each half maps deterministically to a
    # distinct label; the third label has zero probability under every input.
    with jax.enable_x64(enabled):
        dtype = np.float64 if enabled else np.float32
        channel = np.zeros((3, states), dtype=dtype)
        channel[0, :states // 2] = 1
        channel[1, states // 2:] = 1
        prior = np.full(states, 1 / states, dtype=dtype)
        goal = np.array([0.8, 0.2, 0.0], dtype=dtype)
        template = make_model(channel, goal, prior)
        template.variables.update(z=VariableSpec("z", states), x=VariableSpec("x", 3))
        engine = Engine(compile_template(template, arrays={"A": channel, "c": goal, "d": prior}))
        report = assert_reconciles(engine, engine.init_state())
        entry, = [e for e in report.entries if e.owner_kind == "observation_gfe"]
        assert entry.owner_name == "obs"

        # For a disjoint deterministic channel, stationary output mass is
        # proportional to sqrt(goal * prior output mass). Expand each label
        # uniformly over its input states and enumerate the joint explicitly.
        px = np.array([2/3, 1/3, 0])
        pz = np.repeat(px[:2] / (states // 2), states // 2)
        joint = channel * pz[None, :]
        supported = joint > 0
        mi = np.sum(joint[supported] * np.log(joint[supported] / (px[:, None] * pz)[supported]))
        tol = 256 * np.finfo(dtype).eps * max(1, abs(entry.total))
        expected = {
            "extrinsic": -px[:2] @ np.log(goal[:2]),
            "epistemic": -mi,
            "state_negative_entropy": pz @ np.log(pz),
        }
        assert set(entry.components) == set(expected)
        for label, value in expected.items():
            np.testing.assert_allclose(entry.components[label], value, atol=tol, rtol=0)
        np.testing.assert_allclose(sum(entry.components.values()), entry.total, atol=tol, rtol=0)


def test_breakdown_names_shared_owners_and_checks_ledger_gates():
    from test_goal_hyperprior import _arrays, _model

    engine = Engine(compile_template(_model(slices=2), arrays=_arrays()))
    report = assert_reconciles(engine, engine.init_state())
    shared, = [e for e in report.entries if e.owner_kind == "goal_dirichlet_parameter"]
    assert shared.owner_name == engine.arena.goal_parameter_plans[0].prior_blob
    assert sum(e.owner_kind == "observation_gfe" for e in report.entries) == 2

    engine = engine_for("mu3")
    with pytest.raises(RuntimeError):
        engine.fe_breakdown(engine.init_state())
    state, _ = engine.prepare_mu3(engine.init_state(), jax.random.key(7))
    report = assert_reconciles(engine, state)
    kl, = [e for e in report.entries if e.owner_kind == "importance_sampled_parameter_kl"]
    group, = engine.arena.mu3_groups
    assert kl.owner_name == group.param_blob
    np.testing.assert_allclose(kl.total, state.params[group.name + ":kl"])
    assert all("parameter_aware_epistemic" in e.components for e in report.entries
               if e.owner_kind == "observation_gfe")

    template, arrays = model("hmm")
    engine = Engine(compile_template(template, arrays=arrays, semiring="max"))
    with pytest.raises(RuntimeError, match="objective"):
        engine.fe_breakdown(engine.init_state())


@pytest.mark.parametrize("schedule", ["flood", "two_pass"])
def test_breakdown_ordinary_owner_names_and_missing_data(schedule):
    template, arrays = model("gaussian")
    engine = Engine(compile_template(template, arrays=arrays), schedule=schedule)
    initial = engine.init_state()
    assert_reconciles(engine, initial)
    state = engine.run(initial, 100, tol=1e-8).require_converged()
    entries = assert_reconciles(engine, state).entries
    assert {e.owner_name for e in entries if e.owner_kind == "factor"} == set(template.factors)
    assert {e.owner_name for e in entries if e.owner_kind == "edge_entropy"} == {
        name for name, edge in engine.arena.edges.items() if edge.interior
    }
    assert engine.fe_breakdown(state).entries == entries


@pytest.mark.parametrize("name,kind", [
    ("parameter_edge", "dirichlet_region"), ("constraint", "constraint"),
    ("ef", "factor"), ("mixture", "factor"), ("hybrid_gfe", "hybrid_gfe"),
    ("recurrence", "factor"),
])
def test_detailed_ledger_includes_specialized_owners(name, kind):
    engine = engine_for(name)
    state = engine.init_state()
    report = assert_reconciles(engine, state)
    owners = [e for e in report.entries if e.owner_kind == kind]
    assert owners
    if name == "hybrid_gfe":
        assert all(e.bound_direction == "reported_gfe_lower_bounds_exact" for e in owners)


def test_transition_diagnostic_excludes_downstream_message_energy():
    from cffg.models import build_rollout

    template, arrays = build_rollout(
        "tm", steps=2, past=0, d=np.full(4, 0.25), A=np.eye(4),
        c=np.full(4, 0.25), e=np.full(2, 0.5), Bs=np.full((2, 4, 4), 0.25),
        gfe=False, epistemic_b=True,
    )
    engine = Engine(compile_template(template, arrays=arrays))
    report = assert_reconciles(engine, engine.init_state())
    rows = [r for r in report.entries if r.owner_kind == "transition_gfe"]
    assert [r.owner_name for r in rows] == ["tm1", "tm2"]
    for row in rows:
        assert "optimistic_transition_information_bound" in row.components
        np.testing.assert_allclose(sum(row.components.values()), row.total)


def test_empty_discrete_support_reports_infinite_named_guard():
    from cffg import rules

    engine = engine_for("hmm")
    state = engine.init_state()
    state = _replace(state, messages=jnp.full_like(state.messages, rules.NEG))
    report = assert_reconciles(engine, state)
    proof, = report.entries
    assert proof.owner_name == "empty_exact_support" and proof.total == np.inf

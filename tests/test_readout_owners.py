"""Readout delegation, bounded structure views and public compatibility."""

import ast
import inspect

import numpy as np
import pytest
from test_compiler_contracts import model
from test_shared_goals import mixed_goals

from cffg import engine as engine_module
from cffg.compiler import compile_template
from cffg.engine import Engine
from cffg.execution import transport
from cffg.lifecycle import StateLifecycle
from cffg.readouts import (
    discrete,
    exponential_family,
    gaussian,
    hybrid,
    parameters,
    recurrence,
    registered,
    types,
    views,
)

OWNERS = {
    "marginals": discrete,
    "node_beliefs": discrete,
    "decode": discrete,
    "dirichlet_marginals": parameters,
    "goal_marginals": parameters,
    "goal_prior_updates": parameters,
    "gaussian_marginals": gaussian,
    "gaussian_node_stats": gaussian,
    "gaussian_mixture_predictives": gaussian,
    "recurrence_pair_beliefs": recurrence,
    "recurrence_weight_stats": recurrence,
    "ef_marginals": exponential_family,
    "hybrid_gfe_diagnostics": hybrid,
    "kernel_readouts": registered,
}


def engine_for(name="hmm"):
    template, arrays = model(name)
    return Engine(compile_template(template, arrays=arrays))


@pytest.mark.parametrize("name", OWNERS)
def test_public_method_delegates_to_its_focused_owner(monkeypatch, name):
    engine = engine_for()
    state = engine.init_state()
    marker = object()
    calls = []

    def readout(*args, **kwargs):
        calls.append((args, kwargs))
        return marker

    monkeypatch.setattr(OWNERS[name], name, readout)
    assert getattr(engine, name)(state) is marker
    assert len(calls) == 1
    args, callbacks = calls[0]
    expected = (state,) if name == "decode" else (engine.arena, state)
    assert len(args) == len(expected)
    assert all(left is right for left, right in zip(args, expected))
    for callback in callbacks.values():
        assert callback.__self__ is engine


def test_existing_value_and_transport_imports_remain_aliases():
    assert engine_module.DirichletMarginal is types.DirichletMarginal
    assert engine_module.EfMarginal is types.EfMarginal
    for name in ("_disc_rows", "_disc_both", "_gauss_rows", "_obs_rows"):
        assert getattr(engine_module, name) is getattr(transport, name)


class BoundedView:
    """Test that a declared structural view includes transitive helper reads."""

    def __init__(self, arena, contract):
        self._arena = arena
        self._allowed = {
            name
            for base in contract.__mro__
            for name in vars(base)
            if not name.startswith("_")
        }

    def __getattr__(self, name):
        assert name in self._allowed, f"Undeclared structural read: {name}"
        return getattr(self._arena, name)


def test_readouts_accept_only_their_declared_structural_capabilities():
    cases = (
        ("hmm", discrete.marginals, views.DiscreteView),
        ("hmm", discrete.node_beliefs, views.NodeView),
        ("hmm", registered.kernel_readouts, views.KernelView),
        ("gaussian", gaussian.gaussian_marginals, views.GaussianView),
        ("gaussian", registered.kernel_readouts, views.KernelView),
        ("parameter_edge", parameters.dirichlet_marginals, views.DiscreteView),
        ("ef", exponential_family.ef_marginals, views.EdgesView),
        ("ef", registered.kernel_readouts, views.KernelView),
        ("mixture", gaussian.gaussian_mixture_predictives, views.MixtureView),
        ("recurrence", recurrence.recurrence_pair_beliefs, views.RecurrenceView),
        ("hybrid_gfe", hybrid.hybrid_gfe_diagnostics, views.HybridGfeView),
    )
    for name, readout, contract in cases:
        engine = engine_for(name)
        assert isinstance(
            readout(BoundedView(engine.arena, contract), engine.init_state()), dict
        )


@pytest.mark.parametrize("method", ("goal_marginals", "goal_prior_updates"))
def test_goal_readouts_keep_the_public_parameter_belief_dependency(monkeypatch, method):
    template, arrays = mixed_goals()
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.init_state()
    beliefs = {
        edge: types.DirichletMarginal(
            np.ones(plan.cardinality),
            np.ones(plan.cardinality),
            np.zeros(plan.cardinality),
            True,
        )
        for plan in engine.arena.goal_parameter_plans
        for edge in plan.parameter_edges
    }
    calls = []

    def parameter_beliefs(current):
        assert current is state
        calls.append(current)
        return beliefs

    monkeypatch.setattr(engine, "dirichlet_marginals", parameter_beliefs)
    result = getattr(engine, method)(state)
    assert len(calls) == 1
    if method == "goal_marginals":
        assert all(
            any(value is belief for belief in beliefs.values())
            for value in result.values()
        )
    else:
        assert set(result) == {
            plan.prior_blob for plan in engine.arena.goal_parameter_plans
        }
        assert all(
            np.array_equal(value, np.ones_like(value)) for value in result.values()
        )


def test_gaussian_statistics_keep_the_public_pair_belief_dependency(monkeypatch):
    engine = engine_for()
    state = engine.init_state()
    pair = {
        "destination_probability": np.array([0.2, 0.8]),
        "source_probability": np.array([0.6, 0.4]),
        "destination_support": np.array([True, True]),
        "source_support": np.array([True, True]),
    }
    calls = []

    def pairs(current):
        assert current is state
        calls.append(current)
        return {"example": pair}

    monkeypatch.setattr(engine, "recurrence_pair_beliefs", pairs)
    result = engine.gaussian_node_stats(state)
    assert len(calls) == 1
    assert result["example"]["out_probability"] is pair["destination_probability"]
    assert result["example"]["in_support"] is pair["source_support"]


def test_decode_keeps_public_marginal_order_and_tie_breaking(monkeypatch):
    engine = engine_for()
    state = engine.init_state()
    monkeypatch.setattr(
        engine,
        "marginals",
        lambda current: {"last": np.array([0.1, 0.9]), "first": np.array([0.5, 0.5])},
    )
    result = engine.decode(state)
    assert list(result.items()) == [("last", 1), ("first", 0)]


def test_goal_prior_replacement_delegates_to_state_lifecycle(monkeypatch):
    engine = engine_for()
    state = engine.init_state()
    priors = {"example": np.ones(2)}
    marker = object()

    def install(owner, current, supplied):
        assert owner is engine._state_lifecycle
        assert current is state and supplied is priors
        return marker

    monkeypatch.setattr(StateLifecycle, "with_goal_priors", install)
    assert engine.with_goal_priors(state, priors) is marker


def test_readout_owners_do_not_import_the_facade_or_run_controllers():
    for module in set(OWNERS.values()):
        tree = ast.parse(inspect.getsource(module))
        imports = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        assert not imports.intersection(
            {
                "cffg.engine",
                "cffg.agent",
                "cffg.iteration",
                "cffg.outer_strategy",
                "cffg.exact_strategy",
                "cffg.run_results",
            }
        )
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert not names.intersection({"Engine", "Arena", "RunResult"})


def test_engine_contains_no_family_rule_or_local_readout_calculation():
    tree = ast.parse(inspect.getsource(engine_module))
    family_modules = {
        "ef",
        "efrules",
        "gauss",
        "hybrid_gfe",
        "kernels",
        "mixg",
        "recurrence",
        "rules",
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        ):
            assert node.func.value.id not in family_modules | {"np", "jnp"}

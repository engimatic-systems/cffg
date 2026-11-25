"""Readout delegation, bounded structure views and public compatibility."""

import ast
import inspect

import jax
import numpy as np
import pytest
from test_compiler_contracts import model
from test_shared_goals import mixed_goals

from cffg import engine as engine_module
from cffg.compiler import compile_template
from cffg.engine import Engine
from cffg.execution import transport
from cffg.lifecycle import StateLifecycle
from cffg.models.kalman import lgssm
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
from cffg.runtime_types import _replace

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


@pytest.mark.parametrize("x64", [False, True])
def test_gaussian_readouts_reject_a_rank_deficient_observation_then_recover(x64):
    # Three measured combinations of four unknowns leave one direction unobserved.
    # Both matrices have produced a spurious positive final pivot in host LAPACK.
    sensing = ([[4, -16, 9, -1], [11, 16, 14, -15], [11, 17, 15, 6]] if x64 else
               [[-6, 2, -4, 10], [11, 11, 3, -7], [-4, 13, -6, 15]])
    with jax.enable_x64(x64):
        dtype = np.float64 if x64 else np.float32
        h = np.asarray(sensing, dtype=dtype)
        identity = np.eye(4, dtype=dtype)
        template, arrays = lgssm(2, identity, identity, h, np.eye(3), np.zeros(4), identity)
        engine = Engine(compile_template(template, arrays=arrays), schedule="flood")
        state = engine.step(engine.bind_data(engine.init_state(), {"y0": np.ones(3)}))
        info = engine.arena.edges["x0o"]
        slots = (info.slot0, info.slot1)
        packed = np.asarray(state.g_lam)[engine.arena.g_lam_indices(slots, 4)].sum(axis=0)
        np.testing.assert_array_equal(packed, h.T @ h)
        for mean, cov in (engine.gaussian_marginal(state, "x0o"), engine.gaussian_marginals(state)["x0o"]):
            assert mean.dtype == cov.dtype == dtype
            assert np.isnan(mean).all() and np.isnan(cov).all()

        # The next sweep delivers the prior, which supplies the missing information.
        state = engine.step(state)
        posterior = np.linalg.inv(np.eye(4) + h.T @ h)
        mean, cov = engine.gaussian_marginal(state, "x0o")
        bulk = engine.gaussian_marginals(state)
        assert set(bulk) == {edge.name for edge in engine.arena.edges.values() if edge.dim}
        np.testing.assert_array_equal(mean, bulk["x0o"][0])
        np.testing.assert_array_equal(cov, bulk["x0o"][1])
        tol = 16 * np.finfo(dtype).eps
        np.testing.assert_allclose(mean, posterior @ h.T @ np.ones(3), rtol=tol, atol=1e-14)
        np.testing.assert_allclose(cov, posterior, rtol=tol, atol=1e-14)


@pytest.mark.parametrize("x64", [False, True])
def test_gaussian_readouts_at_mixed_coordinate_scales(x64):
    dim = 40
    with jax.enable_x64(x64):
        dtype = np.float64 if x64 else np.float32
        eye = np.eye(dim)
        template, arrays = lgssm(1, eye, eye, eye, eye, np.zeros(dim), eye)
        engine = Engine(compile_template(template, arrays=arrays))
        initial = engine.init_state()
        edge = engine.arena.edges["x0"]
        p_indices = engine.arena.g_lam_indices(edge.slot0, dim)
        h_indices = engine.arena.g_eta_indices(edge.slot0, dim)
        scale = np.geomspace(1e-5, 1e5, dim).astype(dtype)
        normalized = eye + np.ones((dim, dim)) / dim
        p = (scale[:, None] * normalized * scale).astype(dtype)
        h = scale.copy()
        state = _replace(initial, g_lam=initial.g_lam.at[p_indices].set(p),
                         g_eta=initial.g_eta.at[h_indices].set(h))
        single = engine.gaussian_marginal(state, "x0")
        bulk = engine.gaussian_marginals(state)["x0"]
        for value, other in zip(single, bulk, strict=True):
            assert value.dtype == dtype
            np.testing.assert_array_equal(value, other)
        tol = 2 * dim * np.finfo(dtype).eps
        np.testing.assert_allclose(single[0] * scale, .5, rtol=tol)
        np.testing.assert_allclose(single[1] * scale[:, None] * scale, eye - .5 / dim, rtol=tol)

        for bad in (np.zeros_like(p), -p, np.triu(p), np.full_like(p, np.nan)):
            invalid = _replace(state, g_lam=state.g_lam.at[p_indices].set(bad))
            for moments in (engine.gaussian_marginal(invalid, "x0"), engine.gaussian_marginals(invalid)["x0"]):
                assert all(np.isnan(value).all() for value in moments)


def test_float32_readout_checks_the_precision_and_returned_covariance_in_their_own_dtype():
    with jax.enable_x64(False):
        eye = np.eye(4, dtype=np.float32)
        template, arrays = lgssm(1, eye, eye, eye, eye, np.zeros(4), eye)
        engine = Engine(compile_template(template, arrays=arrays))
        initial = engine.init_state()
        edge = engine.arena.edges["x0"]
        indices = engine.arena.g_lam_indices(edge.slot0, 4)
        eps = np.finfo(np.float32).eps

        # Rounding a rank-three sensor Gram matrix in float32 can give it a
        # genuinely positive eigenvalue when its stored entries are promoted.
        sensor = np.array([[-13, 8, 5, -2], [-2, 11, -13, 6], [-9, -13, 1, 15]], np.float32) / 7
        # Round each dot product once, independent of the host BLAS backend.
        gram = (sensor.astype(np.float64).T @ sensor.astype(np.float64)).astype(np.float32)
        assert np.linalg.eigvalsh(gram.astype(np.float64))[0] > 0
        # Inverting the second precision leaves an unresolved direction in
        # the returned covariance. The third has margin in both directions.
        for precision, proper in ((gram, False), (8*eps*eye + (1-8*eps), False),
                                  (64*eps*eye + (1-64*eps), True)):
            state = _replace(initial, g_lam=initial.g_lam.at[indices].set(precision))
            for mean, cov in (engine.gaussian_marginal(state, "x0"), engine.gaussian_marginals(state)["x0"]):
                if proper:
                    np.testing.assert_array_equal(mean, 0.)
                    reference = np.linalg.inv(precision.astype(np.float64)).astype(np.float32)
                    np.testing.assert_allclose(cov, reference, rtol=2*eps)
                else:
                    assert np.isnan(mean).all() and np.isnan(cov).all()


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
    if name in ("marginals", "gaussian_marginals"):
        assert callbacks.pop("edges") is None
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

"""Lifecycle ownership, call-time settings and explicit stochastic state."""

from dataclasses import fields

import jax
import numpy as np
import pytest
from test_compiler_contracts import model

from cffg import engine as engine_module
from cffg import outer, runtime_types
from cffg.compiled import NEG
from cffg.compiler import compile_template
from cffg.engine import Engine, RunResult, RunStatus
from cffg.mu3_lifecycle import Mu3Lifecycle, Mu3StateView


def make_engine(name, **options):
    template, arrays = model(name)
    return Engine(compile_template(template, arrays=arrays), **options)


def assert_tree_equal(left, right):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for a, b in zip(
        jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True
    ):
        np.testing.assert_array_equal(a, b)


def test_public_value_aliases_and_state_fields_remain_unchanged():
    for name in (
        "State",
        "RunStatus",
        "RunResult",
        "InferenceError",
        "Mu3ScoreResult",
        "_replace",
    ):
        assert getattr(engine_module, name) is getattr(runtime_types, name)
    assert outer.OuterTerminationReason is runtime_types.OuterTerminationReason
    assert tuple(field.name for field in fields(runtime_types.State)) == (
        "messages",
        "params",
        "g_eta",
        "g_lam",
        "g_obs",
        "g_mask",
        "ef_eta",
        "ef_obs",
        "ef_mask",
    )


@pytest.mark.parametrize(
    "name", ("hmm", "gaussian", "ef", "constraint", "gfe", "parameter_edge")
)
def test_ordinary_initialization_observes_call_time_precision_without_mu3(name):
    with jax.enable_x64(True):
        engine = make_engine(name)
    assert engine._mu3_lifecycle is None
    for enabled in (False, True, False):
        with jax.enable_x64(enabled):
            state = engine.init_state()
            expected = np.dtype(np.float64 if enabled else np.float32)
            assert all(
                value.dtype == expected
                for value in jax.tree_util.tree_leaves(state)
                if value.dtype.kind == "f"
            )
            assert_tree_equal(
                state,
                engine._state_lifecycle.init_state(
                    engine._state_lifecycle.initial_parameters()
                ),
            )
            assert not np.asarray(state.g_lam).any()
            assert not np.asarray(state.ef_eta).any()


@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("name", ("hmm", "gaussian", "ef"))
def test_binding_and_unbinding_preserve_dtype_masks_and_atomic_failure(name, enabled):
    engine = make_engine(name)
    with jax.enable_x64(enabled):
        state = engine.init_state()
        snapshot = jax.tree_util.tree_map(np.array, state)
        arena = engine.arena
        if name == "hmm":
            port, slot = next(iter(arena.data_slots.items()))
            value = 0
        elif name == "gaussian":
            port, (row, dim) = next(iter(arena.gauss_data.items()))
            value = np.full(dim, 0.25)
        else:
            port, (row, _) = next(iter(arena.ef_data.items()))
            value = 2
        bound = engine.bind_data(state, {port: value})
        assert_tree_equal(
            bound, engine._state_lifecycle.bind_data(state, {port: value})
        )
        missing = engine.bind_data(bound, {port: None})
        if name == "hmm":
            card = next(
                edge.card
                for edge in arena.edges.values()
                if slot in (edge.slot0, edge.slot1) and not edge.dim and not edge.ef
            )
            indices = arena.message_indices(slot, card)
            values = np.asarray(bound.messages[indices])
            assert values[0] == 0.0
            assert np.all(values[1:] == np.asarray(NEG, dtype=values.dtype))
            np.testing.assert_allclose(
                missing.messages[indices],
                -np.log(card),
                rtol=8 * np.finfo(values.dtype).eps,
            )
        elif name == "gaussian":
            assert bound.g_mask[row] == 1 and missing.g_mask[row] == 0
            np.testing.assert_array_equal(bound.g_obs, missing.g_obs)
        else:
            assert bound.ef_mask[row] == 1 and missing.ef_mask[row] == 0
            np.testing.assert_array_equal(bound.ef_obs, missing.ef_obs)
        for before, after in zip(
            jax.tree_util.tree_leaves(state),
            jax.tree_util.tree_leaves(bound),
            strict=True,
        ):
            assert before.dtype == after.dtype
        with pytest.raises(KeyError, match="no DATA port"):
            engine.bind_data(state, {port: value, "__invalid_lifecycle_port__": 0})
        assert_tree_equal(state, snapshot)


@pytest.mark.parametrize("enabled", (False, True))
def test_functional_views_replay_current_particle_budget_and_explicit_key(enabled):
    engine = make_engine("mu3", mu3_samples=16)
    owner = engine._mu3_lifecycle
    assert isinstance(owner, Mu3Lifecycle)
    assert Mu3StateView._fields == ("messages", "params")
    with jax.enable_x64(enabled):
        for budget in (16, 32):
            engine.mu3_samples = budget
            state = engine.init_state()
            for plan in owner.view.plans:
                assert state.params[plan.name + ":samples"].shape[0] == budget
            with pytest.raises(ValueError, match="Engine.prepare_mu3"):
                engine._require_mu3_ready(state)
            key = jax.random.key(31)
            prepared, next_key = engine.prepare_mu3(state, key)
            view, direct_key = owner.prepare(
                Mu3StateView(state.messages, state.params), key, samples=budget
            )
            assert_tree_equal(prepared.params, view.params)
            expected_key = key
            for _ in owner.view.plans:
                expected_key, _ = jax.random.split(expected_key)
            np.testing.assert_array_equal(
                jax.random.key_data(next_key), jax.random.key_data(expected_key)
            )
            np.testing.assert_array_equal(
                jax.random.key_data(next_key), jax.random.key_data(direct_key)
            )
            engine._require_mu3_ready(prepared)
            for plan in owner.view.plans:
                assert not bool(state.params[plan.name + ":ready"])
                assert bool(prepared.params[plan.name + ":ready"])


def test_update_receives_current_settings_instead_of_constructor_snapshot(monkeypatch):
    engine = make_engine("mu3", mu3_samples=16, mu3_min_ess_fraction=0.1)
    state = engine.init_state()
    received = []

    def update(owner, view, *, samples, min_ess_fraction):
        assert isinstance(view, Mu3StateView)
        received.append((samples, min_ess_fraction))
        return view, 0.0, 0, 0.0

    monkeypatch.setattr(Mu3Lifecycle, "update", update)
    engine.mu3_samples = 32
    engine.mu3_min_ess_fraction = 0.75
    updated, delta, code, residual = engine._update_mu3(state)
    assert received == [(32, 0.75)]
    assert_tree_equal(updated, state)
    assert (delta, code, residual) == (0.0, 0, 0.0)


@pytest.mark.parametrize(
    "failure", (None, RunStatus.BUDGET_EXHAUSTED, RunStatus.DIVERGED)
)
def test_epoch_scoring_uses_supplied_run_authority_and_explicit_key(
    monkeypatch, failure
):
    engine = make_engine("mu3", mu3_samples=16)
    state = engine.init_state()
    calls = []

    def prepare(current, key):
        next_key, _ = jax.random.split(key)
        return current, next_key

    def run(current, **options):
        assert options == {"n_iters": 4, "tol": 0.125, "harden_last": 0}
        calls.append(None)
        status = (
            failure if failure is not None and len(calls) == 2 else RunStatus.CONVERGED
        )
        return RunResult(current, status, 1, (0.0,), (2.0 * len(calls),), 0.0)

    monkeypatch.setattr(engine, "prepare_mu3", prepare)
    monkeypatch.setattr(engine, "run", run)
    key = jax.random.key(7)
    result = engine.score_mu3_epochs(
        state, key=key, epochs=3, burn_in=1, n_iters=4, tol=0.125
    )
    expected_key = key
    for _ in calls:
        expected_key, _ = jax.random.split(expected_key)
    np.testing.assert_array_equal(
        jax.random.key_data(result.next_key), jax.random.key_data(expected_key)
    )
    assert len(result.runs) == len(calls)
    if failure is None:
        assert result.status is RunStatus.CONVERGED
        assert result.epoch_scores == (2.0, 4.0, 6.0)
        assert result.policy_score == 5.0
    else:
        assert result.status is failure
        assert result.epoch_scores == (2.0,)
        assert np.isnan(result.policy_score)

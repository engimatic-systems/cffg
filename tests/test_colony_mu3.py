"""Fixed particle epochs, independent random streams and member-local outcomes."""

from dataclasses import replace
import re

import jax
import numpy as np
import pytest

from cffg.colony import Colony
from cffg.compiler import compile_template
from cffg.core.compose import merge, scoped
from cffg.core.serialize import from_json, structure_hash, to_json
from cffg.engine import Engine, RunStatus, _replace

from test_mu3 import _arrays, _model
from test_schedule import unit_categorical


def key_bits(key):
    return np.asarray(jax.random.key_data(key))


def state_equal(actual, expected, *, exact=False):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert a.dtype == e.dtype
        if exact or not np.issubdtype(e.dtype, np.inexact):
            np.testing.assert_array_equal(a, e)
        else:
            eps = 64 * np.finfo(e.dtype).eps
            np.testing.assert_allclose(a, e, rtol=eps, atol=eps)


def assert_diagnostics(actual, expected, dtype):
    assert actual.keys() == expected.keys()
    eps = 64 * np.finfo(dtype).eps
    for name, group in expected.items():
        assert actual[name].keys() == group.keys()
        for field, value in group.items():
            if np.issubdtype(np.asarray(value).dtype, np.inexact):
                np.testing.assert_allclose(actual[name][field], value, rtol=eps, atol=eps)
            else:
                np.testing.assert_array_equal(actual[name][field], value)


def test_equal_and_distinct_keys_replay_against_independent_engine():
    model = _model()
    json = to_json(model)
    fingerprint = structure_hash(model)
    colony = Colony()
    left, right = jax.random.key(3), jax.random.PRNGKey(5)
    references = []
    for key, alpha in ((left, .4), (right, 2.), (left, .4)):
        arrays = _arrays()
        arrays["Ac"][0, 1] = alpha
        iid = colony.spawn(model, arrays, mu3_key=key)
        engine = Engine(compile_template(from_json(json), arrays=arrays))
        initial, cursor = engine.prepare_mu3(engine.init_state(), key)
        np.testing.assert_array_equal(key_bits(colony.mu3_key_cursor(iid)), key_bits(cursor))
        state_equal(colony.instances[iid].state, initial, exact=True)
        references.append((iid, engine, initial, cursor))
    assert len(colony.engines) == 1
    results = colony.run(2, tol=1e-6)
    for iid, engine, initial, cursor in references:
        expected = engine.run(initial, 2, tol=1e-6)
        actual = results[iid]
        assert actual.status is expected.status
        assert actual.mu3_statuses == expected.mu3_statuses
        state_equal(actual.state, expected.state)
        eps = 64 * np.finfo(actual.state.messages.dtype).eps
        np.testing.assert_allclose(actual.free_energies, expected.free_energies, rtol=eps, atol=eps)
        assert_diagnostics(colony.mu3_diagnostics(iid), engine.mu3_diagnostics(expected.state), actual.state.messages.dtype)
        np.testing.assert_array_equal(key_bits(colony.mu3_key_cursor(iid)), key_bits(cursor))
        samples = engine.arena.mu3_groups[0].name + ":samples"
        np.testing.assert_array_equal(actual.state.params[samples], initial.params[samples])
    state_equal(results[0].state, results[2].state, exact=True)
    samples = references[0][1].arena.mu3_groups[0].name + ":samples"
    original = results[0].state.params[samples]
    assert not np.array_equal(original, results[1].state.params[samples])
    colony.run(1, tol=1e-6)
    np.testing.assert_array_equal(colony.instances[0].state.params[samples], original)
    np.testing.assert_array_equal(key_bits(colony.mu3_key_cursor(0)), key_bits(references[0][3]))
    peer_result = colony.last_results[1]
    peer_cursor = colony.mu3_key_cursor(1)
    colony.refresh_mu3(0)
    assert 0 not in colony.last_results and colony.last_results[1] is peer_result
    assert colony.instances[1].state is peer_result.state
    assert colony.mu3_key_cursor(1) is peer_cursor
    assert to_json(model) == json and structure_hash(model) == fingerprint


def test_refresh_splits_once_per_evaluator_and_failures_are_atomic():
    model = merge(scoped(_model(), "a"), scoped(_model(), "b"), "pair")
    arrays = {f"{scope}/{k}": v for scope in ("a", "b") for k, v in _arrays().items()}
    colony = Colony()
    with pytest.raises(ValueError, match="mu3_key"):
        colony.spawn(model, arrays)
    with pytest.raises((TypeError, ValueError), match="key"):
        colony.spawn(model, arrays, mu3_key=np.array([17], dtype=np.uint32))
    assert not colony.engines and not colony.instances
    start = jax.random.key(17)
    iid = colony.spawn(model, arrays, mu3_key=start)
    with pytest.raises(ValueError, match="mu3_key"):
        colony.spawn(model, arrays)
    assert list(colony.instances) == [iid]
    engine = next(iter(colony.engines.values()))
    plans = engine.arena.mu3_groups
    assert len(plans) == 2
    state, cursor = engine.prepare_mu3(engine.init_state(), start)
    expected, new = engine.prepare_mu3(state, cursor)
    np.testing.assert_array_equal(key_bits(colony.refresh_mu3(iid)), key_bits(new))
    state_equal(colony.instances[iid].state, expected, exact=True)
    for _ in range(2 * len(plans)):
        start, _ = jax.random.split(start)
    np.testing.assert_array_equal(key_bits(new), key_bits(start))
    inst = colony.instances[iid]
    valid = inst.state
    last = plans[-1]
    for value in (0., np.nan):
        invalid = dict(valid.params)
        invalid[last.count_key] = invalid[last.count_key].at[int(last.rows[0]), 0, 0].set(value)
        inst.state = _replace(valid, params=invalid)
        before = inst.state
        with pytest.raises(ValueError, match=f"{re.escape(last.name)}.*finite, strictly positive"):
            colony.refresh_mu3(iid)
        assert inst.state is before
        np.testing.assert_array_equal(key_bits(colony.mu3_key_cursor(iid)), key_bits(new))
    stale = dict(valid.params)
    stale[last.count_key] = stale[last.count_key].at[int(last.rows[0])].multiply(2)
    inst.state = _replace(valid, params=stale)
    with pytest.raises(ValueError, match=re.escape(last.name)):
        colony.run(2)


def test_deterministic_and_stochastic_members_keep_their_execution_contracts():
    colony = Colony()
    template, arrays = unit_categorical(), {"p": np.array([.25, .25, .50])}
    ordinary = colony.spawn(template, arrays)
    stochastic = colony.spawn(_model(), _arrays(), mu3_key=jax.random.key(13))
    assert colony.mu3_key_cursor(ordinary) is None
    assert colony.mu3_diagnostics(ordinary) == {}
    before = colony.instances[ordinary].state
    with pytest.raises(ValueError, match="compiled_trust_region.*deterministic"):
        colony.run(1, outer_solver="compiled_trust_region")
    assert colony.instances[ordinary].state is before and not colony.last_results
    result = colony.run(2)
    engine = Engine(compile_template(template, arrays=arrays))
    ref = engine.run(engine.init_state(), 2)
    state_equal(result[ordinary].state, ref.state, exact=True)
    assert result[ordinary].free_energies == ref.free_energies
    assert result[ordinary].status is ref.status
    assert len(result[stochastic].mu3_statuses) == 1
    colony.kill(stochastic)
    assert list(colony.last_results) == [ordinary]
    spare = jax.random.key(17)
    other = colony.spawn(template, arrays, mu3_key=spare)
    assert colony.refresh_mu3(other) is spare


@pytest.mark.slow
def test_one_group_routes_converged_low_ess_and_impossible_goal():
    colony = Colony()
    originals = []
    for i, goal in enumerate(([.75, .25], [1., 1e-60], [1., 0.])):
        arrays = dict(_arrays(), c0=np.array(goal))
        key = jax.random.key(i + 13)
        iid = colony.spawn(_model(), arrays, mu3_key=key)
        engine = Engine(compile_template(_model(), arrays=arrays))
        initial, cursor = engine.prepare_mu3(engine.init_state(), key)
        originals.append((iid, engine, initial, cursor))
    assert len(colony.engines) == 1
    outcomes = colony.run(70, tol=1e-7)
    statuses = (RunStatus.CONVERGED, RunStatus.BUDGET_EXHAUSTED, RunStatus.DIVERGED)
    for (iid, engine, initial, cursor), desired in zip(originals, statuses, strict=True):
        expected = engine.run(initial, 70, tol=1e-7)
        actual = outcomes[iid]
        assert actual.status is expected.status is desired
        assert actual.mu3_statuses == expected.mu3_statuses == (desired,)
        state_equal(actual.state, expected.state)
        np.testing.assert_array_equal(actual.mu3_ess, expected.mu3_ess)
        np.testing.assert_array_equal(actual.mu3_log_normalizers, expected.mu3_log_normalizers)
        np.testing.assert_array_equal(key_bits(colony.mu3_key_cursor(iid)), key_bits(cursor))


@pytest.mark.slow
def test_float32_large_domain_matches_engine():
    with jax.enable_x64(False):
        model = _model()
        model.variables["z0"] = replace(model.variables["z0"], cardinality=40)
        model.variables["x0"] = replace(model.variables["x0"], cardinality=20)
        arrays = {"Ac": np.ones((20, 40), np.float32), "d": np.ones(40, np.float32) / 40,
                  "c0": np.ones(20, np.float32) / 20}
        key = jax.random.PRNGKey(27)
        colony = Colony()
        iid = colony.spawn(model, arrays, mu3_key=key)
        engine = Engine(compile_template(model, arrays=arrays))
        initial, cursor = engine.prepare_mu3(engine.init_state(), key)
        plan = engine.arena.mu3_groups[0]
        particles = initial.params[plan.name + ":samples"]
        assert particles.dtype == np.float32 and particles.shape == (256, 20, 40)
        actual = colony.run(1)[iid]
        expected = engine.run(initial, 1)
        state_equal(actual.state, expected.state)
        np.testing.assert_array_equal(key_bits(cursor), key_bits(colony.mu3_key_cursor(iid)))
        assert actual.status == expected.status
        assert actual.mu3_statuses == expected.mu3_statuses
        assert_diagnostics(colony.mu3_diagnostics(iid), engine.mu3_diagnostics(expected.state), np.float32)


@pytest.mark.slow
def test_key_state_and_result_associations_at_32_and_128_members():
    colony = Colony()
    model, arrays = _model(), _arrays()
    cursors = {}
    particles = {}

    def spawn():
        key = jax.random.key(len(cursors))
        iid = colony.spawn(model, arrays, mu3_key=key)
        cursors[iid] = key_bits(jax.random.split(key)[0])
        engine = colony.engines[colony.instances[iid].key]
        samples = engine.arena.mu3_groups[0].name + ":samples"
        particles[iid] = (samples, colony.instances[iid].state.params[samples])
        return iid

    for size in (32, 128):
        while len(colony) < size:
            spawn()
        result = colony.run(1)
        assert set(result) == set(colony.instances)
        for iid, item in result.items():
            assert item.state is colony.instances[iid].state
            assert colony.last_results[iid] is item
            np.testing.assert_array_equal(key_bits(colony.mu3_key_cursor(iid)), cursors[iid])
            samples, expected = particles[iid]
            np.testing.assert_array_equal(item.state.params[samples], expected)
            group, = colony.mu3_diagnostics(iid).values()
            assert group["ess"] == item.mu3_ess[0]
            assert group["log_normalizer"] == item.mu3_log_normalizers[0]
        survivor = next(iter(result))
        keep = result[survivor]
        removed = max(result)
        colony.kill(removed)
        fresh = spawn()
        assert fresh > removed and fresh not in colony.last_results
        np.testing.assert_array_equal(
            key_bits(colony.mu3_key_cursor(fresh)), cursors[fresh]
        )
        assert not np.array_equal(particles[fresh][1], particles[removed][1])
        assert colony.last_results[survivor] is keep and colony.instances[survivor].state is keep.state
        np.testing.assert_array_equal(key_bits(colony.mu3_key_cursor(survivor)), cursors[survivor])

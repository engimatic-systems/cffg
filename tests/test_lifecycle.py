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


def tile_observations(count=99):
    from cffg.core import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec

    t = Template(name="binary_tile_observations")
    arrays = {"probabilities": np.array([.4, .6])}
    t.blobs["probabilities"] = BlobRef("probabilities", "tiles.npy")
    for k in range(count):
        port = f"tile{k}"
        t.variables[port] = VariableSpec(port, 2)
        t.ports[port] = PortSpec(port, port, PortRole.DATA)
        t.factors[port] = FactorSpec(
            port, "categorical", {"out": port}, param_blobs={"statistic": "probabilities"}
        )
    return t, arrays


@pytest.mark.parametrize("enabled", [False, True])
def test_99_binary_tiles_batch_and_partial_updates_match_independent_rows(enabled):
    with jax.enable_x64(enabled):
        template, arrays = tile_observations()
        engine = Engine(compile_template(template, arrays=arrays))
        state = engine.init_state()
        ports = list(reversed(template.ports))
        slots = [engine.arena.data_slots[p] for p in ports]
        indices = engine.arena.message_indices(np.array(slots), 2)
        observations = dict(zip(ports, ([0, np.int32(1), None] * 33), strict=True))
        observed = engine.bind_data(state, observations)
        expected = np.asarray(state.messages).copy()
        for i, v in enumerate(observations.values()):
            row = np.full(2, -np.log(2) if v is None else NEG)
            if v is not None:
                row[v] = 0.
            expected[indices[i]] = row
        np.testing.assert_array_equal(observed.messages, expected)

        # A pair of booleans previously used NumPy's scalar boolean indexing
        # (all entries or none). The public integer labels 0 and 1 remain one-hot.
        changed = engine.bind_data(observed, {ports[0]: True, ports[1]: np.bool_(False),
                                             ports[2]: 1, ports[3]: None})
        expected[indices[:2]] = [[0., 0.], [NEG, NEG]]
        expected[indices[2]] = [NEG, 0.]
        expected[indices[3]] = -np.log(2)
        np.testing.assert_array_equal(changed.messages, expected)
        assert observed.messages.dtype == changed.messages.dtype == state.messages.dtype


def mixed_observation_model():
    from test_ef_engine import gamma_poisson

    from cffg.core.compose import merge, scoped
    from cffg.models.kalman import lgssm

    template, arrays = tile_observations(4)
    for prefix, (other, other_arrays) in (
        ("counts", gamma_poisson(2)),
        ("scalar", lgssm(2, np.eye(2), np.eye(2), np.ones((1, 2)),
                         np.eye(1), np.zeros(2), np.eye(2))),
        ("vector", lgssm(2, np.eye(3), np.eye(3), np.eye(3), np.eye(3),
                         np.zeros(3), np.eye(3))),
    ):
        template = merge(template, scoped(other, prefix), name="mixed_data")
        arrays.update({f"{prefix}/{name}": value for name, value in other_arrays.items()})
    return template, arrays


@pytest.mark.parametrize("enabled", [False, True])
def test_mixed_grouped_bind_matches_scalar_binds_and_preserves_masked_caches(enabled):
    with jax.enable_x64(enabled):
        template, arrays = mixed_observation_model()
        engine = Engine(compile_template(template, arrays=arrays))
        base = engine.init_state()
        full = {"tile1": 1, "counts/count1": 7, "vector/y0": [[3., 4., 6.]],
                "scalar/y1": -2.5, "tile3": None, "counts/count0": 2,
                "scalar/y0": [9.], "vector/y1": [-1., -2., -3.]}
        # Preserve the contribution of omitted ports and the cached values of
        # masked ports. Each row deliberately uses a different value or domain.
        partial = {"tile1": None, "counts/count0": None, "scalar/y0": None,
                   "vector/y0": [-5., 8., .5], "tile2": 0, "counts/count1": 10}

        scalar = base
        grouped = base
        for batch in (full, partial, {}):
            for name, v in batch.items():
                scalar = engine.bind_data(scalar, {name: v})
            grouped = engine.bind_data(grouped, batch)
            assert_tree_equal(grouped, scalar)
        for port, expected, mask in (("scalar/y1", [-2.5], 1.), ("scalar/y0", [9.], 0.),
                                     ("vector/y1", [-1., -2., -3.], 1.),
                                     ("vector/y0", [-5., 8., .5], 1.)):
            row, dim = engine.arena.gauss_data[port]
            idx = engine.arena.g_obs_indices(row, dim)
            np.testing.assert_array_equal(np.asarray(grouped.g_obs)[idx], expected)
            assert np.asarray(grouped.g_mask)[row] == mask
        for port, expected, mask in (("counts/count0", 2., 0.), ("counts/count1", 10., 1.)):
            row, _ = engine.arena.ef_data[port]
            assert np.asarray(grouped.ef_obs)[row] == expected
            assert np.asarray(grouped.ef_mask)[row] == mask
        # The public host binder also composes with a traced state and static observations.
        compiled_bind = jax.jit(lambda s: engine.bind_data(s, partial))
        assert_tree_equal(grouped, compiled_bind(engine.bind_data(base, full)))


def test_colony_mixed_partial_bind_validates_entire_batch_before_device_writes(monkeypatch):
    from cffg import lifecycle
    from cffg.colony import Colony

    template, arrays = mixed_observation_model()
    colony = Colony()
    first = colony.spawn(template, arrays)
    other = colony.spawn(template, arrays)
    other_before = colony.instances[other].state
    colony.bind(first, {"vector/y0": [1., 2., 3.], "counts/count0": 5, "tile0": 1})
    colony.bind(first, {"counts/count0": None, "tile0": None})
    state = colony.instances[first].state
    arena = colony.engines[colony.instances[first].key].arena
    row, _ = arena.ef_data["counts/count0"]
    assert np.asarray(state.ef_mask)[row] == 0 and np.asarray(state.ef_obs)[row] == 5
    gaussian, dim = arena.gauss_data["vector/y0"]
    np.testing.assert_array_equal(np.asarray(state.g_obs)[arena.g_obs_indices(gaussian, dim)],
                                  [1., 2., 3.])
    assert colony.instances[other].state is other_before
    assert not np.asarray(other_before.g_mask).any()

    def premature_write(*_):
        pytest.fail("invalid partial update reached the device")

    monkeypatch.setattr(lifecycle, "_scatter_observations", premature_write)
    for name, value, exception, text in (
        ("bad-port", 0, KeyError, "no DATA port"),
        ("tile0", 2, ValueError, "out of range"),
        ("counts/count1", -.5, ValueError, "port 'counts/count1'"),
        ("scalar/y1", [1., 2.], ValueError, "observation has dim 2"),
    ):
        with pytest.raises(exception, match=text):
            colony.bind(first, {"tile1": 1, "vector/y0": None, name: value})
        assert colony.instances[first].state is state


def test_partial_bind_reuses_compilation_when_small_batch_size_and_missingness_change():
    from cffg.models.kalman import lgssm

    eye = np.eye(2)
    t, arrays = lgssm(20, eye, eye, eye, eye, np.zeros(2), eye)
    engine = Engine(compile_template(t, arrays=arrays))
    bound = engine.bind_data(engine.init_state(), {"y0": [2., -3.]})
    with jax.no_tracing():
        rebound = engine.bind_data(bound, {"y7": [0., 0.], "y0": None, "y18": [1., 4.]})
    for port, expected, mask in (("y0", [2., -3.], 0), ("y18", [1., 4.], 1),
                                 ("y7", [0., 0.], 1)):
        row, dim = engine.arena.gauss_data[port]
        idx = engine.arena.g_obs_indices(row, dim)
        np.testing.assert_array_equal(np.asarray(rebound.g_obs)[idx], expected)
        assert np.asarray(rebound.g_mask)[row] == mask

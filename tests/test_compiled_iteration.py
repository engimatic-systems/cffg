"""Compare compiled stopping with the pre-existing host iteration controller."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_compiler_contracts import model
from test_engine_contracts import assert_state_equal, record_compiled_batches
from test_estimate import _hostile_heterogeneous_engine
from test_schedule import unit_categorical
from test_schedule_program import program_arena, state_in_dtype

from cffg import Engine, FailureDiagnostic, RunStatus, compile_template
from cffg.colony import Colony
from cffg.compiled import NEG
from cffg.estimate import estimate
from cffg.iteration import run_iterations
from cffg.run_results import finalize_iteration


def host(engine, state, n_iters, *, tol=None, damp=0.0, harden_last=0):
    services = replace(engine._iteration_services(), free_energy=lambda s: float(engine.bfe(s)))
    return finalize_iteration(run_iterations(
        state, engine._iteration_program(), services=services,
        policy=engine._iteration_policy(state, tol), n_iters=n_iters, damp=damp,
        harden_last=harden_last,
    ), factor_names=engine._failure_names)


@pytest.mark.parametrize("kind", ["flood", "colored", "mixed", "gaussian", "ef", "recurrence"])
def test_scheduled_traces_match_host(kind):
    if kind in ("flood", "colored", "mixed"):
        engine = Engine(program_arena(kind), schedule="flood" if kind == "flood" else "auto")
    else:
        template, arrays = model(kind)
        engine = Engine(compile_template(template, arrays=arrays), schedule="flood")
    state = engine.init_state()
    if kind == "mixed":
        state = engine.bind_data(state, {"px": 1})
    for tol in [None, 1e-9]:
        reference = host(engine, state, 8, tol=tol)
        result = engine.run(state, 8, tol=tol)
        assert (result.status, result.iterations, result.failure) == (
            reference.status, reference.iterations, reference.failure
        )
        for name in ("messages", "g_eta", "g_lam", "ef_eta"):
            actual, expected = getattr(result.state, name), getattr(reference.state, name)
            if kind == "flood":
                np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        if kind == "flood":
            np.testing.assert_array_equal(result.message_deltas, reference.message_deltas)
        else:
            np.testing.assert_allclose(result.message_deltas, reference.message_deltas,
                                       rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result.free_energies, reference.free_energies, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("x64", [False, True])
def test_float32_tolerance_between_representable_deltas(x64):
    with jax.enable_x64(x64):
        engine = Engine(program_arena("flood"), schedule="flood")
        state = state_in_dtype(engine, np.float32)

        def increment(m, e, l, obs, mask, f, fo, fm, params, damp, harden):
            return m.at[0].add(.25), e, l, f, 0., 0

        engine._iter_step = increment
        engine.bfe = lambda s: 0.
        limit = float(np.nextafter(.25, 0.))
        assert host(engine, state, 2, tol=limit).status is RunStatus.BUDGET_EXHAUSTED
        assert engine.run(state, 2, tol=limit).status is RunStatus.BUDGET_EXHAUSTED
        assert engine.run(state, 2, tol=.25).status is RunStatus.CONVERGED


def test_float32_residual_preserves_difference_above_dtype_range():
    with jax.enable_x64(False):
        engine = Engine(program_arena("flood"), schedule="flood")
        state = state_in_dtype(engine, np.float32)

        def increment(m, e, l, obs, mask, f, fo, fm, params, damp, harden):
            return m.at[0].add(1), e, l, f, 0., 0

        origin = state.messages[0]
        engine._iter_step = increment
        engine.bfe = lambda s: jnp.where(s.messages[0] == origin + 1, -3e38, 3e38)
        ref = host(engine, state, 2)
        assert np.isfinite(ref.residual) and ref.residual > float(np.finfo(np.float32).max)
        assert engine.run(state, 2).residual == ref.residual


def test_public_colony_completes_600_sweeps_and_freezes_rejected_members(monkeypatch):
    width, budget, late_sweep = 320, 600, 300
    damp = np.float32(.995)
    left = np.arange(width, dtype=np.float32) / np.float32(2000) + np.float32(.125)
    priors = np.column_stack((left, np.full_like(left, .375), .625 - left))
    template = unit_categorical()

    with jax.enable_x64(False):
        colony = Colony()
        ids = [colony.spawn(template, {"p": prior}) for prior in priors]
        (key,) = {instance.key for instance in colony.instances.values()}
        engine = colony.engines[key]
        assert not engine._can_run_exact(budget, damp, 0)
        arena = engine.arena
        out_slot = arena.groups[0].scatter[0, 0]
        outward = arena.message_indices(out_slot, priors.shape[1])
        inbound = arena.message_indices(arena.data_slots["data"], priors.shape[1])
        target = np.log(float(priors[2, 0]))
        trigger = np.float32(target + (-np.log(3.) - target) * float(damp)**(late_sweep - .5))
        original_step = engine._iter_step
        traces = []

        def watch_step(messages, eta, lam, obs, mask, ef_eta, ef_obs, ef_mask,
                       params, damping, harden):
            traces.append(isinstance(messages, jax.core.Tracer))
            updated, *aux = original_step(
                messages, eta, lam, obs, mask, ef_eta, ef_obs, ef_mask,
                params, damping, harden,
            )
            fails = params["test:late_failure"] & (updated[outward[0]] < trigger)
            marked = updated.at[outward[-1]].set(jnp.where(fails, jnp.nan, updated[outward[-1]]))
            return marked, *aux

        monkeypatch.setattr(engine, "_iter_step", watch_step)
        for index, iid in enumerate(ids):
            instance = colony.instances[iid]
            state = instance.state
            messages = state.messages
            if index == 0:
                messages = jnp.full_like(messages, NEG)
            if index == 1:
                messages = messages.at[0].set(jnp.nan)
            instance.state = replace(
                state, messages=messages,
                params={**state.params, "test:late_failure": jnp.asarray(index == 2)},
            )
        before = [colony.instances[iid].state for iid in ids]
        batches = record_compiled_batches(monkeypatch)
        result_by_id = colony.run(budget, damp=damp, tol=None)

    assert [(len(group), sweeps) for group, sweeps in batches] == [(width, budget)]
    assert traces == [True]
    assert tuple(result_by_id) == tuple(ids)
    results = list(result_by_id.values())
    assert [r.status for r in results[:2]] == [RunStatus.INFEASIBLE, RunStatus.DIVERGED]
    assert results[0].failure == FailureDiagnostic("empty_support", None, "initialization", 0)
    assert results[1].failure == FailureDiagnostic("nonfinite_state", None, "initialization", 0)
    for result, initial in zip(results[:2], before[:2], strict=True):
        assert result.iterations == 0
        assert result.message_deltas == ()
        assert_state_equal(result.state, initial)
    late = results[2]
    assert late.status is RunStatus.DIVERGED
    assert late.iterations == len(late.message_deltas) == late_sweep
    assert late.failure == FailureDiagnostic("nonfinite_update", None, "soft", late_sweep)
    assert len(late.free_energies) == late_sweep - 1
    assert np.isnan(late.message_deltas[-1])
    assert np.isnan(np.asarray(late.state.messages)[outward[-1]])

    healthy = results[3:]
    assert all(r.status is RunStatus.BUDGET_EXHAUSTED for r in healthy)
    assert all(
        r.iterations == len(r.message_deltas) == len(r.free_energies) == budget
        for r in healthy
    )
    assert np.isfinite(np.asarray([r.free_energies for r in healthy])).all()
    assert all(r.message_deltas[-1] > 0 for r in healthy)

    # Each real sweep damps the outward log message towards its own log(p).
    actual = np.stack([r.state.messages for r in healthy])
    initialized = np.stack([state.messages for state in before[3:]])
    weight = float(damp) ** budget
    target_log = np.log(priors[3:].astype(np.float64))
    distance = target_log - initialized[:, outward]
    expected = initialized[:, outward] + (1 - weight) * distance
    # Float32 multiply/add roundoff accumulates with the damping weights.
    eps = float(np.finfo(np.float32).eps)
    bound = eps * (np.max(np.abs(target_log)) / (1 - float(damp)) + 4)
    assert np.max(np.abs(distance)) * weight * (1 / float(damp) - 1) > 4 * bound
    np.testing.assert_allclose(actual[:, outward], expected, rtol=0, atol=bound)
    np.testing.assert_array_equal(actual[:, inbound], initialized[:, inbound])
    late_slots = outward[:-1]
    late_start = np.asarray(before[2].messages)[late_slots]
    late_target = np.log(priors[2, :-1].astype(np.float64))
    np.testing.assert_allclose(
        np.asarray(late.state.messages)[late_slots],
        late_start + (1 - float(damp)**late_sweep) * (late_target - late_start),
        rtol=0, atol=bound,
    )
    for iid, result, initial in zip(ids, results, before, strict=True):
        assert colony.instances[iid].state is result.state
        assert colony.last_results[iid] is result
        for name, value in initial.params.items():
            np.testing.assert_array_equal(result.state.params[name], value)


def test_ordinary_iteration_keeps_exact_heterogeneous_arena_bytes():
    low_width, high_card, high_dim = 64, 32, 32
    with jax.enable_x64(False):
        engine = _hostile_heterogeneous_engine(low_width, high_card, high_dim)
        initial = engine.init_state()
        result = engine.run(initial, 3, damp=.25)
        modeled_bytes = estimate(engine, calibrate=False).arena_bytes

    assert engine.arena.edges["disc_outlier"].card == high_card
    assert engine.arena.edges["gauss_outlier"].dim == high_dim
    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert result.iterations == len(result.message_deltas) == len(result.free_energies) == 3
    assert np.isfinite(result.free_energies).all()
    # Two directional messages per variable. Every family uses its own
    # segment widths, including its squared dimension for precision.
    item_size = np.dtype(np.float32).itemsize
    expected_bytes = {
        "messages": 2 * (low_width * 2 + high_card) * item_size,
        "g_eta": 2 * (low_width + high_dim) * item_size,
        "g_lam": 2 * (low_width + high_dim**2) * item_size,
        "g_obs": 0, "g_mask": 0, "ef_eta": 0, "ef_obs": 0, "ef_mask": 0,
    }
    assert modeled_bytes == expected_bytes
    assert sum(expected_bytes.values()) == 10_752
    for state in (initial, result.state):
        assert {field: getattr(state, field).nbytes for field in expected_bytes} == expected_bytes
        assert all(np.isfinite(value).all() for value in jax.tree.leaves(state))

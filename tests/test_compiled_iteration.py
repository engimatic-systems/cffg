"""Compare compiled stopping with the pre-existing host iteration controller."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_compiler_contracts import model
from test_schedule_program import program_arena, state_in_dtype

from cffg import Engine, RunStatus, compile_template
from cffg.compiled import NEG
from cffg.compiled_iteration import controller_for
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


def test_scaled_batch_freezes_invalid_members_and_reuses_parameters():
    engine = Engine(program_arena("flood"), schedule="flood")
    state = state_in_dtype(engine, np.float32)
    empty = replace(state, messages=jnp.full_like(state.messages, NEG))
    poison = replace(state, messages=state.messages.at[0].set(jnp.nan))
    states = [empty, poison, *([state] * 318)]
    reference = host(engine, state, 6, tol=1e-10)
    results = [finalize_iteration(p) for p in controller_for(engine).run_batch(
        states, n_iters=6, harden_last=0, damp=0., tol=1e-10,
    )]
    assert [r.status for r in results[:2]] == [RunStatus.INFEASIBLE, RunStatus.DIVERGED]
    assert all(r.iterations == 0 for r in results[:2])
    for result in results[2:]:
        assert (result.iterations, result.status) == (reference.iterations, reference.status)
        np.testing.assert_allclose(result.message_deltas, reference.message_deltas, atol=2e-6)
        np.testing.assert_allclose(result.free_energies, reference.free_energies, atol=2e-6)

"""Terminal outer evidence is distinct from recoverable numerical proposals."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from nonlinear_fixture import model
from test_failure_diagnostics import fact
from test_run_strategies import engine_for

from cffg import Engine, FailureDiagnostic, RunStatus, compile_template
from cffg import outer, outer_strategy
from cffg.failure_facts import FailurePhase, empty_failure
from cffg.outer import OuterArrays, OuterKernelPlan, OuterSolveResult
from cffg.outer_evidence import terminal_failure
from cffg.runtime_types import OuterTerminationReason as Reason


@pytest.fixture
def scalar_chart(monkeypatch):
    """Isolate controller decisions from chart geometry, tested separately."""
    chart = SimpleNamespace(
        size=1, proper_reference=True,
        support_changed=lambda _: False,
        residual_scale=jnp.ones_like,
    )
    monkeypatch.setattr(outer, "build_outer_chart", lambda *args: chart)
    monkeypatch.setattr(outer, "chart_coordinates", lambda _, arrays: arrays.messages)
    monkeypatch.setattr(outer, "chart_arrays", lambda _, values: scalar_arrays(values))
    return chart


def scalar_arrays(values):
    return OuterArrays(jnp.atleast_1d(values), *(jnp.zeros(0) for _ in range(3)))


def plan(fixed=None, finite=True):
    def trust(coords, *args):
        empty = jnp.empty((0, coords.size), coords.dtype)
        fixed_steps = empty if fixed is None else jnp.full((1, coords.size), fixed)
        return (jnp.ones_like(coords), 1., 1., True, False, finite,
                empty, jnp.zeros(0), fixed_steps,
                jnp.ones(len(fixed_steps)), 0)
    return OuterKernelPlan(trust, lambda *args: (0., 0.), None, False)


def solve(mapping, *, n_iters=30, candidate_limit=0, fixed=None, finite=True,
          initial_failure=None):
    return outer.solve_outer(
        scalar_arrays(jnp.array([0.], jnp.float32)), SimpleNamespace(edges={}),
        mapping, lambda _: 0., lambda _: False,
        n_iters=n_iters, tol=1e-6, candidate_limit=candidate_limit,
        kernel_factory=lambda _: plan(fixed, finite),
        initial_failure=initial_failure,
    )


def mapped(arrays, value, failed=False):
    return (arrays._replace(messages=jnp.full_like(arrays.messages, value)),
            jnp.asarray(0.), jnp.asarray(2 if failed else 0),
            fact() if failed else empty_failure())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("path", ["candidate", "trust", "auxiliary"])
def test_success_returns_the_certified_point_not_its_expansive_successor(scalar_chart, dtype, compiled, path):
    tolerance = max(1e-6, 64 * np.finfo(dtype).eps)
    center = .5 if path == "auxiliary" else 0.
    delta = tolerance * (1 + center) / 2

    def mapping(arrays):
        x = arrays.messages
        value = center + delta + 16 * (x - center)
        if path == "auxiliary":
            value = jnp.where((x > .25) & (x < .75), value, x + 1)
        return arrays._replace(messages=value), jnp.asarray(0.), jnp.asarray(0), empty_failure()

    update = jax.jit(mapping) if compiled else mapping
    energy = lambda arrays: float(arrays.messages[0])
    result = outer.solve_outer(
        scalar_arrays(jnp.array([0.], dtype=dtype)), SimpleNamespace(edges={}),
        update, energy, lambda _: False,
        n_iters=30, tol=1e-6, candidate_limit=128 if path == "candidate" else 0,
        kernel_factory=lambda _: plan(),
    )
    assert result.reason is Reason.ROOT_CONVERGED
    np.testing.assert_array_equal(result.arrays.messages, np.array([center], dtype=dtype))
    assert result.iterations == (1 if path == "auxiliary" else 0)
    assert result.accepted_steps == (1 if path == "auxiliary" else 0)
    assert result.rejected_steps == (1 if path == "auxiliary" else 0)
    assert int(result.failure.fact.reason) == 0 and result.inner_code == 0
    assert result.free_energies[-1] == energy(result.arrays)
    successor = update(result.arrays)[0]
    residual = float(jnp.max(jnp.abs(successor.messages - result.arrays.messages) / (1 + jnp.abs(result.arrays.messages))))
    assert residual == pytest.approx(result.outer_residual)
    assert residual <= tolerance
    assert abs(energy(successor) - energy(result.arrays)) <= tolerance * (1 + abs(energy(result.arrays)))
    next_successor = update(successor)[0]
    successor_residual = float(jnp.max(jnp.abs(next_successor.messages - successor.messages) / (1 + jnp.abs(successor.messages))))
    assert successor_residual > tolerance


@pytest.mark.slow
def test_float32_reactor_returned_state_satisfies_its_frozen_chart_certificate(monkeypatch):
    from cstr_planning_fixture import planning_fixture
    from cffg.outer_chart import chart_coordinates
    from cffg.runtime_types import _replace

    charts = []
    original = outer.build_outer_chart

    def capture(*args, **kwargs):
        chart = original(*args, **kwargs)
        charts.append(chart)
        return chart

    monkeypatch.setattr(outer, "build_outer_chart", capture)
    with jax.enable_x64(False):
        template, arrays, data = planning_fixture("sigma_point", past=0, horizon=2, dtype=np.float32)
        arrays["P0"] = np.diag(np.array([.01, .001, 1e-4, .1], dtype=np.float32))
        engine = Engine(compile_template(template, arrays=arrays), schedule="flood")
        result = engine.run(engine.bind_data(engine.init_state(), data), 60, tol=1e-6)
        state = result.require_converged()
        values = engine._iter_step(
            state.messages, state.g_eta, state.g_lam, state.g_obs, state.g_mask,
            state.ef_eta, state.ef_obs, state.ef_mask, state.params, 0., False,
        )
        assert int(values[6].reason) == 0
        mapped = _replace(state, messages=values[0], g_eta=values[1], g_lam=values[2], ef_eta=values[3])
        before = OuterArrays(state.messages, state.g_eta, state.g_lam, state.ef_eta)
        after = OuterArrays(mapped.messages, mapped.g_eta, mapped.g_lam, mapped.ef_eta)
        u, v = np.asarray(chart_coordinates(charts[-1], before)), np.asarray(chart_coordinates(charts[-1], after))
        tolerance = 64 * np.finfo(np.float32).eps
        residual = np.max(np.abs(v - u) / (1 + np.abs(u)))
        assert result.outer_residual <= tolerance and residual <= tolerance
        energy, mapped_energy = float(engine.bfe(state)), float(engine.bfe(mapped))
        assert np.isfinite([energy, mapped_energy]).all()
        assert abs(mapped_energy - energy) <= tolerance * (1 + abs(energy))
        assert result.free_energies[-1] == energy
        assert result.failure is None


@pytest.mark.parametrize("stage,attempt", [("discovery", 1), ("candidate", 2), ("trust", 1)])
def test_terminal_current_maps_retain_logical_attempt(scalar_chart, stage, attempt):
    calls = []

    def mapping(arrays):
        calls.append(arrays)
        failed = (stage == "discovery" or
                  stage == "candidate" and len(calls) == 2 or
                  stage == "trust" and len(calls) == 3)
        return mapped(arrays, jnp.nan if failed else 1., failed)

    result = solve(mapping, candidate_limit=128 if stage == "candidate" else 0)
    assert result.reason is Reason.INNER_FAILURE
    assert int(result.failure.fact.reason) == int(fact().reason)
    assert int(result.failure.fact.phase) == FailurePhase.OUTER
    assert int(result.failure.iteration) == attempt
    assert len(calls) != attempt if stage == "trust" else len(calls) == attempt


@pytest.mark.parametrize("fixed", [None, -1.])
def test_collapse_names_a_trial_only_if_every_trial_is_invalid(scalar_chart, fixed):
    def mapping(arrays):
        x = float(arrays.messages[0])
        return mapped(arrays, jnp.nan if x > 0 else 1., x > 0)

    result = solve(mapping, fixed=fixed)
    assert result.reason is Reason.TRUST_REGION_COLLAPSED
    assert result.rejected_steps > 1
    if fixed is None:
        assert int(result.failure.fact.reason) == int(fact().reason)
        assert int(result.failure.iteration) == result.iterations
    else:
        assert int(result.failure.fact.reason) == 0


def test_invalid_trial_recovers_and_budget_does_not_borrow_trial_cause(scalar_chart):
    def mapping(arrays):
        x = float(arrays.messages[0])
        return mapped(arrays, jnp.nan if x >= 1 else .5, x >= 1)

    result = solve(mapping, fixed=.5)
    assert result.reason is Reason.ROOT_CONVERGED
    assert result.trial_evaluations >= 2
    assert int(result.failure.fact.reason) == 0
    exhausted = solve(mapping, n_iters=1)
    assert exhausted.reason is Reason.ITERATION_BUDGET
    assert exhausted.rejected_steps == 1
    assert int(exhausted.failure.fact.reason) == 0


def test_nonfinite_derivative_kernel_does_not_invent_local_ownership(scalar_chart):
    result = solve(lambda arrays: mapped(arrays, 1.), finite=False)
    assert result.reason is Reason.NONFINITE_EVALUATION
    assert int(result.failure.fact.reason) == 0


@pytest.mark.parametrize("schedule", ["auto", "flood"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_real_improper_cavity_keeps_origin(schedule, dtype):
    with jax.enable_x64(dtype == jnp.float64):
        template, arrays = model()
        engine = Engine(compile_template(template, arrays=arrays), schedule=schedule)
        state = engine.step(engine.init_state())
        group = next(g for g in engine.arena.gauss_groups if g.kernel.binding.iterative)
        index = engine.arena.g_lam_indices(group.gather[:, 1], 1)
        state = replace(state, g_lam=state.g_lam.at[index].set(-1.))
        result = engine.run(state, 5, tol=1e-5)
        # Invalid input is attributed before exact neighbors can overwrite it.
        assert result.status is RunStatus.DIVERGED
        assert result.failure == FailureDiagnostic("improper_cavity", "sensor", "startup", 1)


@pytest.mark.parametrize("recover", [False, True])
def test_controller_offsets_restarts_without_retaining_rejected_solve(monkeypatch, recover):
    engine = engine_for("gfe")
    calls = []

    def solve(initial, *args, **kwargs):
        calls.append(kwargs["n_iters"])
        first = len(calls) == 1
        reason = Reason.TRUST_REGION_COLLAPSED if first else (
            Reason.ROOT_CONVERGED if recover else Reason.INNER_FAILURE)
        return OuterSolveResult(
            initial, reason, 2 if first else 0, (0.,), (2., 2.),
            0., 0., 0., 1., 0, 1, 0., 0 if first or recover else 2,
            failure=terminal_failure(fact(rank=0 if first else 1), 1),
        )

    monkeypatch.setattr(outer_strategy, "solve_outer", solve)
    result = engine.run(engine.init_state(), 3, tol=1e-6, harden_last=0)
    assert calls == [3, 1]
    if recover:
        assert result.status is RunStatus.CONVERGED
        assert result.failure is None
    else:
        assert result.failure == FailureDiagnostic(
            "improper_cavity", engine._failure_names[1], "outer", 3)


def test_priming_precedes_immediate_functional_failure(monkeypatch):
    engine = engine_for("mu3")
    state, _ = engine.prepare_mu3(engine.init_state(), jax.random.key(7))
    original = engine._prime
    assert original is not None

    def prime(*args):
        values = original(*args)
        return (*values[:4], fact())

    monkeypatch.setattr(engine, "_prime", prime)
    monkeypatch.setattr(engine, "_update_mu3", lambda state: (state, 0., 2, jnp.inf))
    result = engine.run(state, 1, tol=1e-6)
    assert result.status is RunStatus.DIVERGED
    assert result.failure == FailureDiagnostic(
        "improper_cavity", engine._failure_names[0], "priming", 0)


@pytest.mark.parametrize("recovered", [False, True])
def test_discovery_advancement_expires_priming_cause(scalar_chart, recovered):
    calls = []
    scalar_chart.proper_reference = not recovered

    def mapping(arrays):
        calls.append(arrays)
        failed = not recovered or len(calls) > 1
        return mapped(arrays, jnp.nan if failed else 1., failed)

    prefix = terminal_failure(fact(rank=5, phase=FailurePhase.PRIMING), 0)
    result = solve(mapping, initial_failure=prefix)
    assert result.reason is Reason.INNER_FAILURE
    assert result.iterations == 0
    assert int(result.failure.fact.factor) == (0 if recovered else 5)
    assert int(result.failure.iteration) == (1 if recovered else 0)


def test_support_recursion_offsets_child_terminal_attempt(scalar_chart):
    scalar_chart.support_changed = lambda values: float(values[0]) == .5
    calls = []

    def mapping(arrays):
        calls.append(arrays)
        if len(calls) == 1:
            return mapped(arrays, 1.)
        if len(calls) == 2:
            return mapped(arrays, .5)
        return mapped(arrays, jnp.nan, True)

    result = solve(mapping, candidate_limit=128)
    assert result.reason is Reason.INNER_FAILURE
    assert result.iterations == 1
    assert int(result.failure.iteration) == 2


def test_reconstruction_origin_precedes_downstream_nan(monkeypatch):
    engine = engine_for("gfe")
    rank = len(engine._failure_names) - 1

    def reconstruct(*args):
        return (args[0].at[0].set(jnp.nan), *args[1:4], fact(rank=rank))

    monkeypatch.setattr(engine, "_reconstruct", reconstruct)
    result = engine.run(engine.init_state(), 3, tol=1e-6)
    assert result.status is RunStatus.DIVERGED
    assert result.failure == FailureDiagnostic(
        "improper_cavity", engine._failure_names[rank], "reconstruction", 1)

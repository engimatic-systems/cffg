"""Distinct exact/outer evidence, dynamic kernel inputs and facade delegation."""

import ast
import inspect
from dataclasses import FrozenInstanceError, fields, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_compiler_contracts import model

from cffg import exact_strategy, outer_strategy, run_results
from cffg.compiler import compile_template
from cffg.engine import Engine, RunStatus, _replace
from cffg.exact_strategy import ExactOutcome, ExactReason, ExactRunStrategy
from cffg.outer import (
    OuterArrays,
    OuterSolveResult,
    build_outer_chart,
    chart_arrays,
    chart_coordinates,
    outer_chart_signature,
)
from cffg.outer_kernels import OuterKernelProvider
from cffg.outer_outcome import OuterOutcome
from cffg.outer_structure import OuterStructure
from cffg.runtime_types import OuterTerminationReason as Reason


def engine_for(name):
    template, arrays = model(name)
    return Engine(compile_template(template, arrays=arrays), mu3_samples=16)


def forbidden(*args, **kwargs):
    raise AssertionError("This execution path must not call this service")


@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("pre,post,count", ((True, False, 0), (False, True, 1)))
def test_exact_support_counts_are_independent_of_trace_length(
    enabled, pre, post, count
):
    engine = engine_for("hmm")
    with jax.enable_x64(enabled):
        state = engine.init_state()

        @jax.jit
        def unit(messages, eta, lam, obs, mask, ef_eta, ef_obs, ef_mask, params):
            return messages + 1, eta, lam, ef_eta, 0.0, 0.0, -100.0, pre, post

        outcome = ExactRunStrategy(unit, "sum", lambda st: True, forbidden).run(
            state, None
        )
        result = run_results.finalize_exact(outcome)
    assert result.status is RunStatus.INFEASIBLE
    assert result.iterations == count
    assert result.message_deltas == ()
    assert result.free_energies == (float("inf"),)
    assert result.inner_status is RunStatus.CONVERGED
    if pre:
        assert result.state is state
    else:
        np.testing.assert_array_equal(result.state.messages, state.messages + 1)


@pytest.mark.parametrize("enabled", (False, True))
def test_exact_certificate_does_not_require_tolerance(enabled):
    engine = engine_for("hmm")
    with jax.enable_x64(enabled):
        state = engine.init_state()
        outcome = ExactRunStrategy(
            engine._exact_unit, "sum", engine._state_is_finite, forbidden
        ).run(state, None)
    assert outcome.structural
    assert not outcome.by_tolerance
    assert run_results.finalize_exact(outcome).status is RunStatus.CONVERGED
    with pytest.raises(ValueError, match="certificate"):
        run_results.finalize_exact(replace(outcome, structural=False))


@pytest.mark.parametrize("kind", ("exact", "outer"))
def test_strategy_traces_are_immutable_host_precision(kind):
    state = engine_for("hmm").init_state()
    precise = 1.0 + 2.0**-40
    incoming = np.array([precise])
    with jax.enable_x64(False):
        if kind == "exact":
            outcome = ExactOutcome(state, ExactReason.BUDGET, 1, incoming)
        else:
            outcome = OuterOutcome(
                state, Reason.ITERATION_BUDGET, message_deltas=incoming
            )
        incoming[:] = 9
        np.asarray(outcome.message_deltas)[:] = 7
    assert np.asarray(outcome.message_deltas).tolist() == [precise]
    with pytest.raises(FrozenInstanceError):
        outcome.residual = 2.0


@pytest.mark.parametrize("code", (0, 1, 2))
def test_zero_budget_outer_keeps_functional_update_before_budget(monkeypatch, code):
    engine = engine_for("mu3")
    state, _ = engine.prepare_mu3(engine.init_state(), jax.random.key(7))
    updated = _replace(state, messages=state.messages + 0.125)
    calls = []

    def update(current):
        assert current is state
        calls.append(current)
        return updated, 0.25, code, 0.125

    monkeypatch.setattr(engine, "_update_mu3", update)
    monkeypatch.setattr(engine, "_prime", forbidden)
    monkeypatch.setattr(engine, "_iter_step", forbidden)
    result = engine.run(state, 0)
    assert len(calls) == 1
    assert result.state is updated
    assert result.iterations == 0
    assert result.message_deltas == result.free_energies == ()
    assert result.termination_reason is (
        Reason.INNER_FAILURE if code else Reason.ITERATION_BUDGET
    )
    assert result.status is (
        RunStatus.DIVERGED if code == 2 else RunStatus.BUDGET_EXHAUSTED
    )
    assert result.residual == (0.125 if code else float("inf"))


@pytest.mark.parametrize("tol", (None, 1e-6))
def test_outer_preserves_accumulated_work_and_its_tolerance_meaning(monkeypatch, tol):
    engine = engine_for("gfe")
    calls = []

    def solve(initial, structure, *args, **kwargs):
        assert isinstance(structure, OuterStructure)
        calls.append(kwargs["n_iters"])
        return OuterSolveResult(
            initial,
            Reason.STAGNATION if len(calls) == 1 else Reason.ROOT_CONVERGED,
            0,
            (0.0,),
            (2.0, 2.0),
            0.0,
            0.0,
            0.0,
            1.0,
            3,
            4,
            0.0,
            0,
            5,
            6,
            7,
            1,
        )

    monkeypatch.setattr(outer_strategy, "solve_outer", solve)
    result = engine.run(engine.init_state(), 2, tol=tol, harden_last=0)
    assert calls == [2, 1]
    assert result.iterations == 2
    assert result.status is (
        RunStatus.CONVERGED if tol is not None else RunStatus.BUDGET_EXHAUSTED
    )
    assert result.termination_reason is Reason.ROOT_CONVERGED
    assert (result.accepted_steps, result.rejected_steps) == (6, 8)
    assert (result.outer_map_evaluations, result.trial_evaluations) == (10, 12)
    assert (result.linear_iterations, result.solver_compilations) == (14, 2)
    assert result.free_energies == (2.0,) * 4


@pytest.mark.parametrize("name", ("hmm", "gaussian", "parameter_edge", "gfe"))
def test_narrow_structure_reproduces_arena_chart_without_parameter_defaults(name):
    engine = engine_for(name)
    state = engine.init_state()
    arrays = OuterArrays(state.messages, state.g_eta, state.g_lam, state.ef_eta)
    structure = OuterStructure.from_arena(engine.arena)
    assert {field.name for field in fields(structure)} == {
        "edges",
        "goal_parameter_plans",
        "message_starts",
        "g_eta_starts",
        "g_lam_starts",
    }
    full = build_outer_chart(engine.arena, arrays, engine._outer_eliminated)
    narrow = build_outer_chart(structure, arrays, engine._outer_eliminated)
    assert outer_chart_signature(full) == outer_chart_signature(narrow)
    np.testing.assert_array_equal(
        chart_coordinates(full, arrays), chart_coordinates(narrow, arrays)
    )


def test_reused_kernel_context_keeps_current_base_observations_parameters_and_hard_values():
    engine = engine_for("hmm")
    state = engine.init_state()
    structure = OuterStructure.from_arena(engine.arena)
    arrays = OuterArrays(state.messages, state.g_eta, state.g_lam, state.ef_eta)
    edge = next(iter(structure.edges.values()))
    chart = build_outer_chart(structure, arrays, {"messages": frozenset({edge.slot0})})
    coordinates = chart_coordinates(chart, arrays)
    scale = jnp.ones_like(coordinates)
    provider = OuterKernelProvider()
    i0 = structure.message_indices(edge.slot0, edge.card)
    i1 = structure.message_indices(edge.slot1, edge.card)

    def step(
        messages, eta, lam, obs, mask, ef_eta, ef_obs, ef_mask, params, damp, harden
    ):
        weight = obs.sum() + 2 * mask.sum() + 3 * ef_obs.sum() + 4 * ef_mask.sum()
        # A gauge-invariant coordinate must actually see each runtime input.
        offset = (params["offset"] + weight + messages.sum()) * jnp.arange(
            messages.size
        )
        return messages * 0.5 + offset, eta, lam, ef_eta, 0.0, 0

    merits = []
    for value in (1.0, 2.0):
        current = _replace(
            state,
            params={"offset": jnp.asarray(value)},
            g_obs=jnp.asarray([value]),
            g_mask=jnp.asarray([value]),
            ef_obs=jnp.asarray([value]),
            ef_mask=jnp.asarray([value]),
        )
        current_chart = replace(
            chart, base=arrays._replace(messages=arrays.messages + value)
        )
        hot = jnp.arange(edge.card, dtype=state.messages.dtype) * value
        plan = provider.plan(
            current_chart, current, [(i0, i1, hot)], iter_step=step, reconstruct=None
        )
        assert plan.compiled is (value == 1.0)

        def merit(coords, *, current_chart=current_chart, current=current, hot=hot):
            base = chart_arrays(current_chart, coords)
            messages, eta, lam, ef_eta, _, _ = step(
                base.messages,
                base.g_eta,
                base.g_lam,
                current.g_obs,
                current.g_mask,
                base.ef_eta,
                current.ef_obs,
                current.ef_mask,
                current.params,
                0.0,
                False,
            )
            messages = messages.at[i0].set(hot).at[i1].set(hot)
            residual = (
                chart_coordinates(
                    current_chart, OuterArrays(messages, eta, lam, ef_eta)
                )
                - coords
            )
            return 0.5 * jnp.vdot(residual, residual).real

        actual, gradient = plan.root_diagnostics(coordinates, scale, plan.context)
        np.testing.assert_allclose(actual, merit(coordinates), rtol=1e-12)
        np.testing.assert_allclose(
            gradient, jnp.linalg.norm(jax.grad(merit)(coordinates)), rtol=1e-12
        )
        merits.append(float(actual))
    assert merits[0] != merits[1]
    assert len(provider.cache) == 1


@pytest.mark.parametrize("kind,name", (("exact", "hmm"), ("outer", "gfe")))
def test_facade_calls_the_shared_public_result_owner(monkeypatch, kind, name):
    engine = engine_for(name)
    seen = []
    original = getattr(run_results, "finalize_" + kind)

    def record(outcome, **kwargs):
        seen.append(outcome)
        return original(outcome, **kwargs)

    monkeypatch.setattr(run_results, "finalize_" + kind, record)
    engine.run(engine.init_state(), 1, tol=1e-6, harden_last=0)
    assert len(seen) == 1
    assert isinstance(seen[0], ExactOutcome if kind == "exact" else OuterOutcome)


@pytest.mark.parametrize("module", (exact_strategy, outer_strategy))
def test_strategy_owners_do_not_construct_public_results_or_import_engine(module):
    tree = ast.parse(inspect.getsource(module))
    imports = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not imports.intersection(
        {"cffg.engine", "cffg.run_results", "cffg.compiler"}
    )
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names.intersection(
        {"Engine", "Arena", "FactorType", "RunResult", "RunStatus"}
    )

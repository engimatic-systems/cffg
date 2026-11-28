"""Fully adaptive proposals cannot bypass domain, energy, or budget contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg import Engine, FailureDiagnostic, RunStatus, compile_template
from cffg import outer_strategy
from cffg.adaptive_proposals import _anderson, make_adaptive_proposals
from cffg.adaptive_runtime import certify_adaptive_proposal
from cffg.failure_facts import empty_failure
from cffg.fixed_point import RootContext, RootProgram
from cffg.outer_chart import OuterArrays, OuterChart, _VectorBlock
from cffg.runtime_types import OuterTerminationReason, _replace
from nonlinear_fixture import model


def scalar_program(step, reconstruct=None):
    arrays = OuterArrays(jnp.zeros(0), jnp.zeros(2), jnp.ones(1), jnp.zeros(0))
    vector = _VectorBlock("g_eta", np.array([0]), np.eye(1), np.eye(1))
    chart = OuterChart(arrays, (), (vector,), (), ((0, 1),), True, (0,))
    empty = jnp.zeros(0)
    context = RootContext(arrays, empty, empty, empty, empty, {}, metrics=chart.metrics)
    return RootProgram(chart, step, reconstruct), context


def test_nonfinite_mapped_eliminated_field_cannot_be_hidden_by_reconstruction():
    def step(messages, eta, lam, obs, mask, ef_eta, ef_obs, ef_mask, params, damp, harden):
        return messages, eta, jnp.full_like(lam, jnp.nan), ef_eta, 0., 0

    def reconstruct(messages, eta, lam, ef_eta, *context):
        return messages, eta, jnp.ones_like(lam), ef_eta, empty_failure()

    program, context = scalar_program(step, reconstruct)
    proposals = make_adaptive_proposals(program, lambda *args: jnp.asarray(0.))
    result = proposals(jnp.zeros(1), context, 2, 1e-6)
    assert not result.certified and result.rejected == result.rounds == 1
    assert result.count == 0
    np.testing.assert_array_equal(result.coordinates, [0.])


def test_rejected_trial_can_backtrack_to_a_root_from_the_last_valid_point():
    def step(messages, eta, lam, obs, mask, ef_eta, ef_obs, ef_mask, params, damp, harden):
        x = eta[0]
        target = jnp.where(x == 1.5, x, x + 2.)
        return messages, eta.at[0].set(target), lam, ef_eta, 0., jnp.where(x >= 2., 2, 0)

    program, context = scalar_program(step)
    proposals = make_adaptive_proposals(program, lambda *args: jnp.asarray(0.))
    # The first two valid points are 0 and 1. The trial at 2 fails.
    limited = proposals(jnp.zeros(1), context, 3, 1e-6)
    assert not limited.certified and limited.rounds == 3
    assert limited.count == 2 and limited.rejected == 1
    np.testing.assert_array_equal(limited.coordinates, [1.])

    recovered = proposals(jnp.zeros(1), context, 4, 1e-6)
    assert recovered.certified and recovered.rounds == 4
    assert recovered.count == 3 and recovered.rejected == 1
    np.testing.assert_array_equal(recovered.coordinates, [1.5])


@pytest.mark.parametrize("dtype,stress", [(np.float32, 1e20), pytest.param(np.float64, 1e200, marks=pytest.mark.slow)])
def test_first_invalid_cubic_proposal_preserves_fallback_attempt_and_budget(
    monkeypatch, dtype, stress,
):
    with jax.enable_x64(dtype is np.float64):
        template, arrays = model()
        engine = Engine(compile_template(template, arrays=arrays))
        state = engine.run(engine.init_state(), 60, tol=1e-9).require_converged()
        key = engine.arena.member_of["sensor"].parameter("parameters").key
        parameter = jnp.full_like(state.params[key], stress)
        assert parameter.dtype == dtype and np.isfinite(parameter).all()
        state = _replace(state, params={**state.params, key: parameter})

        fallback_budgets = []
        solve = outer_strategy.solve_outer

        def record_budget(*args, **kwargs):
            fallback_budgets.append(kwargs["n_iters"])
            return solve(*args, **kwargs)

        monkeypatch.setattr(outer_strategy, "solve_outer", record_budget)
        for budget, proposal_rounds in ((8, 0), (9, 1), (40, 1)):
            result = engine.run(state, budget, tol=1e-9)
            assert result.status is RunStatus.DIVERGED
            assert result.termination_reason is OuterTerminationReason.INNER_FAILURE
            assert result.failure == FailureDiagnostic("improper_tilt", "sensor", "soft", 1)
            assert result.iterations == 1 and result.rejected_steps == proposal_rounds
            assert result.outer_map_evaluations == proposal_rounds + 1
            assert len(result.message_deltas) == 0
            for actual, expected in zip(
                jax.tree.leaves(result.state), jax.tree.leaves(state), strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)
        assert fallback_budgets == [8, 8, 39]


def test_canonical_successor_energy_is_required_even_when_raw_energy_cancels():
    def step(messages, eta, lam, obs, mask, ef_eta, ef_obs, ef_mask, params, damp, harden):
        return messages, jnp.array([5e-7, -5e-5]), lam, ef_eta, 0., 0

    def energy(messages, eta, *args):
        return 100 * eta[0] + eta[1]

    program, context = scalar_program(step)
    result = make_adaptive_proposals(program, energy)(jnp.zeros(1), context, 1, 1e-6)
    assert result.residual < 1e-6 and result.energy == 0.
    assert not result.certified and result.rounds == 1


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rank_deficient_history_changes_directions_not_domain(dtype):
    with jax.enable_x64(dtype is np.float64):
        current = jnp.zeros(1000, dtype)
        residual = jnp.ones_like(current)
        history = jnp.tile(current, (6, 1))
        residuals = jnp.tile(residual, (6, 1))
        proposal = jax.jit(_anderson)(history, residuals, current, residual, 6, 5, 1.)
        assert np.isfinite(proposal).all()
        assert np.linalg.norm(proposal) <= 1. + 16 * np.finfo(dtype).eps


def test_small_budgets_skip_proposal_execution_and_remain_bounded():
    template, arrays = model()
    engine = Engine(compile_template(template, arrays=arrays))
    for budget in (0, 1, 4, 7):
        result = engine.run(engine.init_state(), budget, tol=1e-12)
        assert result.iterations <= budget
        if budget <= 1:
            assert result.status is not RunStatus.CONVERGED
        elif result.status is RunStatus.CONVERGED:
            assert result.outer_residual <= 1e-12 and result.failure is None
        assert all(plan[2]._cache_size() == 0 for plan in engine._outer_kernel_cache.values())


@pytest.mark.parametrize("dtype,dimension", [pytest.param(np.float64, 1, marks=pytest.mark.slow), (np.float32, 10)])
def test_generic_cubic_proposals_and_forced_trust_fallback_agree(monkeypatch, dtype, dimension):
    with jax.enable_x64(dtype is np.float64):
        template, arrays = model(dim=dimension)
        engine = Engine(compile_template(template, arrays=arrays))
        tolerance = 1e-9 if dtype is np.float64 else 2e-5
        fast = engine.run(engine.init_state(), 60, tol=tolerance)
        fast.require_converged()
        original = engine._outer_kernel_plan
        monkeypatch.setattr(engine, "_outer_kernel_plan", lambda *args: original(*args)._replace(proposal=None))
        reference = engine.run(engine.init_state(), 60, tol=tolerance)
        reference.require_converged()
        mean, covariance = engine.gaussian_marginals(fast.state)["x"]
        other_mean, other_covariance = engine.gaussian_marginals(reference.state)["x"]
        np.testing.assert_allclose(mean, other_mean, rtol=64 * tolerance, atol=64 * tolerance)
        np.testing.assert_allclose(covariance, other_covariance, rtol=64 * tolerance, atol=64 * tolerance)
        assert fast.outer_residual <= max(tolerance, 64 * np.finfo(dtype).eps)
        assert fast.free_energies[-1] == engine.bfe(fast.state)
        assert fast.iterations <= 60 and fast.failure is None


def test_host_energy_authority_can_veto_a_compiled_certificate():
    from cffg.adaptive_runtime import try_adaptive_proposals
    template, arrays = model()
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.run(engine.init_state(), 60, tol=1e-9).require_converged()
    attempt = try_adaptive_proposals(engine.arena, engine._outer_eliminated, engine._outer_kernel_plan,
                                    state, [], 30, 1e-9)
    assert attempt.result is not None

    def identity(arrays):
        return arrays, 0., 0, empty_failure()

    vetoed = certify_adaptive_proposal(attempt, identity, lambda _: float("inf"), lambda _: False, 1e-9)
    assert vetoed.result is None and vetoed.maps == attempt.maps + 1


def test_uncertified_proposal_consumes_budget_and_retains_traces_before_fallback(monkeypatch):
    template, arrays = model()
    engine = Engine(compile_template(template, arrays=arrays))
    original = engine._outer_kernel_plan

    def limited(*args):
        plan = original(*args)
        proposal = plan.proposal
        return plan._replace(proposal=lambda x, ctx, budget, tol: proposal(x, ctx, jnp.int32(1), tol))

    monkeypatch.setattr(engine, "_outer_kernel_plan", limited)
    result = engine.run(engine.init_state(), 40, tol=1e-9)
    result.require_converged()
    assert result.iterations <= 40 and result.linear_iterations > 0
    assert len(result.message_deltas) > 1 and result.message_deltas[0] > 1e-9
    assert result.free_energies[-1] == engine.bfe(result.state)

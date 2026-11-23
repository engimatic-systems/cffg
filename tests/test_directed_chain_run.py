"""Automatic directed solves through the public inference and controller APIs."""

import jax
import numpy as np
import pytest

from cffg import Engine, compile_template, set_params
from cffg.colony import Colony
from cffg.runtime_types import OuterTerminationReason, RunResult, RunStatus, _replace
from cstr_planning_fixture import planning_fixture
from test_cstr_planning import affine_reference
from test_directed_chain_plan import small_chain


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_partial_goal_posterior_and_energy_match_dense_gaussian(dtype):
    with jax.enable_x64(dtype is np.float64):
        template, arrays, data = planning_fixture(past=0, horizon=2, affine=True, dtype=dtype)
        engine = Engine(compile_template(template, arrays=arrays))
        initial = engine.bind_data(engine.init_state(), data)
        result = engine.run(initial, 1, tol=1e-6)
        assert result.termination_reason is OuterTerminationReason.DIRECTED_CHAIN_CERTIFIED
        assert result.directed_attempts == result.iterations == 1
        assert result.linear_iterations == result.solver_compilations == 0
        assert result.inner_status is RunStatus.CONVERGED
        state = result.require_converged()
        successor = engine.step(state)
        residuals = []
        for field in ("g_eta", "g_lam"):
            current = np.asarray(getattr(state, field))
            mapped = np.asarray(getattr(successor, field))
            residuals.append(np.max(np.abs(mapped - current) / (1 + np.abs(current))))
        assert result.outer_residual == max(residuals)
        energy = engine.bfe(state)
        assert result.free_energies[-1] == energy
        assert abs(engine.bfe(successor) - energy) <= max(1e-6, 64 * np.finfo(dtype).eps) * (1 + abs(energy))
        mu, cov, cost = affine_reference({k: v.astype(np.float64) for k, v in arrays.items()}, data)
        atol = 512 * np.finfo(dtype).eps if dtype is np.float32 else 1e-10
        np.testing.assert_allclose(result.free_energies[-1], cost, rtol=atol, atol=atol)
        for index in range(3):
            block = slice(index * 4, (index + 1) * 4)
            mean, covariance = engine.gaussian_marginal(state, f"s{index}")
            np.testing.assert_allclose(mean, mu[block], rtol=atol, atol=atol)
            np.testing.assert_allclose(covariance, cov[block, block], rtol=atol, atol=atol)


def test_float32_directed_proposal_rejects_disagreement_with_engine_step():
    with jax.enable_x64(False):
        template, arrays, data = planning_fixture(past=0, horizon=2, affine=True, dtype=np.float32,
                                                  process_scale=.02)
        engine = Engine(compile_template(template, arrays=arrays))
        initial = engine.bind_data(engine.init_state(), data)
        proposal, _ = engine._directed.candidate(initial)
        mapped = engine.step(proposal)
        before, after = np.asarray(proposal.g_eta), np.asarray(mapped.g_eta)
        error = np.max(np.abs(after - before) / (1 + abs(before)))
        result = engine.run(initial, 1, tol=1e-6)
        # Backend rounding can differ. Certification must follow this backend's actual step.
        if error > 64 * np.finfo(np.float32).eps:
            assert result.termination_reason is OuterTerminationReason.DIRECTED_CHAIN_UNCERTIFIED
            assert result.state is initial and not result.free_energies
            assert result.residual >= error
        elif result.status is RunStatus.CONVERGED:
            assert result.outer_residual >= error


def test_masks_parameters_and_metrics_are_live_on_reused_engine():
    template, arrays = small_chain()
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.bind_data(engine.init_state(), {"port0": [.25], "port2": [1.]})
    original = engine.run(state, 1, tol=1e-9).require_converged()

    def change(source):
        updated = engine.bind_data(source, {"port0": None, "port1": [.75]})
        return _replace(updated, params=set_params(engine, updated.params, {
            "alpha": {"parameters": np.array([-.2])},
            "source": {"mean": np.array([.1])},
        }))

    seeded = engine.run(change(original), 1, tol=1e-9).require_converged()
    fresh = engine.run(change(state), 1, tol=1e-9).require_converged()
    np.testing.assert_array_equal(seeded.g_eta, fresh.g_eta)
    np.testing.assert_array_equal(seeded.g_lam, fresh.g_lam)
    assert not np.allclose(original.g_eta, seeded.g_eta)
    assert np.isnan(engine.gaussian_marginal(engine.init_state(), "start")[0]).all()
    with pytest.raises(KeyError):
        engine.gaussian_marginal(seeded, "missing")


def test_invalid_factor_data_and_undefined_taylor_evaluation_report_owner():
    template, arrays = small_chain()
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.init_state()
    for ref in (engine.directed_chain_plan.prior, engine.directed_chain_plan.sites[0].transition,
                engine.directed_chain_plan.sites[1].emission):
        key = dict(ref.group.parameter_paths)["W"]
        for precision in (0., -1.):
            params = {**state.params, key: state.params[key].at[ref.row].set(precision)}
            result = engine.run(_replace(state, params=params), 1, tol=1e-6)
            assert result.status is RunStatus.DIVERGED
            assert (result.failure.reason, result.failure.factor) == ("invalid_noise", ref.group.members[ref.row])
            assert not result.free_energies
    # The declared cubic overflows at a finite upstream mean.
    params = set_params(engine, state.params, {"source": {"mean": np.array([1e104])}})
    result = engine.run(_replace(state, params=params), 1, tol=1e-6)
    assert result.status is RunStatus.DIVERGED and result.failure.reason == "nonfinite_function"


def test_dispatch_respects_explicit_schedules_budgets_and_failed_certificate(monkeypatch):
    from cffg import directed_chain_certificate as cert

    template, arrays = small_chain()
    arena = compile_template(template, arrays=arrays)
    engine = Engine(arena)
    state = engine.init_state()
    assert Engine(arena, "flood").directed_chain_plan is None
    assert Engine(arena, engine.schedule).directed_chain_plan is None
    calls = []

    def stock(state, budget, **kwargs):
        calls.append(budget)
        return RunResult(state, RunStatus.BUDGET_EXHAUSTED, budget, (), (), np.inf)

    monkeypatch.setattr(engine, "_run_outer_trust_region", stock)
    for budget, kwargs in [(0, {"tol": 1e-6}), (1, {}), (2, {"tol": 1e-6, "outer_solver": "trust_region"})]:
        assert engine.run(state, budget, **kwargs).directed_attempts == 0
    assert calls == [0, 1, 2]
    with pytest.raises(ValueError, match="damp"):
        engine.run(state, 2, tol=1e-6, damp=.5)
    monkeypatch.setattr(cert, "certify", lambda *args: cert.ChainCertificate(.5, 1., 2., False))
    result = engine.run(state, 1, tol=1e-6)
    assert result.termination_reason is OuterTerminationReason.DIRECTED_CHAIN_UNCERTIFIED
    assert result.state is state and not result.free_energies
    retry = engine.run(state, 5, tol=1e-6)
    assert retry.iterations == 5 and retry.directed_attempts == 1
    assert calls == [0, 1, 2, 4]


def test_colony_actions_use_each_instances_parameters_and_data():
    template, arrays = small_chain()
    colony = Colony()
    first = colony.spawn(template, arrays, observations={"port2": [1.]})
    second = colony.spawn(template, {**arrays, "mean": np.array([-1.])}, observations={"port1": [-1.]})
    results = colony.run(1, tol=1e-7)
    for row in results.values():
        assert row.termination_reason is OuterTerminationReason.DIRECTED_CHAIN_CERTIFIED
    engine, = colony.engines.values()
    mean = lambda i: engine.gaussian_marginal(results[i].require_converged(), "start")[0].item()
    assert mean(first) > 0 > mean(second)

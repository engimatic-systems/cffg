"""Retained planning convergence, initialization, and honest partial-goal scope."""

import jax
import numpy as np
import pytest

from cffg import Engine, RunStatus, compile_template
from cffg.colony import Colony
from cffg.compiled import NEG
from cffg.runtime_types import _replace
from hybrid_planning_fixture import declared_rule_reference, planning_fixture


@pytest.fixture(scope="module")
def retained():
    template, arrays, observations = planning_fixture()
    engine = Engine(compile_template(template, arrays=arrays))
    initial = engine.bind_data(engine.init_state(), observations)
    return engine, initial, arrays, observations


def assert_reference(engine, result, arrays, observations, *, tolerance=1e-8):
    assert result.status is RunStatus.CONVERGED
    assert result.failure is None
    assert len(result.free_energies) >= 2
    assert np.isfinite(result.free_energies).all()
    reference = declared_rule_reference(arrays, observations)
    marginal = engine.marginals(result.state)
    gaussian = engine.gaussian_marginals(result.state)
    for index in range(len(observations)):
        np.testing.assert_allclose(marginal[f"z{index}"], reference.regimes[index],
                                   rtol=tolerance, atol=tolerance)
    for index in range(len(observations) + 1):
        mean, covariance = gaussian[f"x{index}"]
        np.testing.assert_allclose(mean, reference.means[index], rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(covariance, reference.covariances[index], rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(engine.bfe(result.state), reference.energy, rtol=tolerance, atol=10 * tolerance)
    # Fresh undamped flooding is independent of the stopping trace/schedule.
    checked = engine.step(result.state, damp=0., harden=False)
    for field in ("messages", "g_eta", "g_lam"):
        np.testing.assert_allclose(getattr(checked, field), getattr(result.state, field),
                                   rtol=tolerance, atol=10 * tolerance)


def test_retained_hybrid_reaches_independent_moment_and_ledger_fixed_point(retained):
    engine, initial, arrays, observations = retained
    result = engine.run(initial, 200, tol=1e-9)
    assert_reference(engine, result, arrays, observations)
    assert result.iterations == 16
    assert len(result.free_energies) == 2
    damped = engine.run(initial, 200, tol=1e-9, damp=.5)
    assert_reference(engine, damped, arrays, observations)
    assert result.iterations < damped.iterations < 200


@pytest.mark.parametrize("budget,status,finite_entries", [
    (0, RunStatus.BUDGET_EXHAUSTED, 0),
    (1, RunStatus.DIVERGED, 0),
    (14, RunStatus.DIVERGED, 0),
    (15, RunStatus.BUDGET_EXHAUSTED, 1),
])
def test_initialization_is_bounded_and_never_certifies_undefined_energy(retained, budget, status, finite_entries):
    engine, initial, _, _ = retained
    result = engine.run(initial, budget, tol=1e-6)
    assert result.status is status
    assert result.iterations == budget
    assert len(result.free_energies) == finite_entries
    assert result.failure is not None
    if budget in (1, 14):
        assert result.failure.reason == "undefined_objective"


def test_fixed_budget_and_initial_rejections_do_not_start_hidden_retries(retained):
    engine, initial, _, _ = retained
    fixed = engine.run(initial, 17, tol=None)
    assert fixed.status is RunStatus.BUDGET_EXHAUSTED
    assert fixed.iterations == 17
    assert np.isfinite(fixed.free_energies).all()
    hardened = engine.run(initial, 16, tol=1e-9, harden_last=16)
    assert hardened.status is RunStatus.CONVERGED
    assert hardened.iterations == 16  # No delta constraint exists in this model.
    invalid = _replace(initial, g_eta=initial.g_eta.at[0].set(np.nan))
    rejected = engine.run(invalid, 200, tol=1e-6)
    assert rejected.status is RunStatus.DIVERGED
    assert rejected.iterations == 0
    edge = engine.arena.edges["z0"]
    indices = engine.arena.message_indices(edge.slot0, 3)
    empty = _replace(initial, messages=initial.messages.at[indices].set(NEG))
    result = engine.run(empty, 200, tol=1e-6)
    assert result.status is RunStatus.INFEASIBLE
    assert result.iterations == 0


def test_partial_goal_limitation_is_not_misrepresented_as_full_horizon_planning(retained):
    engine, initial, arrays, observations = retained
    baseline = engine.run(initial, 200, tol=1e-9)
    changed = {**observations, "g15": np.array([2.])}
    later = engine.run(engine.bind_data(initial, changed), 200, tol=1e-9)
    assert_reference(engine, later, arrays, changed)
    first = engine.marginals(baseline.state)
    second = engine.marginals(later.state)
    np.testing.assert_allclose(first["z0"], second["z0"], rtol=0, atol=1e-12)
    assert np.max(np.abs(first["z14"] - second["z14"])) > .5
    assert abs(later.free_energies[-1] - baseline.free_energies[-1]) > 100


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("schedule", ["auto", "flood"])
def test_neighboring_noise_and_sparse_actions_have_truthful_results(dtype, schedule):
    with jax.enable_x64(dtype is np.float64):
        template, arrays, observations = planning_fixture(3, dtype, noise_scale=2., habit=[.4, .6, 0.])
        engine = Engine(compile_template(template, arrays=arrays), schedule=schedule)
        initial = engine.bind_data(engine.init_state(), observations)
        result = engine.run(initial, 100, tol=1e-6)
        assert result.status in (RunStatus.CONVERGED, RunStatus.BUDGET_EXHAUSTED)
        if result.status is RunStatus.CONVERGED:
            tolerance = 256 * np.finfo(dtype).eps if dtype is np.float32 else 1e-8
            assert_reference(engine, result, arrays, observations, tolerance=tolerance)
            reference = declared_rule_reference(arrays, observations)
            for index in range(3):
                probabilities = engine.marginals(result.state)[f"z{index}"]
                assert probabilities[2] == 0.
                if dtype is np.float32:
                    # Local log odds have a tighter accuracy check than the solver residual.
                    actual = np.log(float(probabilities[0]) / float(probabilities[1]))
                    expected = np.log(reference.regimes[index, 0] / reference.regimes[index, 1])
                    assert abs(actual - expected) <= 64 * np.finfo(dtype).eps
        else:
            assert result.iterations == 100
            assert result.failure is not None


@pytest.mark.slow
def test_retained_singleton_and_batched_colony_match_individual_engines(retained):
    engine, initial, arrays, observations = retained
    expected = engine.run(initial, 200, tol=1e-9)
    template, _, _ = planning_fixture()
    singleton = Colony()
    identity = singleton.spawn(template, arrays, observations=observations)
    assert_reference(next(iter(singleton.engines.values())), singleton.run(200, tol=1e-9)[identity],
                     arrays, observations)
    colony = Colony()
    inputs = (arrays, {**arrays, "Q": arrays["Q"] * .5})
    identities = [colony.spawn(template, values, observations=observations) for values in inputs]
    results = colony.run(200, tol=1e-9)
    assert len(colony.engines) == 1
    batch_engine = next(iter(colony.engines.values()))
    assert results[identities[0]].iterations == expected.iterations
    for identity, values in zip(identities, inputs):
        assert_reference(batch_engine, results[identity], values, observations)


@pytest.mark.slow
def test_tenfold_horizon_preserves_bounded_initialization_and_fixed_point():
    template, arrays, observations = planning_fixture(150)
    engine = Engine(compile_template(template, arrays=arrays))
    initial = engine.bind_data(engine.init_state(), observations)
    result = engine.run(initial, 200, tol=1e-9)
    assert_reference(engine, result, arrays, observations)
    assert result.iterations == 151
    assert len(result.free_energies) == 2

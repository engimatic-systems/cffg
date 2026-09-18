"""Adaptive factors use certified iterative inference on both schedule routes."""

import numpy as np
import pytest
import jax
from nonlinear_fixture import adaptive_pair, model

from cffg.compiler import compile_template
from cffg.engine import Engine, RunStatus


@pytest.mark.parametrize("method", ["linearization", "sigma_point"])
@pytest.mark.parametrize("schedule", ["auto", "flood"])
def test_curved_sensor_fixed_point_and_energy_contract(method, schedule):
    template, arrays = model(method)
    engine = Engine(compile_template(template, arrays=arrays), schedule=schedule)
    assert engine.has_adaptive_gaussian
    assert engine._propagate is None
    outcome = engine.run(engine.init_state(), 30, tol=1e-9)
    state = outcome.require_converged()
    expected = (.606501293489, .154735136965) if method == "linearization" else (.510302853248, .137267504139)
    mean, covariance = engine.gaussian_marginals(state)["x"]
    np.testing.assert_allclose([mean[0], covariance[0, 0]], expected, atol=1e-8)
    assert np.isfinite(engine.bfe(state))
    metadata = engine.objective_metadata["sensor"]
    assert metadata["objective"] == method + "_frozen_gaussian_bethe"
    assert metadata["approximate"] and metadata["evidence_bound"] is None
    assert metadata["stationarity"] == "frozen coefficients only"
    assert engine.objective_contract["approximate"]
    assert engine.objective_metadata["prior"]["objective"] == "analytic"
    stats = engine.gaussian_node_stats(state)["sensor"]
    assert stats["proper"] and stats["failure_code"] == 0


def test_budget_and_unsupported_unglobalized_controls():
    template, arrays = model()
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.init_state()
    assert engine.run(state, 0).status == RunStatus.BUDGET_EXHAUSTED
    with pytest.raises(ValueError, match="globalization"):
        engine.run(state, 10, outer_solver="picard")
    with pytest.raises(ValueError, match="damp"):
        engine.run(state, 10, damp=.5)


@pytest.mark.parametrize("method", ["linearization", "sigma_point"])
@pytest.mark.parametrize("observed", [False, True])
def test_affine_limit_matches_full_analytic_energy(method, observed):
    from dataclasses import replace

    template, arrays = model(method, affine=True, observed=observed)
    nonlinear = Engine(compile_template(template, arrays=arrays))
    factor = template.factors["sensor"]
    arrays = {**arrays, "A": np.eye(1), "b": np.zeros(1)}
    from cffg.core import BlobRef
    template.blobs.update({n: BlobRef(n, n + ".npy") for n in ("A", "b")})
    template.factors["sensor"] = replace(factor, type="gaussian_linear", function=None,
                                         param_blobs={"matrix": "A", "offset": "b", "cov": "V"})
    analytic = Engine(compile_template(template, arrays=arrays))
    states = []
    for engine in (nonlinear, analytic):
        state = engine.init_state()
        if observed:
            state = engine.bind_data(state, {"y": np.array([.8])})
        states.append(engine.run(state, 20, tol=1e-9).require_converged())
    np.testing.assert_allclose(nonlinear.bfe(states[0]), analytic.bfe(states[1]), atol=1e-8)
    np.testing.assert_allclose(nonlinear.gaussian_marginals(states[0])["x"][0],
                               analytic.gaussian_marginals(states[1])["x"][0], atol=1e-8)


def test_distinct_observations_and_parameters_follow_vmap_and_host():
    from cffg import set_params
    from cffg.runtime_types import _replace

    template, arrays = model("sigma_point", observed=True)
    engine = Engine(compile_template(template, arrays=arrays), schedule="flood")
    states = []
    for y, theta in ((.3, .1), (1.1, .4)):
        state = engine.bind_data(engine.init_state(), {"y": np.array([y])})
        state = _replace(state, params=set_params(engine, state.params, {"sensor": {"parameters": np.array([theta])}}))
        states.append(engine.step(state))
    batched = jax.tree.map(lambda *xs: np.stack(xs), *states)
    def step(state):
        values = engine._iter_step(
            state.messages, state.g_eta, state.g_lam, state.g_obs, state.g_mask,
            state.ef_eta, state.ef_obs, state.ef_mask, state.params, 0., False,
        )
        return _replace(state, messages=values[0], g_eta=values[1],
                        g_lam=values[2], ef_eta=values[3])

    actual = jax.jit(jax.vmap(step))(batched)
    host = [engine.step(state) for state in states]
    for index, state in enumerate(host):
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(state), strict=True):
            np.testing.assert_allclose(a[index], b, rtol=1e-10, atol=1e-10)
    assert not np.allclose(host[0].g_eta, host[1].g_eta)


def test_improper_runtime_cavity_is_a_machine_readable_failure():
    from cffg.nonlinear import NonlinearFailure
    from cffg.runtime_types import _replace

    template, arrays = model()
    engine = Engine(compile_template(template, arrays=arrays), schedule="flood")
    state = engine.step(engine.init_state())
    group = next(g for g in engine.arena.gauss_groups if g.kernel.binding.iterative)
    index = engine.arena.g_lam_indices(group.gather[:, 1], 1)
    state = _replace(state, g_lam=state.g_lam.at[index].set(-1.))
    values = engine.gaussian_node_stats(state)["sensor"]
    assert not values["proper"]
    assert values["failure_code"] == NonlinearFailure.IMPROPER_CAVITY
    assert np.isinf(engine.bfe(state))
    assert engine.run(state, 5, tol=1e-8).status == RunStatus.DIVERGED


@pytest.mark.parametrize("method, variance", [
    ("linearization", .15642104055141876), ("sigma_point", .1395200705766338),
])
@pytest.mark.parametrize("schedule", ["auto", "flood"])
def test_mutually_changing_cavities_match_independent_site_root(method, variance, schedule):
    template, arrays = adaptive_pair(method)
    engine = Engine(compile_template(template, arrays=arrays), schedule=schedule)
    result = engine.run(engine.init_state(), 40, tol=1e-8)
    state = result.require_converged()
    mean, covariance = engine.gaussian_marginals(state)["x0"]
    np.testing.assert_allclose(mean, [0.], atol=2e-8)
    np.testing.assert_allclose(covariance, [[variance]], atol=2e-8)
    assert result.outer_residual <= 1e-8
    assert np.isfinite(result.free_energies).all()


def test_rejected_trials_cannot_poison_acceleration_or_claim_small_step_convergence():
    template, arrays = adaptive_pair("sigma_point", evidence=1.5, variance=.05)
    engine = Engine(compile_template(template, arrays=arrays), schedule="flood")
    result = engine.run(engine.init_state(), 4, tol=1e-8)
    assert result.status == RunStatus.BUDGET_EXHAUSTED
    assert result.outer_residual > 1e-8
    assert result.rejected_steps > 0
    assert np.isfinite(engine.bfe(result.state))


@pytest.mark.parametrize("scale", [1e-6, 1e6])
def test_float32_coordinate_units_do_not_invalidate_proper_gaussians(scale):
    with jax.enable_x64(False):
        template, arrays = model(affine=True)
        for name in ("m", "e"):
            arrays[name] *= scale
        for name in ("P", "V", "R"):
            arrays[name] *= scale ** 2
        engine = Engine(compile_template(template, arrays=arrays))
        result = engine.run(engine.init_state(), 30, tol=1e-4)
        state = result.require_converged()
        mean, covariance = engine.gaussian_marginals(state)["x"]
        np.testing.assert_allclose(mean / scale, [3.7 / 6], atol=1e-4)
        np.testing.assert_allclose(covariance / scale ** 2, [[1 / 6]], atol=1e-4)


def test_colony_keeps_independent_adaptive_run_certificates():
    from cffg.colony import Colony

    template, arrays = model(observed=True)
    colony = Colony()
    ids = [colony.spawn(template, {**arrays, "theta": np.array([theta])}, observations={"y": np.array([y])})
           for theta, y in ((.1, .3), (.4, 1.1))]
    results = colony.run(20, tol=1e-8)
    for iid in ids:
        assert results[iid].status == RunStatus.CONVERGED
        assert results[iid].outer_residual < 1e-8
    assert not np.allclose(results[ids[0]].state.g_eta, results[ids[1]].state.g_eta)


@pytest.mark.slow
def test_exact_scan_island_attached_to_adaptive_factor_matches_flooding():
    from dataclasses import replace
    from cffg.core import BlobRef, FactorSpec, VariableSpec
    from cffg.schedule import ScanStage

    template, arrays = model("sigma_point", observed=True)
    length = 8
    template.variables.update({f"x{i}": VariableSpec(f"x{i}", dim=1) for i in range(length)})
    template.factors["prior"] = replace(template.factors["prior"], edges={"out": "x0"})
    arrays.update({"A": np.eye(1), "b": np.zeros(1), "Q": np.eye(1) * .05})
    template.blobs.update({name: BlobRef(name, name + ".npy") for name in ("A", "b", "Q")})
    for i in range(length):
        name = f"link{i}"
        template.factors[name] = FactorSpec(name, "gaussian_linear",
            {"in": f"x{i}", "out": f"x{i+1}" if i+1 < length else "x"},
            param_blobs={"matrix": "A", "offset": "b", "cov": "Q"})
    arena = compile_template(template, arrays=arrays)
    engines = [Engine(arena, schedule=schedule) for schedule in ("auto", "flood")]
    schedule = engines[0].schedule
    assert schedule.kind == "mixed"
    assert any(isinstance(stage, ScanStage) for stage in (*schedule.prime_stages, *schedule.stages, *schedule.forest_stages))
    states = [engine.run(engine.bind_data(engine.init_state(), {"y": np.array([.8])}), 40, tol=1e-8).require_converged()
              for engine in engines]
    for field in ("g_eta", "g_lam"):
        np.testing.assert_allclose(getattr(states[0], field), getattr(states[1], field), atol=1e-7)
    np.testing.assert_allclose(engines[0].bfe(states[0]), engines[1].bfe(states[1]), atol=1e-7)

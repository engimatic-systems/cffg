"""The recent full CSTR window is opt-in; its scientific workload stays fixed."""

import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest

from cffg import RunStatus
from cffg.fixed_point import RootProgram, root_context
from cffg.outer_chart import OuterArrays, build_outer_chart, chart_coordinates

spec = importlib.util.spec_from_file_location(
    "recent_cstr_fixture", Path(__file__).parents[1] / "tools" / "cstr_runtime_fixture.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def test_recent_validation_structure_and_covariances_are_not_the_short_example():
    template, arrays = fixture.model()
    assert len([factor for factor in template.factors.values() if factor.type == "gaussian_nonlinear"]) == 19
    assert len(template.ports) == 20
    np.testing.assert_array_equal(arrays["Q"], np.diag([1e-5, 1e-5, 4e-6, .25]))
    np.testing.assert_array_equal(arrays["P0"], np.diag([1., 1., 1e-4, 1.]))
    assert fixture.CONFIGURATION["substeps"] == 4 and fixture.CONFIGURATION["dt"] == 26 / 60
    assert fixture.PAST == 4 and fixture.HORIZON == 15 and fixture.TOLERANCE == 1e-6


@pytest.mark.slow
def test_recent_cstr_full_certificate_and_changed_observations_reuse_executables():
    engine, _, _ = fixture.make_engine()
    base, binder = engine.init_state(), fixture.Binder(engine)
    executable_counts = None
    certificate = None
    for change, history, goal in ((0, False, 0.), (1, False, 0.), (2, False, 0.),
                                 (1, True, 1 / 3), (2, True, 2 / 15)):
        data = fixture.observations(change, history=history, goal=goal)
        result = engine.run(binder.bind(base, data), fixture.BUDGET, tol=fixture.TOLERANCE)
        state = result.require_converged()
        counts = tuple(tuple(kernel._cache_size() for kernel in plan) for plan in engine._outer_kernel_cache.values())
        if executable_counts is not None:
            assert counts == executable_counts and result.solver_compilations == 0
        executable_counts = counts
        chart = build_outer_chart(engine.arena, OuterArrays(state.messages, state.g_eta, state.g_lam, state.ef_eta),
                                  engine._outer_eliminated)
        if certificate is None:
            program = RootProgram(chart, engine._iter_step, engine._reconstruct)
            certificate = jax.jit(lambda x, ctx: program.certificate(x, ctx, engine._bfe))
        energy, successor, root, inner, status, _ = certificate(chart_coordinates(chart, chart.base), root_context(chart, state))
        assert np.isfinite([energy, successor, root, inner]).all() and status == 0
        assert root <= fixture.TOLERANCE
        assert abs(successor - energy) <= fixture.TOLERANCE * (1 + abs(energy))
        assert result.linear_iterations == 0 and result.iterations <= fixture.BUDGET
        assert result.failure is None and result.free_energies[-1] == engine.bfe(state)


@pytest.mark.slow
def test_recent_sigma_point_startup_failure_is_not_masked_by_proposals():
    engine, _, _ = fixture.make_engine("sigma_point")
    state = fixture.Binder(engine).bind(engine.init_state(), fixture.observations())
    result = engine.run(state, fixture.BUDGET, tol=fixture.TOLERANCE)
    assert result.status is RunStatus.DIVERGED
    assert result.failure.reason == "improper_tilt" and result.failure.factor == "tr1"
    assert result.failure.phase == "startup" and result.failure.iteration == 2
    assert result.solver_compilations == 0

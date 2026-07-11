"""Resource estimator: exact where it can be, honest where it can't."""

import numpy as np

from cffg.compiler import compile_template
from cffg.engine import Engine, RunStatus
from cffg.estimate import estimate
from cffg.models.tmaze import build_tmaze


def make_engine():
    t, arrays = build_tmaze()
    return Engine(compile_template(t, arrays=arrays))


def test_analytic_memory_is_exact():
    engine = make_engine()
    est = estimate(engine, calibrate=False)
    arena = engine.arena
    itemsize = np.asarray(engine.init_state().messages).itemsize
    assert est.message_bytes == arena.n_slots * arena.k_max * itemsize
    assert est.param_bytes > 0
    assert est.state_bytes == est.message_bytes + est.param_bytes


def test_calibration_and_prediction():
    engine = make_engine()
    est = estimate(engine, sample_iters=5)
    assert est.seconds_per_sweep is not None and est.seconds_per_sweep > 0
    assert est.calibration_status is RunStatus.BUDGET_EXHAUSTED
    assert abs(est.predict_seconds(100) - 100 * est.seconds_per_sweep) < 1e-12
    assert "sweep" in est.summary()


def test_cost_analysis_reported_when_available():
    engine = make_engine()
    est = estimate(engine, calibrate=False)
    # Backend-dependent: None is legal, but if reported it must be positive.
    if est.flops_per_sweep is not None:
        assert est.flops_per_sweep > 0
    if est.bytes_accessed_per_sweep is not None:
        assert est.bytes_accessed_per_sweep > 0


def test_bigger_model_costs_more():
    small_t, small_a = build_tmaze(steps=1)
    big_t, big_a = build_tmaze(steps=4)
    small = estimate(Engine(compile_template(small_t, arrays=small_a)), calibrate=False)
    big = estimate(Engine(compile_template(big_t, arrays=big_a)), calibrate=False)
    assert big.message_bytes > small.message_bytes
    if big.flops_per_sweep and small.flops_per_sweep:
        assert big.flops_per_sweep > small.flops_per_sweep

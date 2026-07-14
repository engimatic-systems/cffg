"""Resource estimator: exact where it can be, honest where it can't."""

import jax
import jax.numpy as jnp
import numpy as np

from cffg.compiler import compile_template
from cffg.core.ir import BlobRef, FactorSpec, PortRole, PortSpec, Template, VariableSpec
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
    state = engine.init_state()
    actual = {
        name: int(getattr(state, name).nbytes)
        for name in (
            "messages", "g_eta", "g_lam", "g_obs", "g_mask",
            "ef_eta", "ef_obs", "ef_mask",
        )
    }
    assert est.arena_bytes == actual
    assert est.message_bytes == state.messages.nbytes
    assert est.param_bytes > 0
    assert est.state_bytes == sum(
        leaf.nbytes for leaf in jax.tree_util.tree_leaves(state)
    )


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


def _hostile_heterogeneous_engine(n_small=1_000, k_outlier=10_000, d_outlier=128):
    """The adversarial-review fixture from cff-44we's memory calculation."""
    variables = {}
    factors = {}
    for i in range(n_small):
        dname = f"disc_{i}"
        gname = f"gauss_{i}"
        variables[dname] = VariableSpec(dname, cardinality=2)
        variables[gname] = VariableSpec(gname, dim=1)
        factors[f"cat_{i}"] = FactorSpec(
            f"cat_{i}", "categorical", {"out": dname},
            param_blobs={"statistic": "cat2"},
        )
        factors[f"normal_{i}"] = FactorSpec(
            f"normal_{i}", "gaussian", {"out": gname},
            param_blobs={"mean": "mean1", "cov": "cov1"},
        )

    variables["disc_outlier"] = VariableSpec("disc_outlier", cardinality=k_outlier)
    variables["gauss_outlier"] = VariableSpec("gauss_outlier", dim=d_outlier)
    factors["cat_outlier"] = FactorSpec(
        "cat_outlier", "categorical", {"out": "disc_outlier"},
        param_blobs={"statistic": "cat_outlier"},
    )
    factors["normal_outlier"] = FactorSpec(
        "normal_outlier", "gaussian", {"out": "gauss_outlier"},
        param_blobs={"mean": "mean_outlier", "cov": "cov_outlier"},
    )
    blob_names = ("cat2", "cat_outlier", "mean1", "cov1", "mean_outlier", "cov_outlier")
    template = Template(
        "hostile_heterogeneous",
        variables=variables,
        factors=factors,
        blobs={name: BlobRef(name, f"{name}.npy") for name in blob_names},
        ports={name: PortSpec(name, name, PortRole.GLUE) for name in variables},
    )
    arrays = {
        "cat2": np.full(2, 0.5),
        "cat_outlier": np.full(k_outlier, 1.0 / k_outlier),
        "mean1": np.zeros(1),
        "cov1": np.eye(1),
        "mean_outlier": np.zeros(d_outlier),
        "cov_outlier": np.eye(d_outlier),
    }
    return Engine(compile_template(template, arrays=arrays))


def test_shape_segmented_arenas_are_exact_at_hostile_scale_and_survive_float32():
    n, k, d = 1_000, 10_000, 128
    engine = _hostile_heterogeneous_engine(n, k, d)
    arena, state = engine.arena, engine.init_state()

    expected_disc = 2 * (n * 2 + k)
    expected_eta = 2 * (n + d)
    expected_lam = 2 * (n + d * d)
    assert state.messages.size == expected_disc
    assert state.g_eta.size == expected_eta
    assert state.g_lam.size == expected_lam
    assert arena.n_message_values == expected_disc
    assert arena.n_g_eta_values == expected_eta
    assert arena.n_g_lam_values == expected_lam

    # Exact flat storage is 1.0x the information-form representation (<2x),
    # while the removed model-wide padding would be hundreds of times larger.
    assert state.messages.size / expected_disc == 1.0
    assert state.g_lam.size / expected_lam == 1.0
    assert arena.n_slots * k > 800 * expected_disc
    assert arena.n_gslots * d * d > 900 * expected_lam

    state32 = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float32) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        state,
    )
    stepped = engine.step(state32)
    for leaf in jax.tree_util.tree_leaves(stepped):
        assert bool(jnp.isfinite(leaf).all())
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert leaf.dtype == jnp.float32

"""Engine integration tests for the D5 hybrid regime-epistemics composite."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cffg.compiler import CompileError, compile_template
from cffg.core.ir import (
    BlobRef,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.engine import Engine, RunStatus, _replace
from cffg.experiments.hybrid_epistemics import (
    ProbeWorld,
    collapse,
    gfe_hat,
    predictive,
)

jax.config.update("jax_enable_x64", True)


def _blobs(*names):
    return {name: BlobRef(name, f"{name}.npy") for name in names}


def _linear_template(
    *,
    matrix,
    offset,
    noise,
    state_mean,
    state_cov,
    selector,
    goal_mean,
    goal_cov,
):
    matrix = np.asarray(matrix)
    regimes, out_dim, in_dim = matrix.shape
    template = Template(
        name="hybrid-gfe-linear",
        variables={
            "x": VariableSpec("x", dim=out_dim),
            "z": VariableSpec("z", dim=in_dim),
            "s": VariableSpec("s", cardinality=regimes),
        },
        factors={
            "state": FactorSpec(
                "state",
                "gaussian",
                edges={"out": "z"},
                param_blobs={"mean": "state_mean", "cov": "state_cov"},
            ),
            "emission": FactorSpec(
                "emission",
                "gaussian_mixture_linear",
                edges={"out": "x", "in": "z", "selector": "s"},
                param_blobs={"matrix": "matrix", "offset": "offset", "cov": "noise"},
            ),
            "selector": FactorSpec(
                "selector",
                "categorical",
                edges={"out": "s"},
                param_blobs={"statistic": "selector"},
            ),
            "goal": FactorSpec(
                "goal",
                "gaussian",
                edges={"out": "x"},
                param_blobs={"mean": "goal_mean", "cov": "goal_cov"},
            ),
        },
        constraints=[
            ConstraintSpec(
                ConstraintKind.P_SUBSTITUTION,
                "emission",
                detail=(("substituted_role", "out"),),
            )
        ],
        blobs=_blobs(
            "matrix",
            "offset",
            "noise",
            "state_mean",
            "state_cov",
            "selector",
            "goal_mean",
            "goal_cov",
        ),
    )
    arrays = {
        "matrix": matrix,
        "offset": np.asarray(offset),
        "noise": np.asarray(noise),
        "state_mean": np.asarray(state_mean),
        "state_cov": np.asarray(state_cov),
        "selector": np.asarray(selector),
        "goal_mean": np.asarray(goal_mean),
        "goal_cov": np.asarray(goal_cov),
    }
    return template, arrays


def _probe_template(action="probe", selector=None):
    world = ProbeWorld()
    matrix, offset, noise = world.emission(action)
    template, arrays = _linear_template(
        matrix=matrix,
        offset=offset,
        noise=noise,
        state_mean=np.zeros(1),
        state_cov=np.array([[world.S0]]),
        selector=(
            np.full(2, 0.5) if selector is None else np.asarray(selector)
        ),
        goal_mean=np.array([world.g1]),
        goal_cov=np.array([[world.gamma1]]),
    )
    return world, template, arrays


def _prior_template():
    template = Template(
        name="hybrid-gfe-prior",
        variables={
            "x": VariableSpec("x", dim=1),
            "s": VariableSpec("s", cardinality=2),
        },
        factors={
            "mixture": FactorSpec(
                "mixture",
                "gaussian_mixture",
                edges={"out": "x", "selector": "s"},
                param_blobs={"mean": "mean", "cov": "noise"},
            ),
            "selector": FactorSpec(
                "selector",
                "categorical",
                edges={"out": "s"},
                param_blobs={"statistic": "selector"},
            ),
            "goal": FactorSpec(
                "goal",
                "gaussian",
                edges={"out": "x"},
                param_blobs={"mean": "goal_mean", "cov": "goal_cov"},
            ),
        },
        constraints=[
            ConstraintSpec(
                ConstraintKind.P_SUBSTITUTION,
                "mixture",
                detail=(("substituted_role", "out"),),
            )
        ],
        blobs=_blobs("mean", "noise", "selector", "goal_mean", "goal_cov"),
    )
    arrays = {
        "mean": np.array([[-1.0], [1.0]]),
        "noise": np.array([[[0.4]], [[0.4]]]),
        "selector": np.array([0.5, 0.5]),
        "goal_mean": np.array([0.0]),
        "goal_cov": np.array([[1.2]]),
    }
    return template, arrays


def _run(
    template,
    arrays,
    *,
    dtype=None,
    iterations=200,
    tol=1e-10,
    schedule="auto",
):
    engine = Engine(compile_template(template, arrays=arrays), schedule=schedule)
    state = engine.init_state()
    if dtype is not None:
        state = jax.tree_util.tree_map(
            lambda value: (
                value.astype(dtype)
                if jnp.issubdtype(value.dtype, jnp.inexact)
                else value
            ),
            state,
        )
    result = engine.run(state, iterations, tol=tol)
    return engine, result


def test_compiler_derives_one_composite_and_absorbs_goal():
    _, template, arrays = _probe_template()
    arena = compile_template(template, arrays=arrays)

    assert len(arena.hybrid_gfe_groups) == 1
    group = arena.hybrid_gfe_groups[0]
    assert group.members == ("emission",)
    assert group.goal_members == ("goal",)
    assert group.type == "gaussian_mixture_linear"
    assert group.dims == (1, 1) and group.k_sel == 2
    assert not arena.mix_groups
    assert all("goal" not in gauss.members for gauss in arena.gauss_groups)
    x = arena.edges["x"]
    assert all(
        not np.any(np.all(pairs == np.array([x.slot0, x.slot1]), axis=1))
        for _, pairs in arena.gauss_interior
    )
    schedule = Engine(arena).schedule
    assert schedule.kind == "mixed" and "GFE composites" in schedule.reason


def test_hybrid_gfe_requires_out_role_and_fixed_gaussian_goal():
    _, template, arrays = _probe_template()
    template.constraints[0] = ConstraintSpec(
        ConstraintKind.P_SUBSTITUTION,
        "emission",
        detail=(("substituted_role", "in"),),
    )
    with pytest.raises(CompileError, match="substituted_role 'out' only"):
        compile_template(template, arrays=arrays)

    _, template, arrays = _probe_template()
    del template.factors["goal"]
    template.ports["xout"] = PortSpec("xout", "x", PortRole.GLUE)
    with pytest.raises(CompileError, match="fixed Gaussian goal"):
        compile_template(template, arrays=arrays)

    _, template, arrays = _probe_template()
    del template.factors["state"]
    template.ports["zin"] = PortSpec("zin", "z", PortRole.GLUE)
    with pytest.raises(CompileError, match="interior Gaussian state edge"):
        compile_template(template, arrays=arrays)


def test_gaussian_mixture_prior_composite_runs():
    template, arrays = _prior_template()
    engine, result = _run(template, arrays)
    assert result.status is RunStatus.CONVERGED
    group = engine.arena.hybrid_gfe_groups[0]
    assert group.type == "gaussian_mixture" and group.dims == (1,)
    diagnostic = engine.hybrid_gfe_diagnostics(result.state)["mixture"]
    np.testing.assert_allclose(diagnostic["rho"], [0.5, 0.5], atol=1e-10)
    assert diagnostic["mi_state"] == pytest.approx(0.0, abs=1e-12)
    actual_mean, actual_cov = engine.gaussian_marginals(result.state)["x"]
    np.testing.assert_allclose(actual_mean, [0.0], atol=1e-10)
    np.testing.assert_allclose(actual_cov, [[1.4]], atol=1e-10)


def test_engine_matches_d5_oracle_and_reports_bounds():
    world, template, arrays = _probe_template("probe")
    engine, result = _run(template, arrays)
    assert result.status is RunStatus.CONVERGED

    diagnostic = engine.hybrid_gfe_diagnostics(result.state)["emission"]
    rho = diagnostic["rho"]
    state_means, state_covariances = world.state_belief()
    matrix, offset, noise = world.emission("probe")
    oracle = gfe_hat(
        rho,
        state_means,
        state_covariances,
        matrix,
        offset,
        noise,
        world.g1,
        world.gamma1,
    )
    np.testing.assert_allclose(diagnostic["gfe_lower_bound"], oracle.total, atol=1e-9)
    np.testing.assert_allclose(
        diagnostic["mi_regime_surrogate"], oracle.mi_regime_hat, atol=1e-9
    )
    np.testing.assert_allclose(
        [diagnostic["mi_regime_exact_lower"], diagnostic["mi_regime_exact_upper"]],
        oracle.mi_regime_exact_bounds,
        atol=1e-9,
    )
    assert diagnostic["bound_direction"] == "reported_gfe_lower_bounds_exact"
    assert diagnostic["status"] == RunStatus.CONVERGED.value
    assert np.isfinite(engine.bfe(result.state))

    expected_mean, expected_cov = collapse(
        predictive(rho, state_means, state_covariances, matrix, offset, noise)
    )
    actual_mean, actual_cov = engine.gaussian_marginals(result.state)["x"]
    np.testing.assert_allclose(actual_mean, expected_mean, atol=1e-9)
    np.testing.assert_allclose(actual_cov, expected_cov, atol=1e-9)
    np.testing.assert_allclose(engine.marginals(result.state)["s"], rho, atol=1e-9)

    group = engine.arena.hybrid_gfe_groups[0]
    in_eta = result.state.g_eta[
        engine.arena.g_eta_indices(group.g_scatter[:, 1], 1)
    ]
    in_lam = result.state.g_lam[
        engine.arena.g_lam_indices(group.g_scatter[:, 1], 1)
    ]
    np.testing.assert_array_equal(in_eta, 0.0)
    np.testing.assert_array_equal(in_lam, 0.0)


def test_probe_wins_only_with_regime_epistemics():
    totals = {}
    risks = {}
    for action in ("probe", "idle"):
        _, template, arrays = _probe_template(action)
        engine, result = _run(template, arrays)
        assert result.status is RunStatus.CONVERGED
        diagnostic = engine.hybrid_gfe_diagnostics(result.state)["emission"]
        totals[action] = diagnostic["gfe_lower_bound"]
        risks[action] = diagnostic["goal_risk"]

    assert totals["probe"] < totals["idle"]
    assert risks["probe"] > risks["idle"]


def test_mixed_and_flooding_surfaces_reach_the_same_root():
    _, template, arrays = _probe_template("probe")
    mixed_engine, mixed = _run(template, arrays, schedule="auto")
    flood_engine, flood = _run(template, arrays, schedule="flood")
    assert mixed.status is RunStatus.CONVERGED
    assert flood.status is RunStatus.CONVERGED
    np.testing.assert_allclose(
        mixed_engine.marginals(mixed.state)["s"],
        flood_engine.marginals(flood.state)["s"],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        mixed_engine.bfe(mixed.state), flood_engine.bfe(flood.state), atol=1e-9
    )


def test_exact_zero_selector_support_stays_exact():
    _, template, arrays = _probe_template("probe", selector=np.array([1.0, 0.0]))
    engine, result = _run(template, arrays)
    assert result.status is RunStatus.CONVERGED
    np.testing.assert_array_equal(engine.marginals(result.state)["s"], [1.0, 0.0])
    diagnostic = engine.hybrid_gfe_diagnostics(result.state)["emission"]
    np.testing.assert_array_equal(diagnostic["rho"], [1.0, 0.0])
    assert diagnostic["mi_regime_surrogate"] == pytest.approx(0.0, abs=1e-12)


def test_float32_k32_scale_converges():
    regimes = 32
    offsets = np.linspace(-2.0, 2.0, regimes)[:, None]
    matrix = np.linspace(0.2, 0.8, regimes)[:, None, None]
    noise = np.linspace(0.1, 0.8, regimes)[:, None, None]
    template, arrays = _linear_template(
        matrix=matrix,
        offset=offsets,
        noise=noise,
        state_mean=np.array([0.2]),
        state_cov=np.array([[0.7]]),
        selector=np.full(regimes, 1.0 / regimes),
        goal_mean=np.array([0.3]),
        goal_cov=np.array([[1.5]]),
    )
    arrays = {key: value.astype(np.float32) for key, value in arrays.items()}
    engine, result = _run(
        template, arrays, dtype=jnp.float32, iterations=250, tol=1e-5
    )
    assert result.status is RunStatus.CONVERGED
    diagnostic = engine.hybrid_gfe_diagnostics(result.state)["emission"]
    assert diagnostic["status"] == RunStatus.CONVERGED.value
    assert np.all(np.isfinite(diagnostic["rho"]))
    assert np.isclose(np.sum(diagnostic["rho"]), 1.0, atol=1e-6)
    assert np.isfinite(engine.bfe(result.state))


def test_improper_runtime_goal_is_loud():
    _, template, arrays = _probe_template()
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.init_state()
    group = engine.arena.hybrid_gfe_groups[0]
    params = dict(state.params)
    params[group.name + ":goal_W"] = -jnp.ones_like(
        params[group.name + ":goal_W"]
    )
    result = engine.run(_replace(state, params=params), 50, tol=1e-8)
    assert result.status is RunStatus.DIVERGED
    assert result.inner_status is RunStatus.DIVERGED
    assert not np.isfinite(engine.bfe(result.state))


def test_improper_runtime_state_is_loud():
    _, template, arrays = _probe_template()
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.init_state()
    state_group = next(
        group for group in engine.arena.gauss_groups if "state" in group.members
    )
    params = dict(state.params)
    params[state_group.name + ":W"] = jnp.zeros_like(
        params[state_group.name + ":W"]
    )
    result = engine.run(_replace(state, params=params), 50, tol=1e-8)
    assert result.status is RunStatus.DIVERGED
    assert not np.isfinite(engine.bfe(result.state))


def test_vmap_step_matches_sequential():
    _, template, arrays = _probe_template()
    engine = Engine(compile_template(template, arrays=arrays))
    states = [engine.init_state(), engine.init_state()]
    batched = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *states)

    def one(state):
        messages, eta, lam, ef_eta, _, _, *_ = engine._step(
            state.messages,
            state.g_eta,
            state.g_lam,
            state.g_obs,
            state.g_mask,
            state.ef_eta,
            state.ef_obs,
            state.ef_mask,
            state.params,
        )
        return _replace(
            state, messages=messages, g_eta=eta, g_lam=lam, ef_eta=ef_eta
        )

    batched_result = eqx.filter_vmap(one)(batched)
    sequential = [one(state) for state in states]
    for row, state in enumerate(sequential):
        np.testing.assert_allclose(batched_result.messages[row], state.messages)
        np.testing.assert_allclose(batched_result.g_eta[row], state.g_eta)
        np.testing.assert_allclose(batched_result.g_lam[row], state.g_lam)

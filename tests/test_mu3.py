"""Compiled/runtime contract for the functional Part-II μ3 parameter message."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp

from cffg import rules
from cffg.colony import Colony
from cffg.compiler import NEG, CompileError, compile_template
from cffg.core import ConstraintKind, ConstraintSpec, FactorSpec, Template, VariableSpec
from cffg.core.ir import BlobRef
from cffg.core.serialize import from_json, to_json
from cffg.engine import Engine, RunStatus, _replace


def _model(*, slices=1, parameter_message=True):
    variables = {}
    factors = {}
    constraints = []
    for i in range(slices):
        z, x = f"z{i}", f"x{i}"
        variables[z] = VariableSpec(z, 2)
        variables[x] = VariableSpec(x, 2)
        factors[f"prior{i}"] = FactorSpec(
            f"prior{i}", "categorical",
            edges={"out": z}, param_blobs={"statistic": "d"}
        )
        factors[f"obs{i}"] = FactorSpec(
            f"obs{i}", "matrix_cat",
            edges={"out": x, "in": z}, param_blobs={"matrix": "Ac"}
        )
        factors[f"goal{i}"] = FactorSpec(
            f"goal{i}", "categorical",
            edges={"out": x}, param_blobs={"statistic": f"c{i}"}
        )
        detail = [
            ("parameter_uncertainty", "dirichlet"),
            ("substituted_role", "out"),
        ]
        if parameter_message:
            detail.append(("parameter_message", "mu3_is_v1"))
        constraints.append(
            ConstraintSpec(
                ConstraintKind.P_SUBSTITUTION, f"obs{i}", tuple(sorted(detail))
            )
        )
    blobs = {
        "Ac": BlobRef("Ac", "Ac.npy"),
        "d": BlobRef("d", "d.npy"),
        **{f"c{i}": BlobRef(f"c{i}", f"c{i}.npy") for i in range(slices)},
    }
    return Template(
        name=f"mu3_{slices}",
        variables=variables,
        factors=factors,
        blobs=blobs,
        constraints=constraints,
    )


def _arrays(*, slices=1):
    return {
        "Ac": np.array([[2.2, 1.5], [1.8, 2.5]]),
        "d": np.array([0.4, 0.6]),
        **{
            f"c{i}": np.array([0.75 - 0.2 * i, 0.25 + 0.2 * i])
            for i in range(slices)
        },
    }


def _engine(*, slices=1, samples=4096, min_ess=0.01):
    arena = compile_template(_model(slices=slices), arrays=_arrays(slices=slices))
    return Engine(
        arena, mu3_samples=samples, mu3_min_ess_fraction=min_ess
    )


def test_compiler_serializes_evaluator_identity_and_ties_slices():
    template = _model(slices=2)
    back = from_json(to_json(template))
    arena = compile_template(back, arrays=_arrays(slices=2))
    assert len(arena.mu3_groups) == 1
    plan = arena.mu3_groups[0]
    assert plan.evaluator == "cffg.mu3_is.v1"
    assert plan.param_blob == "Ac"
    assert plan.members == ("obs0", "obs1")
    np.testing.assert_array_equal(plan.rows, [0, 1])

    bad = _model()
    bad.constraints[0] = ConstraintSpec(
        ConstraintKind.P_SUBSTITUTION,
        "obs0",
        (("parameter_message", "mu3_is_v1"), ("substituted_role", "out")),
    )
    with pytest.raises(CompileError, match="proper count-array proposal"):
        compile_template(bad, arrays=_arrays())
    with pytest.raises(ValueError, match="per-instance explicit PRNG keys"):
        Colony().spawn(template, _arrays(slices=2))


def test_explicit_key_replays_particles_weights_and_fixed_point_bitwise():
    engine = _engine(samples=2048)
    key = jax.random.PRNGKey(71)
    state_a, next_a = engine.prepare_mu3(engine.init_state(), key)
    state_b, next_b = engine.prepare_mu3(engine.init_state(), key)
    np.testing.assert_array_equal(next_a, next_b)
    plan = engine.arena.mu3_groups[0]
    np.testing.assert_array_equal(
        state_a.params[plan.name + ":samples"],
        state_b.params[plan.name + ":samples"],
    )

    result_a = engine.run(state_a, 30, tol=1e-9, harden_last=0)
    result_b = engine.run(state_b, 30, tol=1e-9, harden_last=0)
    assert result_a.status is result_b.status is RunStatus.CONVERGED
    for left, right in zip(
        jax.tree_util.tree_leaves(result_a.state),
        jax.tree_util.tree_leaves(result_b.state),
    ):
        np.testing.assert_array_equal(left, right)
    assert result_a.mu3_ess == result_b.mu3_ess
    assert result_a.mu3_log_normalizers == result_b.mu3_log_normalizers


def test_multiple_slices_add_functional_weights_and_parameter_ledger_once():
    engine = _engine(slices=2, samples=4096)
    state, _ = engine.prepare_mu3(engine.init_state(), jax.random.PRNGKey(9))
    result = engine.run(state, 40, tol=1e-9, harden_last=0)
    assert result.status is RunStatus.CONVERGED
    state = result.state
    plan = engine.arena.mu3_groups[0]
    group = engine.arena.gfe_groups[0]
    diag = engine.mu3_diagnostics(state)[plan.name]

    log_d = state.messages[
        engine.arena.message_indices(group.gather_z[plan.rows], 2)
    ]
    rows = jnp.asarray(plan.rows)
    zbars, _, statuses = rules.gfe_stationary_stats(
        log_d,
        state.params[group.name + ":Abar"][rows],
        state.params[group.name + ":hbar"][rows],
        state.params[group.name + ":c"][rows],
    )
    assert int(jnp.max(statuses)) == 0
    logw = rules.mu3_log_message(
        state.params[plan.name + ":samples"],
        zbars,
        state.params[group.name + ":c"][rows],
        state.params[group.name + ":Abar"][0],
    )
    expected_weights = jnp.exp(logw - logsumexp(logw))
    np.testing.assert_allclose(diag["weights"], expected_weights, atol=2e-9)

    assert diag["kl"] == pytest.approx(
        diag["expected_phi"] - diag["log_normalizer"], abs=2e-9
    )
    without_kl = dict(state.params)
    without_kl[plan.name + ":kl"] = jnp.asarray(
        0.0, dtype=state.messages.dtype
    )
    structural = engine.bfe(_replace(state, params=without_kl))
    assert engine.bfe(state) - structural == pytest.approx(diag["kl"], abs=2e-8)
    logd = jnp.log(jnp.asarray(_arrays(slices=2)["d"]))
    state_kl = jnp.sum(zbars * (jnp.log(zbars) - logd))
    # Complete D6b ledger: the shared parameter coordinate contributes
    # -log Z once; each independent state retains KL(q(z_t)||d_t).
    assert engine.bfe(state) == pytest.approx(
        -diag["log_normalizer"] + float(state_kl), abs=2e-5
    )


def test_proposal_refresh_and_stochastic_statuses_are_loud():
    engine = _engine(samples=512, min_ess=1.0)
    with pytest.raises(ValueError, match="prepare_mu3"):
        engine.run(engine.init_state(), 2, tol=1e-6)

    improper = engine.init_state()
    plan = engine.arena.mu3_groups[0]
    changed = dict(improper.params)
    changed[plan.count_key] = changed[plan.count_key].at[0, 0, 0].set(0.0)
    with pytest.raises(ValueError, match="proper Dirichlet"):
        engine.prepare_mu3(
            _replace(improper, params=changed), jax.random.PRNGKey(0)
        )

    state, _ = engine.prepare_mu3(engine.init_state(), jax.random.PRNGKey(4))
    exhausted = engine.run(state, 10, tol=1e-6, harden_last=0)
    assert exhausted.status is RunStatus.BUDGET_EXHAUSTED
    assert exhausted.mu3_statuses == (RunStatus.BUDGET_EXHAUSTED,)
    assert exhausted.mu3_ess[0] < engine.mu3_samples

    diverging = _engine(samples=128)
    state, _ = diverging.prepare_mu3(
        diverging.init_state(), jax.random.PRNGKey(5)
    )
    plan = diverging.arena.mu3_groups[0]
    group = diverging.arena.gfe_groups[0]
    params = dict(state.params)
    params[group.name + ":c"] = jnp.full_like(
        params[group.name + ":c"], NEG
    )
    collapsed = diverging.run(
        _replace(state, params=params), 10, tol=1e-6, harden_last=0
    )
    assert collapsed.status is RunStatus.DIVERGED
    assert collapsed.mu3_statuses == (RunStatus.DIVERGED,)


def test_base_posterior_is_runtime_input_and_burn_in_only_averages_converged_epochs():
    engine = _engine(samples=1024)
    state = engine.init_state()
    plan = engine.arena.mu3_groups[0]
    posterior = jnp.asarray([[8.0, 1.5], [2.0, 6.5]])
    params = dict(state.params)
    params[plan.count_key] = params[plan.count_key].at[plan.rows].set(posterior)
    state = _replace(state, params=params)
    prepared, _ = engine.prepare_mu3(state, jax.random.PRNGKey(12))
    sample_mean = prepared.params[plan.name + ":samples"].mean(axis=0)
    np.testing.assert_allclose(
        sample_mean,
        posterior / posterior.sum(axis=0, keepdims=True),
        atol=0.035,
    )

    scored = engine.score_mu3_epochs(
        state,
        key=jax.random.PRNGKey(20),
        epochs=2,
        burn_in=1,
        n_iters=30,
        tol=1e-8,
    )
    assert scored.status is RunStatus.CONVERGED
    assert scored.policy_score == pytest.approx(scored.epoch_scores[1])
    for run, score in zip(scored.runs, scored.epoch_scores):
        diag = engine.mu3_diagnostics(run.state)[plan.name]
        group = engine.arena.gfe_groups[0]
        log_d = run.state.messages[
            engine.arena.message_indices(group.gather_z[plan.rows], 2)
        ]
        rows = jnp.asarray(plan.rows)
        zbar, _, _ = rules.gfe_stationary_stats(
            log_d,
            run.state.params[group.name + ":Abar"][rows],
            run.state.params[group.name + ":hbar"][rows],
            run.state.params[group.name + ":c"][rows],
        )
        logd = jnp.log(jnp.asarray(_arrays()["d"]))
        oracle = -diag["log_normalizer"] + float(
            jnp.sum(zbar * (jnp.log(zbar) - logd))
        )
        assert score == pytest.approx(oracle, abs=2e-5)


def test_production_rule_float32_at_uncomfortable_scale_and_exact_zero_support():
    with jax.enable_x64(False):
        m, k, samples = 20, 40, 256
        alpha = jnp.full((m, k), 0.05, dtype=jnp.float32)
        column_keys = jax.random.split(jax.random.PRNGKey(8), k)
        by_column = jax.vmap(
            lambda key, concentration: jax.random.dirichlet(
                key, concentration, shape=(samples,)
            )
        )(column_keys, alpha.T)
        particles = jnp.transpose(by_column, (1, 2, 0))
        z = jnp.full((1, k), 1 / k, dtype=jnp.float32)
        c = jnp.full((1, m), -jnp.log(jnp.asarray(m, dtype=jnp.float32)))
        abar = alpha / alpha.sum(axis=0, keepdims=True)
        result = rules.mu3_importance_update(particles, z, c, abar)
        assert int(result[-1]) == 0
        for value in result[:-1]:
            assert bool(jnp.all(jnp.isfinite(value)))
        assert result[0].dtype == jnp.float32

    boundary = jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]])
    score = rules.mu3_log_message(
        boundary,
        jnp.asarray([[1.0, 0.0]]),
        jnp.asarray([[0.0, NEG]]),
        boundary[0],
    )
    np.testing.assert_array_equal(score, [0.0])

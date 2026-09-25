"""Part-II Fig. 5 mu1(c): learned Dirichlet goals in GFE composites."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import digamma, gammaln

from cffg import rules
from cffg.agent import RollingAgent
from cffg.bundle import load_bundle, save_bundle
from cffg.compiler import CompileError, compile_template
from cffg.core import (
    BlobRef,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    Template,
    VariableSpec,
)
from cffg.core.serialize import from_json, structure_hash, to_json
from cffg.engine import Engine, RunStatus
from cffg.models.gridworld import build_gridworld
from cffg.models.rollout import build_rollout


def _model(*, slices=1, observations=3, states=2):
    variables = {}
    factors = {}
    constraints = []
    for index in range(slices):
        z = f"z{index}"
        x = f"x{index}"
        c_edge = f"cparam{index}"
        variables[z] = VariableSpec(z, states)
        variables[x] = VariableSpec(x, observations)
        variables[c_edge] = VariableSpec(c_edge, observations)
        factors[f"state{index}"] = FactorSpec(
            f"state{index}",
            "categorical",
            edges={"out": z},
            param_blobs={"statistic": "d"},
        )
        factors[f"obs{index}"] = FactorSpec(
            f"obs{index}",
            "matrix_cat",
            edges={"out": x, "in": z},
            param_blobs={"matrix": "A"},
        )
        factors[f"goal{index}"] = FactorSpec(
            f"goal{index}",
            "categorical",
            edges={"out": x},
            param_edges={"statistic": c_edge},
        )
        factors[f"cprior{index}"] = FactorSpec(
            f"cprior{index}",
            "dirichlet",
            edges={"out": c_edge},
            param_blobs={"counts": "Cc"},
        )
        constraints.extend(
            [
                ConstraintSpec(ConstraintKind.MEAN_FIELD, f"goal{index}"),
                ConstraintSpec(
                    ConstraintKind.P_SUBSTITUTION,
                    f"obs{index}",
                    (
                        ("goal_uncertainty", "dirichlet"),
                        ("substituted_role", "out"),
                    ),
                ),
            ]
        )
    return Template(
        f"goal-hyper-{slices}",
        variables=variables,
        factors=factors,
        constraints=constraints,
        blobs={
            "A": BlobRef("A", "A.npy"),
            "Cc": BlobRef("Cc", "Cc.npy"),
            "d": BlobRef("d", "d.npy"),
        },
    )


def _arrays(*, observations=3, states=2):
    rng = np.random.default_rng(4)
    matrix = rng.uniform(0.1, 1.0, size=(observations, states))
    matrix /= matrix.sum(axis=0, keepdims=True)
    return {
        "A": matrix,
        "Cc": np.linspace(0.7, 1.4, observations),
        "d": np.linspace(1.0, 2.0, states)
        / np.linspace(1.0, 2.0, states).sum(),
    }


def _log_beta(alpha):
    alpha = jnp.asarray(alpha)
    return jnp.sum(gammaln(alpha)) - gammaln(jnp.sum(alpha))


def test_mu1_shared_goal_posterior_and_complete_fe_match_direct_algebra():
    arrays = _arrays()
    engine = Engine(
        compile_template(_model(slices=2), arrays=arrays)
    )
    result = engine.run(
        engine.init_state(), 100, tol=1e-11, harden_last=0
    )
    assert result.status is RunStatus.CONVERGED
    state = result.state

    marginals = engine.marginals(state)
    predicted = np.stack([marginals["x0"], marginals["x1"]])
    expected_alpha = arrays["Cc"] + predicted.sum(axis=0)
    group = engine.arena.gfe_groups[0]
    raw_mu1 = state.messages[
        engine.arena.message_indices(
            group.goal_count_scatter, group.cards[0]
        )
    ]
    np.testing.assert_allclose(
        raw_mu1,
        np.broadcast_to(predicted.sum(axis=0), raw_mu1.shape),
        atol=3e-6,
    )
    goals = engine.goal_marginals(state)
    assert set(goals) == {"goal0", "goal1"}
    for posterior in goals.values():
        assert posterior.proper
        np.testing.assert_allclose(posterior.alpha, expected_alpha, atol=3e-6)
    np.testing.assert_allclose(
        engine.goal_prior_updates(state)["Cc"], expected_alpha, atol=3e-6
    )

    alpha = jnp.asarray(expected_alpha)
    alpha0 = jnp.asarray(arrays["Cc"])
    np.testing.assert_allclose(
        rules.goal_dirichlet_energy(
            alpha, alpha0, jnp.asarray(predicted.sum(axis=0))
        ),
        _log_beta(alpha0) - _log_beta(alpha),
        atol=3e-6,
    )
    elog = digamma(alpha) - digamma(alpha.sum())
    matrix = jnp.asarray(arrays["A"])
    h = -jnp.sum(jnp.where(matrix > 0, matrix * jnp.log(matrix), 0.0), axis=0)
    direct = _log_beta(alpha0) - _log_beta(alpha)
    for index in range(2):
        z = jnp.asarray(marginals[f"z{index}"])
        qx = matrix @ z
        rho = matrix.T @ (elog - jnp.log(qx)) - h
        structural_without_goal = (
            -z @ rho
            + jnp.sum(z * jnp.log(z))
            + qx @ elog
        )
        state_cross_entropy = -z @ jnp.log(jnp.asarray(arrays["d"]))
        direct = direct + structural_without_goal + state_cross_entropy
    np.testing.assert_allclose(engine.bfe(state), direct, atol=5e-6)


def test_sweeps_replace_mu1_and_explicit_rollover_adds_one_new_trial():
    arrays = _arrays()
    engine = Engine(compile_template(_model(), arrays=arrays))
    first = engine.run(
        engine.init_state(), 100, tol=1e-11, harden_last=0
    ).require_converged()
    first_alpha = engine.goal_prior_updates(first)["Cc"]

    repeated = engine.run(
        first, 30, tol=1e-11, harden_last=0
    ).require_converged()
    np.testing.assert_allclose(
        engine.goal_prior_updates(repeated)["Cc"], first_alpha, atol=2e-6
    )
    with pytest.raises(ValueError, match="fresh inference state"):
        engine.with_goal_priors(first, {"Cc": first_alpha})

    second_start = engine.with_goal_priors(
        engine.init_state(), {"Cc": first_alpha}
    )
    second = engine.run(
        second_start, 100, tol=1e-11, harden_last=0
    ).require_converged()
    second_alpha = engine.goal_prior_updates(second)["Cc"]
    np.testing.assert_allclose(
        second_alpha,
        first_alpha + engine.marginals(second)["x0"],
        atol=3e-6,
    )


def test_rollover_changes_next_policy_in_predicted_direction():
    d = np.array([0.5, 0.5])
    matrix = np.eye(2)
    transitions = np.array(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ]
    )
    template, arrays = build_rollout(
        "goal-policy",
        steps=1,
        d=d,
        A=matrix,
        c=np.array([0.5, 0.5]),
        c_counts=np.array([2.0, 1.0]),
        e=np.array([0.5, 0.5]),
        Bs=transitions,
        learned_c=True,
    )
    engine = Engine(compile_template(template, arrays=arrays))
    first = engine.run(
        engine.init_state(), 100, tol=1e-10, harden_last=0
    ).require_converged()
    first_policy = engine.marginals(first)["u1"]
    rolled = engine.goal_prior_updates(first)
    assert rolled["Cc"][0] / rolled["Cc"].sum() < 2.0 / 3.0

    second = engine.run(
        engine.with_goal_priors(engine.init_state(), rolled),
        100,
        tol=1e-10,
        harden_last=0,
    ).require_converged()
    second_policy = engine.marginals(second)["u1"]
    assert second_policy[0] < first_policy[0]


def test_serialization_bundle_and_compiler_contract(tmp_path):
    template = _model(slices=2)
    arrays = _arrays()
    restored = from_json(to_json(template))
    assert structure_hash(restored) == structure_hash(template)
    arena = compile_template(restored, arrays=arrays)
    group = arena.gfe_groups[0]
    assert group.goal_learned
    assert group.goal_tie_groups == ((0, 1),)
    assert group.goal_prior_blobs == ("Cc", "Cc")

    path = tmp_path / "goal.cffgz"
    save_bundle(path, template, arrays)
    bundled, bundled_arrays, _ = load_bundle(path)
    bundled_arena = compile_template(bundled, arrays=bundled_arrays)
    assert bundled_arena.gfe_groups[0].goal_tie_groups == ((0, 1),)

    missing_detail = _model()
    missing_detail.constraints[-1] = ConstraintSpec(
        ConstraintKind.P_SUBSTITUTION,
        "obs0",
        (("substituted_role", "out"),),
    )
    with pytest.raises(CompileError, match="must be declared together"):
        compile_template(missing_detail, arrays=arrays)

    bad = _arrays()
    bad["Cc"] = np.array([0.0, 1.0, 1.0])
    with pytest.raises(CompileError, match="finite and > 0"):
        compile_template(_model(), arrays=bad)


def test_float32_sparse_zero_mass_and_uncomfortable_observation_scale():
    with jax.enable_x64(False):
        observations, states = 64, 4
        arrays = _arrays(observations=observations, states=states)
        # Exact-zero observation support in row zero means its fractional mu1
        # count is exactly zero; a small proper prior remains finite.
        arrays["A"][0] = 0.0
        arrays["A"] /= arrays["A"].sum(axis=0, keepdims=True)
        arrays["Cc"] = np.linspace(0.01, 0.2, observations).astype(np.float32)
        arrays["Cc"][0] = np.finfo(np.float32).tiny * 16
        engine = Engine(
            compile_template(
                _model(observations=observations, states=states),
                arrays=arrays,
            )
        )
        result = engine.run(
            engine.init_state(), 100, tol=1e-6, harden_last=0
        )
        assert result.status is RunStatus.CONVERGED
        posterior = engine.goal_prior_updates(result.state)["Cc"]
        assert posterior.dtype == np.float32
        assert np.isfinite(posterior).all()
        assert posterior[0] == pytest.approx(arrays["Cc"][0])
        assert np.isfinite(engine.bfe(result.state))


@pytest.mark.slow
def test_rolling_agent_rolls_once_at_episode_boundary_and_checkpoints_pending(tmp_path):
    builder = functools.partial(build_gridworld, k=4)
    agent = RollingAgent(
        builder,
        horizon=2,
        learn_goal=True,
        initial_goal_counts=np.array([1.0, 1.0, 1.0, 2.0]),
        n_sweeps=100,
    )
    before = agent.goal_counts
    agent.act()
    np.testing.assert_array_equal(agent.goal_counts, before)

    path = tmp_path / "goal-agent.npz"
    agent.checkpoint(path)
    resumed = RollingAgent(
        builder,
        horizon=2,
        learn_goal=True,
        initial_goal_counts=np.ones(4),
        n_sweeps=100,
    )
    resumed.restore(path)
    np.testing.assert_array_equal(resumed.goal_counts, agent.goal_counts)

    agent.reset_episode()
    resumed.reset_episode()
    np.testing.assert_array_equal(resumed.goal_counts, agent.goal_counts)
    after = agent.goal_counts
    assert not np.array_equal(after, before)
    agent.reset_episode()
    np.testing.assert_array_equal(agent.goal_counts, after)

"""Serialized model/runtime bindings for rolling boundaries (cff-nhxf)."""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

from cffg.agent import RollingAgent
from cffg.bundle import load_bundle, save_bundle
from cffg.core.ir import (
    AgentSpec,
    BeliefBinding,
    BlobRef,
    BoundaryFactorization,
    ConstraintKind,
    ConstraintSpec,
    FactorSpec,
    LearnableSpec,
    PortRole,
    PortSpec,
    Template,
    VariableSpec,
)
from cffg.core.serialize import from_json, structure_hash, to_json
from cffg.core.validate import ModelError, validate
from cffg.models.rollout import build_rollout


def _shift_stack(k: int, dtype: np.dtype) -> np.ndarray:
    matrices = np.zeros((2, k, k), dtype=dtype)
    for z in range(k):
        matrices[0, max(z - 1, 0), z] = 1
        matrices[1, min(z + 1, k - 1), z] = 1
    return matrices


def _two_block_builder(
    *,
    past: int,
    steps: int,
    learn: tuple[str, ...] = (),
    cards: tuple[int, int] = (3, 2),
    dtype: np.dtype = np.float64,
    **_kwargs,
) -> tuple[Template, dict[str, np.ndarray]]:
    """A deliberately non-rollout-named model for the runtime contract.

    Position is controlled and observed. Mode evolves independently; carrying
    both posteriors proves the runtime has no single-chain naming dependency.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    ka, kb = cards
    t = Template("two-block-agent")
    t.blobs = {
        name: BlobRef(name, f"{name}.npy")
        for name in (
            "initial_position",
            "initial_mode",
            "motion",
            "sensor",
            "habit",
            "mode_motion",
            "position_goal",
            "mode_goal",
        )
    }
    t.variables = {
        "position_now": VariableSpec("position_now", ka),
        "position_next": VariableSpec("position_next", ka),
        "choice": VariableSpec("choice", 2),
        "reading": VariableSpec("reading", ka),
        "mode_now": VariableSpec("mode_now", kb),
        "mode_next": VariableSpec("mode_next", kb),
    }
    t.factors = {
        "position_boundary": FactorSpec(
            "position_boundary",
            "categorical",
            edges={"out": "position_now"},
            param_blobs={"statistic": "initial_position"},
        ),
        "advance_position": FactorSpec(
            "advance_position",
            "transition_mixture",
            edges={
                "out": "position_next",
                "in": "position_now",
                "selector": "choice",
            },
            param_blobs={"matrices": "motion"},
        ),
        "read_position": FactorSpec(
            "read_position",
            "matrix_cat",
            edges={"out": "reading", "in": "position_next"},
            param_blobs={"matrix": "sensor"},
        ),
        "mode_boundary": FactorSpec(
            "mode_boundary",
            "categorical",
            edges={"out": "mode_now"},
            param_blobs={"statistic": "initial_mode"},
        ),
        "advance_mode": FactorSpec(
            "advance_mode",
            "matrix_cat",
            edges={"out": "mode_next", "in": "mode_now"},
            param_blobs={"matrix": "mode_motion"},
        ),
        "prefer_mode": FactorSpec(
            "prefer_mode",
            "categorical",
            edges={"out": "mode_next"},
            param_blobs={"statistic": "mode_goal"},
        ),
    }
    if past:
        t.ports = {
            "execute": PortSpec("execute", "choice", PortRole.DATA),
            "measure": PortSpec("measure", "reading", PortRole.DATA),
        }
    else:
        t.factors |= {
            "habitual_choice": FactorSpec(
                "habitual_choice",
                "categorical",
                edges={"out": "choice"},
                param_blobs={"statistic": "habit"},
            ),
            "prefer_position": FactorSpec(
                "prefer_position",
                "categorical",
                edges={"out": "reading"},
                param_blobs={"statistic": "position_goal"},
            ),
        }

    owned = {
        "position_start": ("initial_position", "position_boundary"),
        "mode_start": ("initial_mode", "mode_boundary"),
    }
    for name in learn:
        blob, _factor = owned[name]
        t.learnables[name] = LearnableSpec(name, "dirichlet", (blob,))

    t.agent = AgentSpec(
        action="choice",
        action_port="execute" if past else None,
        observation_port="measure" if past else None,
        beliefs=(
            BeliefBinding(
                "position",
                ("position",),
                "position_boundary",
                "position_next",
                episode_learnable=(
                    "position_start" if "position_start" in learn else None
                ),
            ),
            BeliefBinding(
                "mode",
                ("mode",),
                "mode_boundary",
                "mode_next",
                episode_learnable="mode_start" if "mode_start" in learn else None,
            ),
        ),
    )

    initial_position = np.zeros(ka, dtype=dtype)
    initial_position[0] = 1
    initial_mode = np.zeros(kb, dtype=dtype)
    initial_mode[-1] = 1
    position_goal = np.full(ka, 1 / ka, dtype=dtype)
    position_goal[-1] = 2
    position_goal /= position_goal.sum()
    return t, {
        "initial_position": initial_position,
        "initial_mode": initial_mode,
        "motion": _shift_stack(ka, dtype),
        "sensor": np.eye(ka, dtype=dtype),
        "habit": np.full(2, 0.5, dtype=dtype),
        "mode_motion": np.eye(kb, dtype=dtype),
        "position_goal": position_goal,
        "mode_goal": np.full(kb, 1 / kb, dtype=dtype),
    }


def _joint_builder(
    *, past: int, steps: int, **kwargs
) -> tuple[Template, dict[str, np.ndarray]]:
    """A flat categorical carrier for the correlated pair (a, b)."""
    kwargs.pop("learned_a", None)
    kwargs.pop("learned_c", None)
    kwargs.pop("learn", None)
    template, arrays = build_rollout(
        "joint-correlated",
        past=past,
        steps=steps,
        d=np.array([0.5, 0.0, 0.0, 0.5]),
        # Read whether the two binary blocks agree. Observation 0 preserves
        # q(00)=q(11)=1/2, whose separate marginals would spuriously put
        # 1/4 on 01 and 10.
        A=np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]),
        c=np.array([0.5, 0.5]),
        e=np.array([0.5, 0.5]),
        Bs=np.stack([np.eye(4), np.eye(4)]),
        gfe=False,
    )
    (binding,) = template.agent.beliefs
    template.agent = dataclasses.replace(
        template.agent,
        beliefs=(
            dataclasses.replace(
                binding,
                name="correlated_pair",
                blocks=("position", "mode"),
                factorization=BoundaryFactorization.JOINT,
            ),
        ),
    )
    return template, arrays


def test_agent_spec_json_bundle_and_hash_round_trip(tmp_path):
    template, arrays = build_rollout(
        "bound",
        steps=2,
        d=np.array([0.7, 0.3]),
        A=np.eye(2),
        c=np.array([0.4, 0.6]),
        e=np.array([0.5, 0.5]),
        Bs=np.stack([np.eye(2), np.eye(2)]),
        learn=("d",),
    )
    restored = from_json(to_json(template))
    assert restored.agent == template.agent
    assert structure_hash(restored) == structure_hash(template)

    changed = from_json(to_json(template))
    changed.agent = dataclasses.replace(changed.agent, action="u2")
    assert structure_hash(changed) != structure_hash(template)

    path = tmp_path / "agent.cffgz"
    save_bundle(path, template, arrays)
    bundled, bundled_arrays, _manifest = load_bundle(path)
    assert bundled.agent == template.agent
    assert bundled.learnables == template.learnables
    assert structure_hash(bundled) == structure_hash(template)
    assert set(bundled_arrays) == set(arrays)


def test_model_local_binding_errors_are_validation_errors():
    template, _ = _two_block_builder(past=1, steps=1)

    bad_action = from_json(to_json(template))
    bad_action.agent = dataclasses.replace(bad_action.agent, action="missing")
    with pytest.raises(ModelError, match="action readout: unknown variable 'missing'"):
        validate(bad_action)

    bad_prior = from_json(to_json(template))
    beliefs = list(bad_prior.agent.beliefs)
    index = next(i for i, belief in enumerate(beliefs) if belief.name == "position")
    beliefs[index] = dataclasses.replace(
        beliefs[index], prior="advance_position"
    )
    bad_prior.agent = dataclasses.replace(bad_prior.agent, beliefs=tuple(beliefs))
    with pytest.raises(ModelError, match="must be a categorical factor"):
        validate(bad_prior)

    bad_domain = from_json(to_json(template))
    beliefs = list(bad_domain.agent.beliefs)
    index = next(i for i, belief in enumerate(beliefs) if belief.name == "position")
    beliefs[index] = dataclasses.replace(beliefs[index], posterior="choice")
    bad_domain.agent = dataclasses.replace(bad_domain.agent, beliefs=tuple(beliefs))
    with pytest.raises(ModelError, match="prior domain .* != filter posterior"):
        validate(bad_domain)

    bad_learnable = from_json(to_json(template))
    beliefs = list(bad_learnable.agent.beliefs)
    index = next(i for i, belief in enumerate(beliefs) if belief.name == "position")
    beliefs[index] = dataclasses.replace(
        beliefs[index], episode_learnable="not-declared"
    )
    bad_learnable.agent = dataclasses.replace(
        bad_learnable.agent, beliefs=tuple(beliefs)
    )
    with pytest.raises(ModelError, match="unknown episode learnable"):
        validate(bad_learnable)


def test_two_block_agent_updates_resets_and_checkpoints_exactly(tmp_path):
    learn = ("position_start", "mode_start")
    priors = {
        "position_start": np.ones(3),
        "mode_start": np.ones(2),
    }

    def make() -> RollingAgent:
        return RollingAgent(
            _two_block_builder,
            horizon=1,
            learn=learn,
            priors=priors,
            n_sweeps=100,
            select="sample",
            seed=7,
        )

    agent = make()
    assert set(agent.beliefs) == {"position", "mode"}
    with pytest.raises(AttributeError, match="single boundary"):
        _ = agent.belief

    before = {
        name: agent.learnables.params[name]["alpha"].copy() for name in learn
    }
    result = agent.observe(1, 1)
    assert set(result) == {"position", "mode"}
    assert np.argmax(agent.beliefs["position"]) == 1
    for name in learn:
        added = agent.learnables.params[name]["alpha"] - before[name]
        np.testing.assert_allclose(added.sum(), 1.0, atol=1e-9)

    path = tmp_path / "two-block.npz"
    agent.checkpoint(path)
    resumed = make()
    resumed.restore(path)
    for name in agent.beliefs:
        np.testing.assert_array_equal(resumed.beliefs[name], agent.beliefs[name])
    for name in learn:
        np.testing.assert_array_equal(
            resumed.learnables.params[name]["alpha"],
            agent.learnables.params[name]["alpha"],
        )
    assert resumed.act() == agent.act()  # checkpoint includes the PRNG stream

    reset = agent.reset_episode()
    resumed_reset = resumed.reset_episode()
    for name in reset:
        np.testing.assert_array_equal(resumed_reset[name], reset[name])


def test_boundary_factorization_is_explicit_and_preserved():
    template, _ = _two_block_builder(past=1, steps=1)
    template.constraints += [
        ConstraintSpec(ConstraintKind.MEAN_FIELD, "read_position"),
        ConstraintSpec(ConstraintKind.MEAN_FIELD, "prefer_mode"),
    ]
    position, mode = template.agent.beliefs
    template.agent = dataclasses.replace(
        template.agent,
        beliefs=(
            dataclasses.replace(
                position,
                factorization=BoundaryFactorization.MEAN_FIELD,
                constraints=("read_position",),
            ),
            dataclasses.replace(
                mode,
                factorization=BoundaryFactorization.MEAN_FIELD,
                constraints=("prefer_mode",),
            ),
        ),
    )
    validate(template)
    restored = from_json(to_json(template))
    assert all(
        b.factorization is BoundaryFactorization.MEAN_FIELD
        for b in restored.agent.beliefs
    )

    missing = from_json(to_json(template))
    missing.constraints = [
        c for c in missing.constraints if c.target != "prefer_mode"
    ]
    with pytest.raises(
        ModelError, match="'prefer_mode'.*not an explicit Template mean_field"
    ):
        validate(missing)

    irrelevant = from_json(to_json(template))
    beliefs = list(irrelevant.agent.beliefs)
    index = next(i for i, belief in enumerate(beliefs) if belief.name == "mode")
    beliefs[index] = dataclasses.replace(
        beliefs[index], constraints=("read_position",)
    )
    irrelevant.agent = dataclasses.replace(
        irrelevant.agent, beliefs=tuple(beliefs)
    )
    with pytest.raises(ModelError, match="does not touch filter posterior"):
        validate(irrelevant)

    joint, arrays = _joint_builder(past=1, steps=1)
    validate(joint)
    assert joint.agent.beliefs[0].blocks == ("position", "mode")
    assert arrays["d"].shape == (4,)

    agent = RollingAgent(_joint_builder, horizon=1, n_sweeps=40)
    posterior = agent.observe(0, 0)["correlated_pair"]
    np.testing.assert_allclose(posterior, [0.5, 0.0, 0.0, 0.5], atol=1e-8)
    separate_product = np.outer(
        posterior.reshape(2, 2).sum(axis=1),
        posterior.reshape(2, 2).sum(axis=0),
    )
    assert separate_product[0, 1] > 0 and posterior[1] == 0


def test_exact_zero_float32_and_32_by_4_boundaries_stay_factored():
    with jax.enable_x64(False):
        agent = RollingAgent(
            lambda **kwargs: _two_block_builder(
                cards=(32, 4), dtype=np.float32, **kwargs
            ),
            horizon=1,
            learn=("position_start", "mode_start"),
            priors={
                "position_start": np.ones(32, dtype=np.float32),
                "mode_start": np.ones(4, dtype=np.float32),
            },
            n_sweeps=40,
        )
        assert agent.beliefs["position"].shape == (32,)
        assert agent.beliefs["mode"].shape == (4,)
        assert all(value.size != 32 * 4 for value in agent.beliefs.values())
        assert agent.beliefs["position"].dtype == np.float32
        assert np.count_nonzero(agent.beliefs["position"]) == 1
        result = agent.observe(1, 1)
        for belief in result.values():
            assert np.all(np.isfinite(belief))
            np.testing.assert_allclose(belief.sum(), 1.0, rtol=1e-5)

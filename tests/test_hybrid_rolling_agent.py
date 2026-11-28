"""Known-selector control, conjugate feedback and transactional restart."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
import subprocess
import sys
from textwrap import dedent

import jax
import numpy as np
import pytest

from cffg import Engine, FailureDiagnostic, RollingAgent, RunStatus, compile_template, conj, save_bundle
from cffg.core import (
    BlobRef, FactorSpec, HybridFilterSpec, HybridPlanSpec, LearnableSpec, ModelError,
    PortRole, PortSpec, Template, VariableSpec, validate,
)
from cffg.core.compose import scoped
from cffg.core.serialize import from_json, structure_hash, to_json
from cffg.learn import Learnables
from cffg.params import set_params
from cffg.runtime_types import InferenceError, RunResult


INITIAL_COV = np.array([[1., .6], [.6, 1.]])
VARIANCE = .3
SENSOR = np.array([[1., 0.]])
OFFSETS = np.array([[.5, 0.], [-.5, .25]])
NOISE = np.eye(2) * .2
COMMANDS = ("left", "right")


def window(plan, *, habit=(1., 0.), horizon=1, learning=True):
    model = Template("preview" if plan else "feedback")
    arrays = {
        "m": np.zeros(2), "P": INITIAL_COV, "A": np.broadcast_to(np.eye(2), (2, 2, 2)),
        "b": OFFSETS, "Q": np.broadcast_to(NOISE, (2, 2, 2)),
        "H": SENSOR, "offset": np.zeros(1), "R": np.array([[VARIANCE]]),
    }
    model.variables["x0"] = VariableSpec("x0", dim=2)
    model.factors["prior"] = FactorSpec("prior", "gaussian", {"out": "x0"},
                                      param_blobs={"mean": "m", "cov": "P"})
    for step in range(horizon):
        following, action = f"x{step+1}", f"a{step}"
        model.variables[following] = VariableSpec(following, dim=2)
        model.variables[action] = VariableSpec(action, cardinality=2)
        motion = f"motion{step}"
        model.factors[motion] = FactorSpec(
            motion, "gaussian_mixture_linear",
            {"in": f"x{step}", "out": following, "selector": action},
            param_blobs={"matrix": "A", "offset": "b", "cov": "Q"},
        )
        if plan:
            arrays["habit"] = np.asarray(habit)
            choice = f"habit{step}"
            model.factors[choice] = FactorSpec(choice, "categorical", {"out": action},
                                               param_blobs={"statistic": "habit"})
    model.variables["y"] = VariableSpec("y", dim=1)
    model.factors["readout"] = FactorSpec(
        "readout", "gaussian_linear", {"in": f"x{horizon}", "out": "y"},
        param_blobs={"matrix": "H", "offset": "offset", "cov": "R"},
    )
    coordinates = ("x", "velocity")
    if plan:
        model.ports["goal"] = PortSpec("goal", "y", PortRole.DATA)
        model.agent = HybridPlanSpec(coordinates, "prior", COMMANDS, "a0", ("goal",))
    else:
        model.ports["action"] = PortSpec("action", "a0", PortRole.DATA)
        model.ports["sensor"] = PortSpec("sensor", "y", PortRole.DATA)
        model.agent = HybridFilterSpec(coordinates, "prior", COMMANDS, "x1", "motion0", "action", "sensor")
    if learning:
        sources = () if plan else ("motion0",)
        model.learnables["motion"] = LearnableSpec("motion", "component_mniw", ("A", "b", "Q"), sources)
    model.blobs = {name: BlobRef(name, name + ".npy") for name in arrays}
    return model, arrays


def pair(root, *, habit=(1., 0.), horizon=1, learning=True, prior_shift=0.):
    paths = root / "plan.cffgz", root / "filter.cffgz"
    priors = [
        conj.MNIW(np.column_stack((np.eye(2), b + prior_shift)), np.eye(3), 5 * NOISE, 5.)
        for b in OFFSETS
    ]
    for plan, path in zip((True, False), paths, strict=True):
        model, arrays = window(plan, habit=habit, horizon=horizon if plan else 1, learning=learning)
        learner = None
        if learning:
            engine = Engine(compile_template(model, arrays=arrays))
            learner = Learnables(engine, priors={"motion": priors})
        save_bundle(path, model, arrays, learnables=learner)
    return paths, priors


def manual(engine, state_prior, data, learner, *, budget=100):
    state = engine.init_state()
    mean, cov = state_prior
    params = set_params(engine, learner.write(engine, state.params), {
        "prior": {"mean": mean, "cov": cov},
    })
    result = engine.run(engine.bind_data(replace(state, params=params), data), budget, tol=1e-7)
    result.require_converged()
    return result


def analytic_update(mean, cov, arm, observed):
    next_mean = mean + OFFSETS[arm]
    next_cov = cov + NOISE
    joint_mean = np.r_[next_mean, mean]
    joint_cov = np.block([[next_cov, cov], [cov, cov]])
    sensor = np.array([[1., 0., 0., 0.]])
    link = joint_cov @ sensor.T
    variance = (sensor @ link)[0, 0] + VARIANCE
    joint_mean += link[:, 0] * (observed - next_mean[0]) / variance
    joint_cov -= link @ link.T / variance
    return joint_mean, joint_cov


def assert_payload_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


def test_hybrid_declarations_round_trip_and_reject_hidden_filter_state():
    for plan in (True, False):
        model, _ = window(plan)
        assert from_json(to_json(model)).agent == model.agent
        expanded = scoped(model, "trajectory")
        validate(expanded)
        assert expanded.agent.prior == "trajectory/prior"
        assert expanded.agent.commands == COMMANDS
        candidate = replace(model.agent, commands=COMMANDS[::-1])
        assert structure_hash(replace(model, agent=candidate)) != structure_hash(model)
        if not plan:
            assert expanded.agent.selector_port == "trajectory/action"
            with pytest.raises(ModelError, match="categorical DATA"):
                validate(replace(model, agent=replace(model.agent, selector_port="sensor")))
            model.variables["hidden"] = VariableSpec("hidden", cardinality=2)
            model.factors["regime"] = FactorSpec("regime", "categorical", {"out": "hidden"},
                                               param_blobs={"statistic": "habit"})
            model.blobs["habit"] = BlobRef("habit", "habit.npy")
            model.ports["external"] = PortSpec("external", "hidden", PortRole.GLUE)
            with pytest.raises(ModelError, match="cannot carry a hidden regime"):
                validate(model)


@pytest.mark.parametrize("x64", [False, pytest.param(True, marks=pytest.mark.slow)])
def test_zero_support_override_matches_manual_kalman_and_selected_arm_learning(tmp_path, x64):
    with jax.enable_x64(x64):
        paths, priors = pair(tmp_path)
        agent = RollingAgent.from_gaussian_bundles(*paths)
        previous = agent.belief
        before = agent.learnables.state_dict()
        preview = {"goal": [1.]}
        expected_plan = manual(agent.plan, previous, preview, agent.learnables)
        proposal = agent.propose(preview)
        assert proposal.command == 0 and proposal.status is RunStatus.CONVERGED
        np.testing.assert_array_equal(proposal.probabilities, [1., 0.])
        np.testing.assert_array_equal(proposal.probabilities, agent.plan.marginals(expected_plan.state)["a0"])
        assert proposal.free_energy == expected_plan.free_energies[-1]
        assert_payload_equal(before, agent.learnables.state_dict())

        expected_filter = manual(agent.filter, previous, {"action": 1, "sensor": [.5]}, agent.learnables)
        if x64:
            flood = Engine(agent.filter.arena, schedule="flood")
            flooded = manual(flood, previous, {"action": 1, "sensor": [.5]}, agent.learnables)
            for scheduled, synchronous in zip(
                agent.filter.gaussian_marginal(expected_filter.state, "x1"),
                flood.gaussian_marginal(flooded.state, "x1"), strict=True,
            ):
                np.testing.assert_allclose(scheduled, synchronous, atol=1e-10)
        actual = agent.observe(1, [.5], tick=proposal.tick)
        assert agent.tick == 1
        joint_mean, joint_cov = analytic_update(*previous, 1, .5)
        expected = priors[1].update_centered(joint_mean[None], joint_cov[None], np.ones(1))
        for got, wanted, modeled in zip(
            actual, (joint_mean[:2], joint_cov[:2, :2]),
            agent.filter.gaussian_marginal(expected_filter.state, "x1"), strict=True,
        ):
            np.testing.assert_array_equal(got, modeled)
            np.testing.assert_allclose(got, wanted, atol=2e-6)
        for field, values in agent.learnables.params["motion"].items():
            np.testing.assert_array_equal(values[0], getattr(priors[0], field))
            np.testing.assert_allclose(values[1], getattr(expected, field), atol=2e-6)

        after = agent.learnables.state_dict()
        with pytest.raises(ValueError, match="outstanding decision tick"):
            agent.observe(1, [.5], tick=proposal.tick)
        assert_payload_equal(after, agent.learnables.state_dict())

        agent.propose({"goal": None})
        before_mean, before_cov = agent.belief
        matrix = agent.learnables.rollover("motion", "matrix")[1]
        offset = agent.learnables.rollover("motion", "offset")[1]
        noise = agent.learnables.rollover("motion", "cov")[1]
        next_mean, next_cov = agent.observe(1, None, tick=1)
        np.testing.assert_allclose(next_mean, matrix @ before_mean + offset, atol=2e-6)
        np.testing.assert_allclose(next_cov, matrix @ before_cov @ matrix.T + noise, atol=2e-6)
        assert_payload_equal(after, agent.learnables.state_dict())


def test_feedback_retry_keeps_evidence_identity_and_two_agents_are_independent(tmp_path, monkeypatch):
    paths, _ = pair(tmp_path, habit=(.4, .6))
    a = RollingAgent.from_gaussian_bundles(*paths, select="sample", seed=22)
    b = RollingAgent.from_gaussian_bundles(*paths, select="sample", seed=22)
    preview = {"goal": [.5]}
    decision = a.propose(preview)
    assert b.propose(preview).command == decision.command
    before = a.learnables.state_dict()
    rng_state = a._rng.bit_generator.state

    def fail(engine, result):
        raise ValueError("injected learner validation failure")

    with monkeypatch.context() as fault:
        fault.setattr(a.learnables, "accumulate", fail)
        with pytest.raises(ValueError, match="injected learner"):
            a.observe(1, [.5], tick=0)
    assert a.tick == 0 and a._rng.bit_generator.state == rng_state
    assert_payload_equal(before, a.learnables.state_dict())
    assert a.propose(preview).command == decision.command
    assert a._rng.bit_generator.state == rng_state
    with pytest.raises(ValueError, match="feedback changed"):
        a.observe(1, [.6], tick=0)

    pending = tmp_path / "retry.npz"
    a.checkpoint(pending)
    restored = RollingAgent.from_gaussian_bundles(*paths, select="sample", seed=99)
    restored.restore(pending)
    assert restored.propose(preview).command == decision.command
    with pytest.raises(ValueError, match="feedback changed"):
        restored.observe(0, [.5], tick=0)
    for agent in (a, restored):
        agent.observe(1, [.5], tick=0)
    assert_payload_equal(a.learnables.state_dict(), restored.learnables.state_dict())
    assert_payload_equal(before, b.learnables.state_dict())
    assert b.tick == 0
    assert a.propose(preview).command == restored.propose(preview).command
    assert a._rng.bit_generator.state == restored._rng.bit_generator.state


def test_marked_hold_uses_actual_action_and_stays_pending(tmp_path, monkeypatch):
    paths, _ = pair(tmp_path, learning=False)
    with pytest.raises(ValueError, match="requires an initial command"):
        RollingAgent.from_gaussian_bundles(*paths, on_failure="hold")
    agent = RollingAgent.from_gaussian_bundles(*paths, on_failure="hold", initial_command=0)
    diagnostic = FailureDiagnostic("iteration_budget", None, "soft", 1)
    monkeypatch.setattr(agent.plan, "run", lambda state, *args, **kwargs: RunResult(
        state, RunStatus.BUDGET_EXHAUSTED, 1, (), (), np.inf, failure=diagnostic,
    ))
    preview = {"goal": [-.5]}
    first = agent.propose(preview)
    assert first.held and first.command == 0 and first.free_energy is None
    assert first.status is RunStatus.BUDGET_EXHAUSTED and first.failure == diagnostic
    assert agent.propose(preview) == first
    agent.observe(1, None, tick=0)
    second = agent.propose(preview)
    assert second.held and second.command == 1 and second.tick == 1
    path = tmp_path / "hold.npz"
    agent.checkpoint(path)
    other = RollingAgent.from_gaussian_bundles(*paths, on_failure="hold", initial_command=0)
    other.restore(path)
    assert other.propose(preview) == second
    other.observe(1, None, tick=1)
    other.reset_episode()
    assert other.episode_tick == 0 and other.tick == 2
    assert other.learnables.plans == {}
    monkeypatch.setattr(other.plan, "marginals", lambda *args: {"a0": np.array([np.nan, 0.])})
    invalid = other.propose(preview)
    assert invalid.held and invalid.command == 0 and invalid.status is RunStatus.DIVERGED
    assert invalid.failure.reason == "invalid_action_readout"


@pytest.mark.slow
def test_longer_plan_and_both_restart_cuts_in_fresh_process(tmp_path):
    paths, _ = pair(tmp_path, habit=(.4, .6), horizon=10)
    agent = RollingAgent.from_gaussian_bundles(*paths, n_sweeps=160, select="sample", seed=91)
    first = {"goal": [.4]}
    second = {"goal": [-.2]}
    first_decision = agent.propose(first)
    agent.checkpoint(tmp_path / "pending.npz")
    agent.observe(1, [.5], tick=0)
    agent.checkpoint(tmp_path / "committed.npz")
    manual_plan = manual(agent.plan, agent.belief, second, agent.learnables, budget=160)
    expected = agent.propose(second)
    assert expected.free_energy == manual_plan.free_energies[-1]
    agent.observe(0, [.25], tick=1)
    agent.checkpoint(tmp_path / "golden.npz")
    script = dedent(f"""\
        from cffg import RollingAgent

        for cut in ('pending', 'committed'):
            agent = RollingAgent.from_gaussian_bundles(
                'plan.cffgz', 'filter.cffgz', n_sweeps=160, select='sample', seed=83,
            )
            agent.restore(cut + '.npz')
            if cut == 'pending':
                first = agent.propose({first!r})
                assert first.command == {first_decision.command}
                agent.observe(1, [.5], tick=0)
            decision = agent.propose({second!r})
            assert decision.command == {expected.command}
            assert decision.free_energy == {expected.free_energy}
            agent.observe(0, [.25], tick=1)
            agent.checkpoint(cut + '-resume.npz')
    """)
    env = {**os.environ, "JAX_ENABLE_X64": str(int(jax.config.jax_enable_x64))}
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, check=True, timeout=120)
    with np.load(tmp_path / "golden.npz", allow_pickle=False) as reference:
        for name in ("pending-resume", "committed-resume"):
            with np.load(tmp_path / (name + ".npz"), allow_pickle=False) as resumed:
                assert_payload_equal(dict(reference), dict(resumed))


def test_affine_filter_sensor_learns_one_concurrent_feedback_and_restores(tmp_path):
    from test_gaussian_agent import window as affine_window

    paths = tmp_path / "plan.cffgz", tmp_path / "filter.cffgz"
    save_bundle(paths[0], *affine_window(True))
    model, arrays = affine_window(False)
    model.learnables["sensor"] = LearnableSpec("sensor", "mniw", ("H", "z", "R"), ("sensor",))
    prior = conj.MNIW(np.column_stack((arrays["H"], arrays["z"])), np.eye(4), 4 * arrays["R"], 4.)
    fit_engine = Engine(compile_template(model, arrays=arrays))
    save_bundle(paths[1], model, arrays, learnables=Learnables(fit_engine, priors={"sensor": prior}))
    agent = RollingAgent.from_gaussian_bundles(*paths)
    agent.propose({"target": [2.]})
    before = agent.learnables.state_dict()
    assert agent.tick == 0

    def feedback():
        try:
            return agent.observe([-.5], [.5], tick=0)
        except ValueError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as workers:
        completed = [workers.submit(feedback) for _ in range(2)]
        outcomes = [future.result(timeout=30) for future in completed]
    assert sum(isinstance(result, ValueError) for result in outcomes) == 1
    assert any("outstanding decision tick" in str(result) for result in outcomes)
    state_mean = np.array([.3, .15, 0.])
    state_cov = np.array([[.24, .12, 0.], [.12, .96, 0.], [0., 0., 1.]])
    np.testing.assert_allclose(agent.belief[0], state_mean, atol=2e-6)
    np.testing.assert_allclose(agent.belief[1], state_cov, atol=2e-6)
    joint_cov = np.pad(state_cov, ((1, 0), (1, 0)))
    expected = prior.update_centered(np.r_[.5, state_mean][None], joint_cov[None], np.ones(1))
    for field, value in agent.learnables.params["sensor"].items():
        np.testing.assert_allclose(value, getattr(expected, field), atol=2e-6)
    assert agent.tick == 1
    assert agent.learnables.params["sensor"]["nu"] == 5
    assert before.keys() == agent.learnables.state_dict().keys()

    checkpoint = tmp_path / "affine-learning.npz"
    agent.checkpoint(checkpoint)
    other = RollingAgent.from_gaussian_bundles(*paths)
    other.restore(checkpoint)
    assert_payload_equal(agent.learnables.state_dict(), other.learnables.state_dict())
    for restored in (agent, other):
        restored.propose({"target": [0.]})
        restored.observe([.2], [.1], tick=1)
    assert_payload_equal(agent.learnables.state_dict(), other.learnables.state_dict())
    np.testing.assert_array_equal(agent.belief[0], other.belief[0])


def test_checkpoint_rejects_other_priors_and_altered_rng_before_commit(tmp_path):
    from cffg._gaussian_agent_checkpoint import _digest

    paths, _ = pair(tmp_path, habit=(.5, .5))
    other_root = tmp_path / "other"
    other_root.mkdir()
    other_paths, _ = pair(other_root, habit=(.5, .5), prior_shift=.125)
    with pytest.raises(ValueError, match="same immutable filter prior"):
        RollingAgent.from_gaussian_bundles(paths[0], other_paths[1])
    first = RollingAgent.from_gaussian_bundles(*paths, select="sample", seed=90)
    different = RollingAgent.from_gaussian_bundles(*other_paths, select="sample", seed=90)
    first.propose({"goal": [.5]})
    path = tmp_path / "agent.npz"
    first.checkpoint(path)
    with pytest.raises(ValueError, match="incompatible Gaussian agent checkpoint"):
        different.restore(path)
    with np.load(path, allow_pickle=False) as checkpoint:
        arrays = {k: checkpoint[k] for k in checkpoint.files}
    meta = json.loads(str(arrays.pop("metadata")))
    meta.pop("digest")
    meta["rng"]["state"]["inc"] = 2
    meta["digest"] = _digest(meta, arrays)
    np.savez(path, metadata=np.asarray(json.dumps(meta)), **arrays)
    rng = first._rng.bit_generator.state
    with pytest.raises(ValueError, match="invalid selection RNG"):
        first.restore(path)
    assert first._rng.bit_generator.state == rng
    assert first.tick == different.tick == 0
    first.observe(1, [.5], tick=0)

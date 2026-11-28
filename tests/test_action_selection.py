"""Seeded environment actions and exact, reproducible MAP ties."""

from functools import partial

import numpy as np
import pytest

from cffg import RollingAgent, RunStatus
from cffg.action_selection import choose_map_action
from cffg.experiments import behavioral_epistemics as behavior
from cffg.models.rollout import build_rollout


def test_uniform_exact_maxima_preserve_single_winner_rng():
    rng = np.random.default_rng(713)
    before = rng.bit_generator.state
    below = np.nextafter(np.float32(0.5), np.float32(0))
    assert choose_map_action(np.array([below, 0.5], np.float32), rng) == 1
    assert rng.bit_generator.state == before
    q = np.array([0.5, below, 0.5, 0, 0.5, 0.5], np.float32)
    draws = [choose_map_action(q, rng) for _ in range(4096)]
    counts = np.bincount(draws, minlength=len(q)) / len(draws)
    assert counts[1] == counts[3] == 0
    np.testing.assert_allclose(counts[[0, 2, 4, 5]], 0.25, atol=0.03, rtol=0)
    large = np.full(128, 1 / 128, np.float32)
    def choices(seed):
        source = np.random.default_rng(seed)
        return [choose_map_action(large, source) for _ in range(128)]
    assert choices(1) == choices(1) != choices(2)
    assert len(set(choices(1))) > 1
    for invalid in ([0, 0], [np.nan, 1], [1, np.inf], [-1, 2], [], [[1]]):
        with pytest.raises(ValueError, match="action probabilities"):
            choose_map_action(invalid, rng)


def test_rolling_agent_replays_seeded_ties_through_both_checkpoint_formats(tmp_path):
    builder = partial(build_rollout, "tie", gfe=False, d=np.full(2, 0.5),
                      A=np.full((2, 2), 0.5), c=np.full(2, 0.5), e=np.full(2, 0.5),
                      Bs=np.stack([np.eye(2), np.eye(2)]))
    def make(seed, select="map"):
        return RollingAgent(builder, horizon=1, n_sweeps=20, damp=0, seed=seed, select=select)
    agent = make(382)
    start = [agent.act() for _ in range(8)]
    assert set(start) == {0, 1}
    ckpt = tmp_path / "state.npz"
    agent.checkpoint(ckpt)
    expected = [agent.act() for _ in range(8)]
    restored = make(1)
    restored.restore(ckpt)
    assert [restored.act() for _ in range(8)] == expected
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, tick=np.asarray(0), belief=agent.belief)
    seeded = make(382)
    seeded.restore(legacy)
    assert [seeded.act() for _ in range(8)] == start
    sampler = make(382, select="sample")
    expected_rng = np.random.default_rng(382)
    assert [sampler.act() for _ in range(8)] == list(expected_rng.choice(2, size=8, p=[0.5, 0.5]))


def test_executed_cue_actions_use_the_matching_first_action_history(monkeypatch):
    def solve(engine, *, past, data=None):
        if not past:
            q = [0.5, 0.5, 0, 0]
        elif data["act1"] == 0:
            q = [0, 0.5, 0.5, 0]
        else:
            q = [0.5, 0, 0, 0.5]
        return behavior.DecisionPlan(np.array(q), np.full(2, 0.5), RunStatus.CONVERGED,
                                     0.0, 1, 0.0, True)

    executed = []
    original = behavior.CueTaskEnv.step
    def step(env, action):
        executed.append(action)
        return original(env, action)
    monkeypatch.setattr(behavior, "_solve", solve)
    monkeypatch.setattr(behavior, "_engine", lambda *args, **kwargs: None)
    monkeypatch.setattr(behavior.CueTaskEnv, "step", step)
    report = behavior.run_benchmark(seeds=(31,), episodes_per_seed=128)
    assert report.first_plan.action == 0
    assert set(zip(executed[::2], executed[1::2])) == {(0, 1), (0, 2), (1, 0), (1, 3)}
    assert set(report.conditional_by_history) == {
        (action, obs) for action in (0, 1) for obs in (behavior.REWARD, behavior.LOSS)
    }

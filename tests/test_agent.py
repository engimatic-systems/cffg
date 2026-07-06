"""RollingAgent: the receding-horizon runtime on the gridworld corridor."""

import functools

import numpy as np
import pytest

from cffg.agent import RollingAgent
from cffg.models.gridworld import RIGHT, STAY, CorridorEnv, build_gridworld

K = 6
builder = functools.partial(build_gridworld, k=K)


def fresh(seed=1, **kwargs):
    return CorridorEnv(k=K, seed=seed), RollingAgent(builder, horizon=3, seed=seed, **kwargs)


def test_agent_walks_to_goal_optimally():
    env, agent = fresh()
    actions = [agent.step(env.step)[0] for _ in range(K - 1)]
    assert actions == [RIGHT] * (K - 1)
    assert env.pos == K - 1
    # and then it stays put — goal reached, nothing left to want
    assert agent.act() == STAY


def test_belief_tracks_true_position():
    env, agent = fresh(seed=3)
    hits = 0
    for _ in range(8):
        agent.step(env.step)
        hits += int(np.argmax(agent.belief)) == env.pos
    assert hits >= 7  # noisy obs (0.9) may blur a tick, not the trajectory


def test_two_engines_regardless_of_episode_length():
    env, agent = fresh()
    for _ in range(10):
        agent.step(env.step)
    # the runtime owns exactly two compiled models: plan + filter
    assert agent.plan is not agent.filter
    assert agent.tick == 10


def test_checkpoint_roundtrip_resumes_identically(tmp_path):
    env_a, agent_a = fresh(seed=5)
    for _ in range(3):
        agent_a.step(env_a.step)
    ckpt = tmp_path / "agent.npz"
    agent_a.checkpoint(ckpt)

    env_b, agent_b = fresh(seed=5)
    agent_b.restore(ckpt)
    np.testing.assert_allclose(agent_b.belief, agent_a.belief)
    assert agent_b.tick == agent_a.tick == 3
    assert agent_b.act() == agent_a.act()


def test_sample_selection_supported():
    env, agent = fresh(select="sample")
    u, x = agent.step(env.step)
    assert 0 <= u < 3 and 0 <= x < K


def test_learned_a_agent_accumulates_counts():
    env, agent = fresh(
        learn_a=True, initial_counts=np.ones((K, K)) * 0.5
    )
    before = agent.counts.sum()
    for _ in range(4):
        agent.step(env.step)
    np.testing.assert_allclose(agent.counts.sum() - before, 4.0, atol=1e-8)
    # counts concentrate along the visited diagonal path
    assert agent.counts[0, 0] > 0.5 or agent.counts[1, 1] > 0.5

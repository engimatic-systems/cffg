"""RollingAgent: the receding-horizon runtime on the gridworld corridor."""

import functools

import jax
import numpy as np
import pytest

from cffg.agent import RollingAgent
from cffg.compiler import NEG, _log_of
from cffg.models.gridworld import RIGHT, STAY, CorridorEnv, build_gridworld

K = 6
builder = functools.partial(build_gridworld, k=K)


def fresh(seed=1, **kwargs):
    return CorridorEnv(k=K, seed=seed), RollingAgent(builder, horizon=3, seed=seed, **kwargs)


@pytest.mark.slow
def test_agent_walks_to_goal_optimally():
    env, agent = fresh()
    actions = [agent.step(env.step)[0] for _ in range(K - 1)]
    assert actions == [RIGHT] * (K - 1)
    assert env.pos == K - 1
    # and then it stays put — goal reached, nothing left to want
    assert agent.act() == STAY


@pytest.mark.slow
def test_belief_tracks_true_position():
    env, agent = fresh(seed=3)
    hits = 0
    for _ in range(8):
        agent.step(env.step)
        hits += int(np.argmax(agent.belief)) == env.pos
    assert hits >= 7  # noisy obs (0.9) may blur a tick, not the trajectory


@pytest.mark.slow
def test_two_engines_regardless_of_episode_length():
    env, agent = fresh()
    for _ in range(10):
        agent.step(env.step)
    # the runtime owns exactly two compiled models: plan + filter
    assert agent.plan is not agent.filter
    assert agent.tick == 10


@pytest.mark.slow
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


@pytest.mark.slow
def test_sample_selection_supported():
    env, agent = fresh(select="sample")
    u, x = agent.step(env.step)
    assert 0 <= u < 3 and 0 <= x < K


@pytest.mark.slow
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


# -- cff-ey4p: the belief log-floor must scale with the belief's dtype -------


def test_log_of_exact_zero_is_neg_without_warnings_in_both_dtypes():
    """A 1e-300 literal floor underflows to 0 under float32, so log evaluates
    log(0) and warns even though np.where discards that lane (cff-ey4p)."""
    for dtype in (np.float32, np.float64):
        belief = np.array([0.0, 0.5, 0.5, 0.0], dtype=dtype)
        with np.errstate(divide="raise", invalid="raise"):
            out = _log_of(belief)
        assert out.dtype == dtype
        np.testing.assert_array_equal(out[[0, 3]], np.full(2, NEG, dtype=dtype))
        np.testing.assert_allclose(out[[1, 2]], np.log(0.5), rtol=1e-6)
        # positive subnormals are not "semantically zero": finite log, not NEG
        sub = np.array([np.finfo(dtype).tiny / 4], dtype=dtype)
        assert np.isfinite(_log_of(sub)) and _log_of(sub) > NEG


def test_log_of_unchanged_for_full_support_float64():
    rng = np.random.default_rng(0)
    belief = rng.dirichlet(np.ones(64))
    np.testing.assert_array_equal(_log_of(belief), np.log(belief))


@pytest.mark.slow
def test_sparse_float32_belief_keeps_agent_paths_finite():
    """After one observe() the belief carries the engine dtype; a one-hot
    float32 belief must feed the planner finite log-space parameters."""
    with jax.enable_x64(False):
        env, agent = fresh()
        agent.step(env.step)
        assert agent.belief.dtype == np.float32
        agent.belief = np.zeros(K, dtype=np.float32)
        agent.belief[env.pos] = 1.0
        with np.errstate(divide="raise", invalid="raise"):
            u = agent.act()
            agent.observe(u, env.step(u))
        assert 0 <= u < 3
        assert np.all(np.isfinite(agent.belief))
        np.testing.assert_allclose(agent.belief.sum(), 1.0, rtol=1e-5)


@pytest.mark.slow
def test_uncomfortable_cardinality_float32_tick_stays_finite():
    """K = 96 states with exact-zero belief mass, consumer-default float32."""
    big = 96
    with jax.enable_x64(False):
        env = CorridorEnv(k=big, seed=2)
        agent = RollingAgent(functools.partial(build_gridworld, k=big), horizon=2, seed=2)
        agent.belief = np.zeros(big, dtype=np.float32)
        agent.belief[0] = 1.0
        with np.errstate(divide="raise", invalid="raise"):
            u, x = agent.step(env.step)
        assert np.all(np.isfinite(agent.belief))
        assert 0 <= x < big

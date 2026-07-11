"""B-learning tier (cff-cd1i): node beliefs and conjugate transition updates.

The load-bearing object is Engine.node_beliefs — the smoothed joint at a
factor. On trees BP node beliefs are exact posteriors, so the pair belief at
a transition node must match brute-force enumeration; the conjugate update
integrating those pairs must then recover a known stochastic B from data.
"""

import numpy as np

from cffg.agent import RollingAgent
from cffg.compiler import compile_template
from cffg.core.ir import ConstraintKind, ConstraintSpec
from cffg.engine import Engine
from cffg.models.rollout import build_rollout

RNG = np.random.default_rng(11)

K, MOBS, U = 3, 2, 2
D = np.array([0.5, 0.3, 0.2])
A = np.array([[0.8, 0.3, 0.5], [0.2, 0.7, 0.5]])
BS = np.stack([
    np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.3], [0.1, 0.2, 0.6]]),
    np.array([[0.1, 0.5, 0.4], [0.6, 0.3, 0.3], [0.3, 0.2, 0.3]]),
])


def two_step_engine():
    t, arrays = build_rollout(
        "pair", steps=2, past=2, d=D, A=A,
        c=np.full(MOBS, 1 / MOBS), e=np.full(U, 1 / U), Bs=BS,
    )
    return Engine(compile_template(t, arrays=arrays))


def brute_force_pair(u1, x1, u2, x2):
    """p(z1, z0 | data) by enumeration over the 2-step chain."""
    joint = np.zeros((K, K))
    for z0 in range(K):
        for z1 in range(K):
            for z2 in range(K):
                joint[z1, z0] += (
                    D[z0] * BS[u1][z1, z0] * A[x1, z1] * BS[u2][z2, z1] * A[x2, z2]
                )
    return joint / joint.sum()


def test_pair_belief_matches_enumeration():
    engine = two_step_engine()
    for u1, x1, u2, x2 in [(0, 1, 1, 0), (1, 0, 0, 0), (1, 1, 1, 1)]:
        st = engine.bind_data(
            engine.init_state(), {"act1": u1, "obs1": x1, "act2": u2, "obs2": x2}
        )
        st = engine.run(st, 30, tol=1e-13).require_converged()
        pair = engine.node_beliefs(st)["tm1"]  # [z', z, u]
        assert pair.shape == (K, K, U)
        np.testing.assert_allclose(pair[:, :, u1], brute_force_pair(u1, x1, u2, x2), atol=1e-9)
        assert pair[:, :, 1 - u1].max() < 1e-9  # clamped action: no stray mass


def test_node_belief_of_leaf_matches_edge_marginal():
    engine = two_step_engine()
    st = engine.bind_data(engine.init_state(), {"act1": 0, "obs1": 1, "act2": 1, "obs2": 0})
    st = engine.run(st, 30, tol=1e-13).require_converged()
    np.testing.assert_allclose(
        engine.node_beliefs(st)["prior"], engine.marginals(st)["z0"], atol=1e-9
    )


def test_vmp_node_belief_is_outer_product_of_marginals():
    t, arrays = build_rollout(
        "mf", steps=1, past=1, d=D, A=A,
        c=np.full(MOBS, 1 / MOBS), e=np.full(U, 1 / U), Bs=BS,
    )
    t.constraints.append(ConstraintSpec(ConstraintKind.MEAN_FIELD, "obs1"))
    engine = Engine(compile_template(t, arrays=arrays))
    st = engine.bind_data(engine.init_state(), {"act1": 0, "obs1": 1})
    st = engine.run(st, 40, tol=1e-13).require_converged()
    q = engine.node_beliefs(st)["obs1"]  # [x, z]
    m = engine.marginals(st)
    np.testing.assert_allclose(q, np.outer(m["x1"], m["zo1"]), atol=1e-9)


# --- conjugate B recovery on a slip corridor ---------------------------------

KC = 4  # corridor cells
SLIP = 0.15


def corridor_bs() -> np.ndarray:
    """left/right with slip: move as intended w.p. 1−SLIP, else stay."""
    bs = np.zeros((2, KC, KC))
    for z in range(KC):
        left, right = max(z - 1, 0), min(z + 1, KC - 1)
        bs[0, left, z] += 1 - SLIP
        bs[0, z, z] += SLIP
        bs[1, right, z] += 1 - SLIP
        bs[1, z, z] += SLIP
    return bs


def corridor_builder(**kw):
    acc = 0.9
    a = np.full((KC, KC), (1 - acc) / (KC - 1))
    np.fill_diagonal(a, acc)
    return build_rollout(
        "corridor", d=np.full(KC, 1 / KC), A=a,
        c=np.full(KC, 1 / KC), e=np.full(2, 0.5),
        Bs=np.full((2, KC, KC), 1 / KC),  # agent starts ignorant of dynamics
        gfe=False, **kw,
    )


def test_b_recovery_from_random_walk():
    true_bs = corridor_bs()
    acc = 0.9
    a = np.full((KC, KC), (1 - acc) / (KC - 1))
    np.fill_diagonal(a, acc)

    agent = RollingAgent(
        corridor_builder, horizon=2, learn_b=True,
        initial_b_counts=np.ones((2, KC, KC)), n_sweeps=60, seed=3,
    )
    z = int(RNG.integers(KC))
    for _ in range(2500):
        u = int(RNG.integers(2))
        z = int(RNG.choice(KC, p=true_bs[u, :, z]))
        x = int(RNG.choice(KC, p=a[:, z]))
        agent.observe(u, x)

    learned = agent.b_counts / agent.b_counts.sum(axis=1, keepdims=True)
    tv = 0.5 * np.abs(learned - true_bs).sum(axis=1)  # [U, K] per-column TV
    assert tv.max() < 0.12, f"worst column TV {tv.max():.3f}\n{np.round(learned, 2)}"


def test_checkpoint_roundtrip_with_b_counts(tmp_path):
    agent = RollingAgent(
        corridor_builder, horizon=2, learn_b=True,
        initial_b_counts=np.ones((2, KC, KC)), n_sweeps=40, seed=5,
    )
    agent.observe(1, 2)
    path = tmp_path / "agent.npz"
    agent.checkpoint(path)
    other = RollingAgent(
        corridor_builder, horizon=2, learn_b=True,
        initial_b_counts=np.ones((2, KC, KC)), n_sweeps=40, seed=5,
    )
    other.restore(path)
    np.testing.assert_array_equal(other.b_counts, agent.b_counts)
    np.testing.assert_array_equal(other.belief, agent.belief)


def test_learning_templates_render(viz_gallery):
    # The CFFGs the learner actually runs: the one-slice filter (where the
    # pair belief lives) and the two-slice corridor planner.
    filt_t, _ = corridor_builder(past=1, steps=1)
    plan_t, _ = corridor_builder(past=0, steps=2)
    viz_gallery(filt_t, tag="filter slice: pair belief at TM")
    viz_gallery(plan_t, tag="corridor planner, horizon 2")


def test_partial_pairs_still_sum_to_one():
    # A soft (unclamped-action) filter: node belief must still normalize and
    # marginalize consistently even when u carries real uncertainty.
    engine = two_step_engine()
    st = engine.bind_data(
        engine.init_state(), {"act1": None, "obs1": 1, "act2": 1, "obs2": 0}
    )
    st = engine.run(st, 30, tol=1e-13).require_converged()
    pair = engine.node_beliefs(st)["tm1"]
    np.testing.assert_allclose(pair.sum(), 1.0, atol=1e-9)
    np.testing.assert_allclose(
        pair.sum(axis=(1, 2)), engine.marginals(st)["zm1"], atol=1e-9
    )

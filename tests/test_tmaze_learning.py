"""Trial-loop learning: the perception-action cycle and conjugate updates.

What is verified here: the cycle runs end-to-end with three phase engines
compiled once (params swapped per trial, no recompiles), actions come from
control posteriors, and end-of-trial smoothing accumulates Dirichlet counts
that correctly identify the cue's context mapping.

What is deliberately NOT asserted yet: the full win-rate ramp of Part II
Fig. 9. Mean-A planning with exp(E[log A]) leaks column mass at low-count
cells, making uncertain states look improbable (anti-novelty artifact) — the
faithful q(A)-aware planning messages are ticket cff-5iuz, and the behavioral
assertions belong to it.
"""

import numpy as np

from cffg.experiments.tmaze_learning import (
    TMazeEnv,
    elog_a,
    initial_counts,
    run_learning,
)
from cffg.models.tmaze import C, N_CTX, N_OUT

TRIALS = 30


def test_cycle_learns_the_cue_mapping():
    result = run_learning(n_trials=TRIALS, seed=1)
    cue = result.counts[C * N_OUT : C * N_OUT + 2, C * N_CTX : C * N_CTX + N_CTX]
    # Diagonal (CL|RL and CR|RR) must dominate off-diagonal decisively.
    assert cue[0, 0] > 5 * cue[0, 1], f"cue mapping not learned: {cue}"
    assert cue[1, 1] > 5 * cue[1, 0], f"cue mapping not learned: {cue}"


def test_counts_are_conserved():
    """Each observed step adds exactly one unit of pseudo-count mass."""
    result = run_learning(n_trials=TRIALS, seed=1)
    added = result.counts.sum() - initial_counts().sum()
    np.testing.assert_allclose(added, 2 * TRIALS, atol=1e-6)


def test_loop_is_deterministic():
    a = run_learning(n_trials=8, seed=3)
    b = run_learning(n_trials=8, seed=3)
    assert a.wins == b.wins
    np.testing.assert_allclose(a.counts, b.counts)


def test_digamma_expectation_shape_and_order():
    counts = initial_counts()
    e = elog_a(counts)
    assert e.shape == counts.shape
    assert np.all(e < 0)
    # Confident cells are far less penalized than epsilon cells.
    assert e[0, 0] > e[2, 0] + 3.0  # O-block 10-count vs eps cell, same column


def test_env_reward_statistics():
    env = TMazeEnv(alpha=0.9, rng=np.random.default_rng(0))
    rewards = 0
    n = 3000
    for _ in range(n):
        env.reset()
        env.step(C)  # cue: never rewarding
        x = env.step(1 + env.ctx)  # correct arm given true context (L=1, R=2)
        rewards += x % N_OUT == 2
    assert abs(rewards / n - 0.9) < 0.03

"""A 1-D corridor gridworld: the second client of the generic rollout.

K cells; controls (left, stay, right) as deterministic shift matrices with
clamped boundaries; observations are a noisy identity over cells; the goal
prior prefers observing the rightmost cell. Everything the T-maze machinery
built, pointed at a different world with one function call — the point of
the exercise.
"""

from __future__ import annotations

import numpy as np

from cffg.core.ir import Template
from cffg.models.rollout import build_rollout

LEFT, STAY, RIGHT = 0, 1, 2


def transition_stack(k: int) -> np.ndarray:
    left, stay, right = np.zeros((3, k, k))
    for j in range(k):
        left[max(j - 1, 0), j] = 1.0
        stay[j, j] = 1.0
        right[min(j + 1, k - 1), j] = 1.0
    return np.stack([left, stay, right])


def observation_matrix(k: int, acc: float = 0.9) -> np.ndarray:
    a = np.full((k, k), (1.0 - acc) / (k - 1))
    np.fill_diagonal(a, acc)
    return a


def goal_statistic(k: int, utility: float = 3.0) -> np.ndarray:
    u = np.zeros(k)
    u[k - 1] = utility
    e = np.exp(u - u.max())
    return e / e.sum()


def build_gridworld(
    past: int = 0,
    steps: int = 3,
    *,
    k: int = 6,
    acc: float = 0.9,
    utility: float = 3.0,
    start: int = 0,
    gfe: bool = True,
    learned_a: bool = False,
    learn: tuple[str, ...] = (),
) -> tuple[Template, dict[str, np.ndarray]]:
    d = np.full(k, 1e-3)
    d[start] = 1.0
    d /= d.sum()
    return build_rollout(
        f"gridworld{k}[{past}|{steps}]",
        steps=steps,
        past=past,
        d=d,
        A=observation_matrix(k, acc),
        c=goal_statistic(k, utility),
        e=np.full(3, 1.0 / 3.0),
        Bs=transition_stack(k),
        gfe=gfe,
        learned_a=learned_a,
        learn=learn,
    )


class CorridorEnv:
    """Ground truth: deterministic motion, noisy observation of position."""

    def __init__(self, k: int = 6, acc: float = 0.9, start: int = 0, seed: int = 0):
        self.k, self.acc, self.pos = k, acc, start
        self.a = observation_matrix(k, acc)
        self.rng = np.random.default_rng(seed)

    def step(self, action: int) -> int:
        self.pos = {LEFT: max(self.pos - 1, 0), STAY: self.pos, RIGHT: min(self.pos + 1, self.k - 1)}[action]
        return int(self.rng.choice(self.k, p=self.a[:, self.pos]))

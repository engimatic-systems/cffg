"""The classic T-maze (Friston et al. 2015), Part I §7 / Part II §6 layout.

Positions P = (O, L, R, C): start, left arm, right arm, cue. Hidden reward
context R = (RL, RR). State = position ⊗ context (position-major, 8). An
observation is position ⊗ outcome with outcomes (CL, CR, RW, NR) (16).
Controls select one of four transition matrices B_u = P_u ⊗ I₂; invalid moves
bounce to O, reward arms are followed by a mandatory move back.

`build_tmaze` produces the policy-inference model of Part I Fig. 26: state
prior d at the current time, then per future step a control prior, a
transition-mixture node, an equality branch, and a P-substituted
goal-observation composite. With gfe=False the P-beads are omitted — the
Part II §7.1 BFE-ablation agent that lacks the epistemic drive.
"""

from __future__ import annotations

import numpy as np

from cffg.core.ir import Template
from cffg.models.rollout import build_rollout

O, L, R, C = 0, 1, 2, 3  # noqa: E741  (paper's position names)
N_POS, N_CTX, N_OUT = 4, 2, 4
N_STATE, N_OBS = N_POS * N_CTX, N_POS * N_OUT


def transition_stack() -> np.ndarray:
    """[4, 8, 8] stack of B_u = P_u ⊗ I₂ (Part II §6.2)."""
    p = np.zeros((4, N_POS, N_POS))
    p[O] = [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    p[L] = [[0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
    p[R] = [[0, 1, 1, 0], [0, 0, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]]
    p[C] = [[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]
    return np.stack([np.kron(p[u], np.eye(N_CTX)) for u in range(4)])


def observation_matrix(alpha: float) -> np.ndarray:
    """[16, 8] block-diagonal likelihood (Part II Table 3), position-major."""
    blocks = {
        O: [[0.5, 0.5], [0.5, 0.5], [0, 0], [0, 0]],
        L: [[0, 0], [0, 0], [alpha, 1 - alpha], [1 - alpha, alpha]],
        R: [[0, 0], [0, 0], [1 - alpha, alpha], [alpha, 1 - alpha]],
        C: [[1, 0], [0, 1], [0, 0], [0, 0]],
    }
    a = np.zeros((N_OBS, N_STATE))
    for pos in range(N_POS):
        a[pos * N_OUT : (pos + 1) * N_OUT, pos * N_CTX : (pos + 1) * N_CTX] = blocks[pos]
    return a


def goal_statistic(c_util: float) -> np.ndarray:
    """softmax of per-outcome utilities (0, 0, +c, −c), tiled per position."""
    u = np.tile([0.0, 0.0, c_util, -c_util], N_POS)
    e = np.exp(u - u.max())
    return e / e.sum()


def initial_state() -> np.ndarray:
    d = np.zeros(N_STATE)
    d[O * N_CTX : O * N_CTX + N_CTX] = 0.5  # at O, context unknown
    return d


def build_tmaze(
    steps: int = 2,
    alpha: float = 0.9,
    c_util: float = 2.0,
    *,
    past: int = 0,
    gfe: bool = True,
    map_policy: bool = False,
    learned_a: bool = False,
) -> tuple[Template, dict[str, np.ndarray]]:
    """Policy-inference CFFG over `steps` timesteps, plus its arrays.

    A thin wrapper over the generic rollout template (models/rollout.py) with
    the T-maze's d/A/c/e/Bs; see build_rollout for the past/future semantics."""
    return build_rollout(
        f"tmaze[{past}|{steps}]",
        steps=steps,
        past=past,
        d=initial_state(),
        A=observation_matrix(alpha),
        c=goal_statistic(c_util),
        e=np.full(N_POS, 1.0 / N_POS),
        Bs=transition_stack(),
        gfe=gfe,
        map_policy=map_policy,
        learned_a=learned_a,
    )

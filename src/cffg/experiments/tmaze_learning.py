"""Learning the T-maze observation model over repeated trials (Part II §7.1).

Perception-action cycle, mean-A approximation (the true IS/Dirichlet-edge
machinery is ticket cff-5iuz):

  per trial: plan (GFE, A = normalized Dirichlet counts) → act (argmax u1)
             → observe → replan step 2 with the past slice data-clamped
             → act → observe → end-of-trial smoothing (pure BP)
             → Dirichlet counts += x̂_k ⊗ q(z_k)   (conjugate VMP update)

Three phase structures (plan-2, plan-1-after-1, smooth-2) are compiled once;
per-trial parameter changes ride through the params dict — the engine never
recompiles (the colony contract paying off in the runtime loop).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from cffg.compiler import compile_template
from cffg.engine import Engine, State
from cffg.models.tmaze import (
    C,
    N_CTX,
    N_OUT,
    N_POS,
    N_STATE,
    O,
    build_tmaze,
    observation_matrix,
)

RW = 2  # outcome index of "reward" within a position block


class TMazeEnv:
    """Ground-truth simulator matching the generative model's conventions."""

    def __init__(self, alpha: float, rng: np.random.Generator):
        self.a_true = observation_matrix(alpha)
        self.rng = rng
        self.ctx = 0
        self.pos = O

    def reset(self) -> None:
        self.ctx = int(self.rng.integers(N_CTX))
        self.pos = O

    def step(self, action: int) -> int:
        """Move, then sample an observation index (position ⊗ outcome)."""
        self.pos = int(action) if self.pos in (O, C) else O  # arms bounce to O
        state = self.pos * N_CTX + self.ctx
        outcome = int(self.rng.choice(N_OUT, p=self.a_true[
            self.pos * N_OUT : (self.pos + 1) * N_OUT, state
        ] / self.a_true[self.pos * N_OUT : (self.pos + 1) * N_OUT, state].sum()))
        return self.pos * N_OUT + outcome


def initial_counts(eps: float = 0.1) -> np.ndarray:
    """Part II A₀: O known uninformative (strong 10s), L/R/C possibly
    disambiguating (weak identity), ε elsewhere."""
    counts = np.full((N_POS * N_OUT, N_STATE), eps)
    blocks = {
        O: np.array([[10.0, 10.0], [10.0, 10.0], [eps, eps], [eps, eps]]),
        1: np.array([[1.0, eps], [eps, 1.0], [eps, eps], [eps, eps]]),
        2: np.array([[1.0, eps], [eps, 1.0], [eps, eps], [eps, eps]]),
        C: np.array([[1.0, eps], [eps, 1.0], [eps, eps], [eps, eps]]),
    }
    for pos, b in blocks.items():
        counts[pos * N_OUT : (pos + 1) * N_OUT, pos * N_CTX : (pos + 1) * N_CTX] = b
    return counts


def elog_a(counts: np.ndarray) -> np.ndarray:
    """E_q(A)[log A] under per-column Dirichlets: ψ(α) − ψ(α₀).

    This is the proper VMP expectation of log A and the load-bearing term:
    the digamma punishes low-count cells hard (ψ(0.1) ≈ −10.4), so the agent
    distinguishes *confident ambiguity* (O's strong 10s → genuinely no info)
    from *unconfident structure* (C's weak identity → worth investigating).
    Taking log of the mean instead flattens that distinction and the agent
    never leaves the start position."""
    from jax.scipy.special import digamma

    a = np.asarray(digamma(jnp.asarray(counts)))
    a0 = np.asarray(digamma(jnp.asarray(counts.sum(axis=0, keepdims=True))))
    return a - a0


@dataclass
class Phase:
    engine: Engine
    a_keys: list[tuple[str, int]]  # (params key, n_members) to refresh with Ā

    def state_with(self, log_a: np.ndarray) -> State:
        st = self.engine.init_state()
        params = dict(st.params)
        for key, m in self.a_keys:
            params[key] = jnp.asarray(np.broadcast_to(log_a, (m, *log_a.shape)))
        return State(messages=st.messages, params=params)


def _make_phase(past: int, steps: int, alpha: float, c_util: float, gfe: bool) -> Phase:
    template, arrays = build_tmaze(steps, alpha, c_util, past=past, gfe=gfe)
    engine = Engine(compile_template(template, arrays=arrays))
    a_keys = []
    for g in engine.arena.groups:
        if g.type == "matrix_cat":
            a_keys.append((g.name, len(g.members)))
    for g in engine.arena.gfe_groups:
        a_keys.append((g.name + ":A", len(g.members)))
    return Phase(engine, a_keys)


@dataclass
class LearningResult:
    wins: list[bool]
    counts: np.ndarray

    def win_rate(self, last: int | None = None) -> float:
        w = self.wins[-last:] if last else self.wins
        return sum(w) / len(w)


def run_learning(
    n_trials: int = 25,
    alpha: float = 0.9,
    c_util: float = 2.0,
    *,
    gfe: bool = True,
    seed: int = 0,
    damp: float = 0.5,
    n_sweeps: int = 400,
) -> LearningResult:
    env = TMazeEnv(alpha, np.random.default_rng(seed))
    plan2 = _make_phase(0, 2, alpha, c_util, gfe)
    plan1 = _make_phase(1, 2, alpha, c_util, gfe)
    smooth = _make_phase(2, 2, alpha, c_util, gfe)

    counts = initial_counts()
    wins: list[bool] = []
    for _ in range(n_trials):
        env.reset()
        log_a = elog_a(counts)

        # Move 1: plan both steps with the current world model; sample the
        # control posterior (argmax is brittle on early near-uniform beliefs).
        st = plan2.state_with(log_a)
        st, _ = plan2.engine.run(st, n_sweeps, tol=1e-10, damp=damp)
        q_u1 = np.asarray(plan2.engine.marginals(st)["u1"])
        u1 = int(env.rng.choice(N_POS, p=q_u1 / q_u1.sum()))
        x1 = env.step(u1)

        # Move 2: replan with move 1 clamped as data.
        st = plan1.state_with(log_a)
        st = plan1.engine.bind_data(st, {"act1": u1, "obs1": x1})
        st, _ = plan1.engine.run(st, n_sweeps, tol=1e-10, damp=damp)
        q_u2 = np.asarray(plan1.engine.marginals(st)["u2"])
        u2 = int(env.rng.choice(N_POS, p=q_u2 / q_u2.sum()))
        x2 = env.step(u2)

        # Learning: smooth the completed trial, conjugate Dirichlet update.
        st = smooth.state_with(log_a)
        st = smooth.engine.bind_data(
            st, {"act1": u1, "obs1": x1, "act2": u2, "obs2": x2}
        )
        st, _ = smooth.engine.run(st, 60, tol=1e-12)
        q = smooth.engine.marginals(st)
        for k, x in ((1, x1), (2, x2)):
            counts[x] += np.asarray(q[f"zm{k}"])

        wins.append(x1 % N_OUT == RW or x2 % N_OUT == RW)
    return LearningResult(wins=wins, counts=counts)

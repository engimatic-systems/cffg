"""The receding-horizon agent runtime (M5).

Two compiled engines cover an unbounded episode:

- **plan** — `build_rollout(past=0, steps=H)`: GFE planning window from the
  current belief; read the first control posterior.
- **filter** — `build_rollout(past=1, steps=1)`: one perception slice; clamp
  the executed control and received observation, and the smoothed state
  marginal becomes the next belief. This *is* the receding-horizon
  constraint flip: the window slides by re-anchoring the prior, not by
  growing the graph.

The belief d is a per-member parameter update (Arena.member_of), so ticks
never recompile — the same property that lets a Colony vmap many agents.
Optional continual learning: with `learn_a=True` the observation model is a
Dirichlet count array updated conjugately each tick and fed to the planner's
learned-GFE composites; `checkpoint`/`restore` round-trip belief + counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import numpy as np

from cffg.compiler import compile_template
from cffg.engine import Engine, State, _replace

Builder = Callable[..., tuple]  # build_rollout-style: (past, steps) -> (Template, arrays)


def _elog(counts: np.ndarray) -> np.ndarray:
    from jax.scipy.special import digamma

    return np.asarray(digamma(jnp.asarray(counts))) - np.asarray(
        digamma(jnp.asarray(counts.sum(axis=0, keepdims=True)))
    )


class RollingAgent:
    def __init__(
        self,
        builder: Builder,
        horizon: int,
        *,
        learn_a: bool = False,
        initial_counts: np.ndarray | None = None,
        damp: float = 0.5,
        n_sweeps: int = 400,
        select: str = "map",
        seed: int = 0,
    ):
        self.horizon, self.damp, self.n_sweeps, self.select = horizon, damp, n_sweeps, select
        self.learn_a = learn_a
        self.rng = np.random.default_rng(seed)

        plan_t, plan_arrays = builder(past=0, steps=horizon, learned_a=learn_a)
        filt_t, filt_arrays = builder(past=1, steps=1, learned_a=False)
        self.plan = Engine(compile_template(plan_t, arrays=plan_arrays))
        self.filter = Engine(compile_template(filt_t, arrays=filt_arrays))

        self.belief = np.asarray(plan_arrays["d"], dtype=float).copy()
        self.counts = None if not learn_a else np.asarray(initial_counts, dtype=float).copy()
        self.n_ctrl = len(plan_arrays["e"])
        self.tick = 0

    # -- parameter plumbing (no recompiles, ever) -------------------------

    def _with_params(self, engine: Engine) -> State:
        st = engine.init_state()
        params = dict(st.params)
        key, idx = engine.arena.member_of["prior"]
        log_d = np.log(np.maximum(self.belief, 1e-300))
        params[key] = params[key].at[idx].set(jnp.asarray(log_d))
        if self.counts is not None:
            for g in engine.arena.gfe_groups:
                if g.learned:
                    params[g.name + ":counts"] = jnp.asarray(
                        np.broadcast_to(self.counts, (len(g.members), *self.counts.shape))
                    )
            elog = None
            for g in engine.arena.groups:
                if g.type == "matrix_cat":
                    elog = _elog(self.counts) if elog is None else elog
                    params[g.name] = jnp.asarray(
                        np.broadcast_to(elog, (len(g.members), *elog.shape))
                    )
        return _replace(st, params=params)

    # -- the perception-action cycle --------------------------------------

    def act(self) -> int:
        """Plan over the horizon from the current belief; pick a control."""
        st = self._with_params(self.plan)
        st, _ = self.plan.run(st, self.n_sweeps, tol=1e-10, damp=self.damp)
        q = np.asarray(self.plan.marginals(st)["u1"])
        if self.select == "map":
            return int(np.argmax(q))
        return int(self.rng.choice(self.n_ctrl, p=q / q.sum()))

    def observe(self, action: int, observation: int) -> np.ndarray:
        """Fold (action, observation) into the belief; conjugate A update."""
        st = self._with_params(self.filter)
        st = self.filter.bind_data(st, {"act1": action, "obs1": observation})
        st, _ = self.filter.run(st, 60, tol=1e-12)
        q_z = np.asarray(self.filter.marginals(st)["zm1"])
        self.belief = q_z / q_z.sum()
        if self.counts is not None:
            self.counts[observation] += self.belief
        self.tick += 1
        return self.belief

    def step(self, env_step: Callable[[int], int]) -> tuple[int, int]:
        u = self.act()
        x = env_step(u)
        self.observe(u, x)
        return u, x

    # -- persistence -------------------------------------------------------

    def checkpoint(self, path: str | Path) -> None:
        data = {"belief": self.belief, "tick": np.asarray(self.tick)}
        if self.counts is not None:
            data["counts"] = self.counts
        np.savez(path, **data)

    def restore(self, path: str | Path) -> None:
        data = np.load(path)
        self.belief = data["belief"].copy()
        self.tick = int(data["tick"])
        if "counts" in data:
            self.counts = data["counts"].copy()

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

Continual learning runs on declared learnables (cff-x0sv, cffg.learn): the
model names its conjugate posteriors, and the agent only decides *cadence*,
which is a modeling choice the API deliberately leaves open:

- observation, transition and habit posteriors update every tick, from the
  perception graph's node beliefs, where the data is. The habit statistic is
  the executed control: the perception slice branches its control edge
  through an equality node so the habit prior sits in the same graph as the
  clamped action, which makes the node belief there a one-hot on what was
  actually done, not on what the planner considered;
- the initial-state prior updates once per episode — it is the belief the
  episode starts from, so the statistic is the first tick's smoothed q(z0).

`act()` does not change the current trial's priors. For learned goals it stages
the latest converged planning posterior, and only `reset_episode()` commits that
posterior as the next trial's prior. Planning an action is not evidence that it
was taken.

The A path feeds the planner's learned-GFE composites (epistemic value of
A-learning, Part II §5). The B path has no GFE channel unless the model asks
for one: dynamics uncertainty generates epistemic value only through the
GFE-TM composite (LIMITATIONS S1c).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import numpy as np

from cffg.compiler import _log_of, compile_template
from cffg.engine import Engine, State, _replace
from cffg.learn import Learnables

Builder = Callable[..., tuple]  # build_rollout-style: (past, steps) -> (Template, arrays)


class RollingAgent:
    def __init__(
        self,
        builder: Builder,
        horizon: int,
        *,
        learn: tuple[str, ...] = (),
        priors: dict[str, np.ndarray] | None = None,
        episode_learnables: tuple[str, ...] = ("d",),
        learn_a: bool = False,
        initial_counts: np.ndarray | None = None,
        learn_b: bool = False,
        initial_b_counts: np.ndarray | None = None,
        learn_goal: bool = False,
        initial_goal_counts: np.ndarray | None = None,
        epistemic_b: bool = False,
        damp: float = 0.5,
        n_sweeps: int = 400,
        select: str = "map",
        seed: int = 0,
    ):
        """`learn` names the learnables the builder should declare; `priors`
        gives their initial hyperparameters. `learn_a` / `learn_b` are the
        older spelling of `learn=("A",)` / `learn=("B",)` with the counts
        passed separately, kept working because callers and checkpoints use
        them.

        `learn_goal` enables the separate Fig. 5 goal-Dirichlet coordinate;
        its posterior is staged by planning and committed at a reset boundary.

        `episode_learnables` names those whose statistic is an episode
        quantity rather than a per-tick one — the initial-state prior, by
        construction. They accumulate on the first tick after a reset."""
        self.horizon, self.damp, self.n_sweeps, self.select = horizon, damp, n_sweeps, select
        self.rng = np.random.default_rng(seed)

        priors = dict(priors or {})
        learn = tuple(learn)
        if learn_a:
            learn += ("A",) if "A" not in learn else ()
            priors.setdefault("A", initial_counts)
        if learn_b:
            learn += ("B",) if "B" not in learn else ()
            priors.setdefault("B", initial_b_counts)
        if epistemic_b and "B" not in learn:
            raise ValueError("epistemic_b needs B counts to score novelty")
        if learn_goal and "A" in learn:
            raise ValueError(
                "learn_goal and learned A are separate composite coordinates for now"
            )
        self.learn = learn
        self.learn_goal = learn_goal

        plan_kwargs = {
            "learned_a": "A" in learn,
            "learned_c": learn_goal,
        }
        if learn_goal:
            plan_kwargs["c_counts"] = initial_goal_counts
        if epistemic_b:
            plan_kwargs["epistemic_b"] = True
        if learn:
            plan_kwargs["learn"] = learn
        plan_t, plan_arrays = builder(past=0, steps=horizon, **plan_kwargs)
        filt_kwargs = {"learned_a": False}
        if learn:
            filt_kwargs["learn"] = learn
        filt_t, filt_arrays = builder(past=1, steps=1, **filt_kwargs)
        self.plan = Engine(compile_template(plan_t, arrays=plan_arrays))
        self.filter = Engine(compile_template(filt_t, arrays=filt_arrays))

        self.belief = np.asarray(plan_arrays["d"], dtype=float).copy()
        self.learnables = Learnables(self.plan, priors={
            k: v for k, v in priors.items() if v is not None
        })
        # Cadence: everything the perception graph can supply statistics for,
        # minus the episode-scale posteriors, updates per tick.
        filter_sources = {
            p.name for p in self.filter.arena.learn_plans if p.sources
        }
        self.episode_learnables = tuple(
            n for n in episode_learnables if n in self.learnables.params
        )
        self.tick_learnables = tuple(
            sorted(filter_sources - set(self.episode_learnables))
        )
        self.n_ctrl = len(plan_arrays["e"])
        self.goal_priors = (
            {"Cc": np.asarray(plan_arrays["Cc"], dtype=float).copy()}
            if learn_goal
            else {}
        )
        self._pending_goal_priors: dict[str, np.ndarray] | None = None
        self.tick = 0
        self.episode_tick = 0

    # -- parameter plumbing (no recompiles, ever) -------------------------

    def _with_params(self, engine: Engine) -> State:
        st = engine.init_state()
        params = self.learnables.write(engine, dict(st.params))
        # The prior slot is written LAST and wins: within an episode it holds
        # the current belief, not the learned initial-state prior. That
        # re-anchoring is the receding-horizon flip; the learned d is what a
        # reset_episode() re-anchors *to*.
        key, idx = engine.arena.member_of["prior"]
        # Exact-zero belief entries stay semantically zero (NEG, the engine's
        # finite log-zero) instead of underflowing to log(0) = -inf when the
        # consumer belief is float32 (cff-ey4p).
        log_d = _log_of(np.asarray(self.belief))
        params[key] = params[key].at[idx].set(jnp.asarray(log_d))
        st = _replace(st, params=params)
        if engine is self.plan and self.goal_priors:
            st = engine.with_goal_priors(st, self.goal_priors)
        return st

    # -- the perception-action cycle --------------------------------------

    def act(self) -> int:
        """Plan over the horizon from the current belief; pick a control."""
        st = self._with_params(self.plan)
        st = self.plan.run(
            st, self.n_sweeps, tol=1e-10, damp=self.damp
        ).require_converged()
        # Planning remains read-only within a trial. Learned goals stage one
        # converged Fig.5 posterior, but only reset_episode's explicit trial
        # boundary rolls it into the next prior. Replanning merely replaces
        # the staged value; it never accumulates another sweep/trial.
        if self.goal_priors:
            self._pending_goal_priors = self.plan.goal_prior_updates(st)
        q = np.asarray(self.plan.marginals(st)["u1"])
        if self.select == "map":
            return int(np.argmax(q))
        return int(self.rng.choice(self.n_ctrl, p=q / q.sum()))

    def observe(self, action: int, observation: int) -> np.ndarray:
        """Fold (action, observation) into the belief; conjugate updates."""
        st = self._with_params(self.filter)
        st = self.filter.bind_data(st, {"act1": action, "obs1": observation})
        st = self.filter.run(st, 60, tol=1e-12).require_converged()
        q_z = np.asarray(self.filter.marginals(st)["zm1"])
        if self.tick_learnables:
            self.learnables.accumulate(self.filter, st, only=self.tick_learnables)
        if self.episode_tick == 0 and self.episode_learnables:
            # q(z0) smoothed by this episode's first observation — the
            # sufficient statistic for where episodes start.
            self.learnables.accumulate(self.filter, st, only=self.episode_learnables)
        self.belief = q_z / q_z.sum()
        self.tick += 1
        self.episode_tick += 1
        return self.belief

    def step(self, env_step: Callable[[int], int]) -> tuple[int, int]:
        u = self.act()
        x = env_step(u)
        self.observe(u, x)
        return u, x

    def reset_episode(self) -> np.ndarray:
        """Start a new episode: re-anchor the belief on the learned
        initial-state prior (its posterior mean), or leave it untouched when
        the model does not learn one."""
        if self._pending_goal_priors is not None:
            self.goal_priors = {
                name: alpha.copy()
                for name, alpha in self._pending_goal_priors.items()
            }
            self._pending_goal_priors = None
        for name in self.episode_learnables:
            self.belief = self.learnables.expected(name).copy()
        self.episode_tick = 0
        return self.belief

    # -- persistence -------------------------------------------------------

    # Checkpoints before the learnables API named the two arrays it could
    # learn. Restoring one has to map them onto their learnables, or say so:
    # silently keeping the constructor priors would look like a resumed agent
    # that had forgotten everything.
    _LEGACY_KEYS = {"counts": "A", "b_counts": "B"}

    def checkpoint(self, path: str | Path) -> None:
        data = {
            "belief": self.belief,
            "tick": np.asarray(self.tick),
            # Cadence is state. Without it, a restore mid-episode reads the
            # next observation as an episode start and folds a second
            # initial-state statistic from the same episode.
            "episode_tick": np.asarray(self.episode_tick),
        }
        for name, alpha in self.learnables.state_dict().items():
            data[f"learn:{name}"] = alpha
        for name, alpha in self.goal_priors.items():
            data[f"goal:prior:{name}"] = alpha
        for name, alpha in (self._pending_goal_priors or {}).items():
            data[f"goal:pending:{name}"] = alpha
        np.savez(path, **data)

    def restore(self, path: str | Path) -> None:
        data = np.load(path)
        self.belief = data["belief"].copy()
        self.tick = int(data["tick"])
        # A checkpoint predating episode cadence never reset, so its ticks are
        # one episode. Falling back to `tick` is the conservative read: it can
        # only decline to fire the episode-start statistic, never fire it twice.
        self.episode_tick = (
            int(data["episode_tick"]) if "episode_tick" in data.files else self.tick
        )
        state = {
            k[len("learn:") :]: data[k] for k in data.files if k.startswith("learn:")
        }
        for old, name in self._LEGACY_KEYS.items():
            if old not in data.files:
                continue
            if name not in self.learnables.params:
                raise KeyError(
                    f"checkpoint holds legacy {old!r} counts but this agent does "
                    f"not learn {name!r}"
                )
            state[name] = data[old]
        self.learnables.load_state_dict(state)
        restored_goal = {
            key[len("goal:prior:") :]: data[key].copy()
            for key in data.files
            if key.startswith("goal:prior:")
        }
        if self.goal_priors:
            if set(restored_goal) != set(self.goal_priors):
                raise KeyError(
                    "checkpoint learned-goal priors do not match this agent: "
                    f"have {sorted(restored_goal)}, expected {sorted(self.goal_priors)}"
                )
            self.goal_priors = restored_goal
            pending = {
                key[len("goal:pending:") :]: data[key].copy()
                for key in data.files
                if key.startswith("goal:pending:")
            }
            unknown_pending = sorted(set(pending) - set(self.goal_priors))
            if unknown_pending:
                raise KeyError(
                    f"checkpoint has unknown pending goal priors {unknown_pending}"
                )
            self._pending_goal_priors = pending or None
        elif restored_goal:
            raise KeyError(
                "checkpoint contains learned-goal priors but this agent has "
                "learn_goal=False"
            )

    # -- compatibility readouts -------------------------------------------

    def _alpha(self, name: str) -> np.ndarray | None:
        params = self.learnables.params.get(name)
        return None if params is None else params["alpha"]

    @property
    def counts(self) -> np.ndarray | None:
        """Observation-model Dirichlet counts (the old `learn_a` array)."""
        return self._alpha("A")

    @property
    def b_counts(self) -> np.ndarray | None:
        """Transition Dirichlet counts [U, K', K] (the old `learn_b` array)."""
        return self._alpha("B")

    @property
    def goal_counts(self) -> np.ndarray | None:
        """Shared goal Dirichlet prior, rolled only at episode boundaries."""
        value = self.goal_priors.get("Cc")
        return None if value is None else value.copy()

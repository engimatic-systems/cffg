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

import json
from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import numpy as np

from cffg.compiler import _log_of, compile_template
from cffg.core.ir import AgentSpec, BeliefBinding, Template
from cffg.core.validate import validate
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
        episode_learnables: tuple[str, ...] | None = None,
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

        Episode-scale cadence comes from each serialized belief binding's
        ``episode_learnable``. The old ``episode_learnables`` argument remains
        as a compatibility assertion: when supplied it must match what the
        model declares."""
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
        plan_spec = _require_agent_spec(plan_t, "planning")
        filter_spec = _require_agent_spec(filt_t, "perception")
        validate(plan_t)
        validate(filt_t)
        _check_window_compatibility(plan_t, plan_spec, filt_t, filter_spec)

        self.plan = Engine(compile_template(plan_t, arrays=plan_arrays))
        self.filter = Engine(compile_template(filt_t, arrays=filt_arrays))
        self._plan_spec = plan_spec
        self._filter_spec = filter_spec
        self._plan_bindings = {b.name: b for b in plan_spec.beliefs}
        self._filter_bindings = {b.name: b for b in filter_spec.beliefs}
        self.beliefs = {
            binding.name: _initial_belief(plan_t, plan_arrays, binding)
            for binding in plan_spec.beliefs
        }

        self.learnables = Learnables(self.plan, priors={
            k: v for k, v in priors.items() if v is not None
        })
        # Cadence: everything the perception graph can supply statistics for,
        # minus the episode-scale posteriors, updates per tick.
        filter_sources = {
            p.name for p in self.filter.arena.learn_plans if p.sources
        }
        declared_episode = tuple(
            sorted(
                {
                    binding.episode_learnable
                    for binding in plan_spec.beliefs
                    if binding.episode_learnable is not None
                }
            )
        )
        if episode_learnables is not None and set(episode_learnables) != set(
            declared_episode
        ):
            raise ValueError(
                "episode_learnables does not match the Template AgentSpec: "
                f"argument {sorted(episode_learnables)}, model "
                f"{list(declared_episode)}"
            )
        self.episode_learnables = declared_episode
        self.tick_learnables = tuple(
            sorted(filter_sources - set(self.episode_learnables))
        )
        self.n_ctrl = plan_t.variables[plan_spec.action].cardinality
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
        bindings = (
            self._plan_bindings if engine is self.plan else self._filter_bindings
        )
        # Boundary prior slots are written LAST and win over learnable writes:
        # within an episode they carry the current filtered beliefs. An
        # episode learnable only re-anchors its declared binding at reset.
        for name, binding in bindings.items():
            address = engine.arena.member_of[binding.prior].parameter("statistic")
            # Exact-zero entries stay semantically zero (NEG, the engine's
            # finite log-zero) rather than passing through log(0).
            log_belief = _log_of(np.asarray(self.beliefs[name]))
            params[address.key] = params[address.key].at[address.index].set(
                jnp.asarray(log_belief)
            )
        st = _replace(st, params=params)
        if engine is self.plan and self.goal_priors:
            st = engine.with_goal_priors(st, self.goal_priors)
        return st

    # -- the perception-action cycle --------------------------------------

    def act(self) -> int:
        """Plan over the horizon from the current belief; pick a control."""
        st = self._with_params(self.plan)
        run_options = (
            {}
            if self.plan.arena.gfe_groups or self.plan.arena.gfe_tm_groups
            else {"damp": self.damp}
        )
        st = self.plan.run(
            st, self.n_sweeps, tol=1e-10, **run_options
        ).require_converged()
        # Planning remains read-only within a trial. Learned goals stage one
        # converged Fig.5 posterior, but only reset_episode's explicit trial
        # boundary rolls it into the next prior. Replanning merely replaces
        # the staged value; it never accumulates another sweep/trial.
        if self.goal_priors:
            self._pending_goal_priors = self.plan.goal_prior_updates(st)
        q = np.asarray(self.plan.marginals(st)[self._plan_spec.action])
        if self.select == "map":
            return int(np.argmax(q))
        return int(self.rng.choice(self.n_ctrl, p=q / q.sum()))

    def observe(
        self, action: int, observation: int
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Fold (action, observation) into the belief; conjugate updates."""
        st = self._with_params(self.filter)
        st = self.filter.bind_data(
            st,
            {
                self._filter_spec.action_port: action,
                self._filter_spec.observation_port: observation,
            },
        )
        st = self.filter.run(st, 60, tol=1e-12).require_converged()
        marginals = self.filter.marginals(st)
        next_beliefs = {
            name: _normalize(
                np.asarray(marginals[binding.posterior]),
                f"filter posterior {binding.posterior!r}",
            )
            for name, binding in self._filter_bindings.items()
        }
        if self.tick_learnables:
            self.learnables.accumulate(self.filter, st, only=self.tick_learnables)
        if self.episode_tick == 0 and self.episode_learnables:
            # q(z0) smoothed by this episode's first observation — the
            # sufficient statistic for where episodes start.
            self.learnables.accumulate(self.filter, st, only=self.episode_learnables)
        self.beliefs = next_beliefs
        self.tick += 1
        self.episode_tick += 1
        return self._belief_result()

    def step(self, env_step: Callable[[int], int]) -> tuple[int, int]:
        u = self.act()
        x = env_step(u)
        self.observe(u, x)
        return u, x

    def reset_episode(self) -> np.ndarray | dict[str, np.ndarray]:
        """Start a new episode and re-anchor each declared learned boundary."""
        if self._pending_goal_priors is not None:
            self.goal_priors = {
                name: alpha.copy()
                for name, alpha in self._pending_goal_priors.items()
            }
            self._pending_goal_priors = None
        for name, binding in self._plan_bindings.items():
            if binding.episode_learnable is not None:
                self.beliefs[name] = _normalize(
                    self.learnables.expected(binding.episode_learnable).copy(),
                    f"episode learnable {binding.episode_learnable!r}",
                )
        self.episode_tick = 0
        return self._belief_result()

    # -- persistence -------------------------------------------------------

    # Checkpoints before the learnables API named the two arrays it could
    # learn. Restoring one has to map them onto their learnables, or say so:
    # silently keeping the constructor priors would look like a resumed agent
    # that had forgotten everything.
    _LEGACY_KEYS = {"counts": "A", "b_counts": "B"}

    def checkpoint(self, path: str | Path) -> None:
        data = {
            "tick": np.asarray(self.tick),
            # Cadence is state. Without it, a restore mid-episode reads the
            # next observation as an episode start and folds a second
            # initial-state statistic from the same episode.
            "episode_tick": np.asarray(self.episode_tick),
            "rng_state": np.asarray(json.dumps(self.rng.bit_generator.state)),
        }
        if self._has_legacy_belief:
            # Preserve the legacy checkpoint spelling byte-for-byte for the
            # public single-boundary API.
            data["belief"] = self.belief
        else:
            for name, belief in self.beliefs.items():
                data[f"belief:{name}"] = belief
        for name, alpha in self.learnables.state_dict().items():
            data[f"learn:{name}"] = alpha
        for name, alpha in self.goal_priors.items():
            data[f"goal:prior:{name}"] = alpha
        for name, alpha in (self._pending_goal_priors or {}).items():
            data[f"goal:pending:{name}"] = alpha
        np.savez(path, **data)

    def restore(self, path: str | Path) -> None:
        data = np.load(path)
        if "belief" in data.files:
            if not self._has_legacy_belief:
                raise KeyError(
                    "legacy single-belief checkpoint cannot restore a "
                    "multi-block agent"
                )
            only = next(iter(self.beliefs))
            self.beliefs[only] = _check_belief(
                data["belief"].copy(), f"checkpoint belief {only!r}"
            )
        else:
            restored_beliefs = {
                key[len("belief:") :]: data[key].copy()
                for key in data.files
                if key.startswith("belief:")
            }
            if set(restored_beliefs) != set(self.beliefs):
                raise KeyError(
                    "checkpoint boundary beliefs do not match this agent: "
                    f"have {sorted(restored_beliefs)}, expected "
                    f"{sorted(self.beliefs)}"
                )
            self.beliefs = {
                name: _check_belief(value, f"checkpoint belief {name!r}")
                for name, value in restored_beliefs.items()
            }
        self.tick = int(data["tick"])
        if "rng_state" in data.files:
            self.rng.bit_generator.state = json.loads(str(data["rng_state"]))
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

    @property
    def _has_legacy_belief(self) -> bool:
        return (
            len(self._plan_spec.beliefs) == 1
            and len(self._plan_spec.beliefs[0].blocks) == 1
        )

    def _belief_result(self) -> np.ndarray | dict[str, np.ndarray]:
        if self._has_legacy_belief:
            return self.belief
        return {name: value.copy() for name, value in self.beliefs.items()}

    @property
    def belief(self) -> np.ndarray:
        """Compatibility view for models with exactly one state block."""
        if not self._has_legacy_belief:
            raise AttributeError(
                "belief is only available for a single boundary block; use beliefs"
            )
        return next(iter(self.beliefs.values()))

    @belief.setter
    def belief(self, value: np.ndarray) -> None:
        if not self._has_legacy_belief:
            raise AttributeError(
                "belief is only available for a single boundary block; use beliefs"
            )
        name = next(iter(self.beliefs))
        self.beliefs[name] = np.asarray(value)

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


def _require_agent_spec(template: Template, window: str) -> AgentSpec:
    if template.agent is None:
        raise ValueError(
            f"{window} template {template.name!r} does not declare an AgentSpec"
        )
    return template.agent


def _check_window_compatibility(
    plan_t: Template,
    plan: AgentSpec,
    filter_t: Template,
    filt: AgentSpec,
) -> None:
    """Cross-template checks that core validation cannot make in isolation."""
    if filt.action_port is None or filt.observation_port is None:
        raise ValueError(
            f"perception template {filter_t.name!r} must declare action and "
            "observation DATA ports"
        )
    plan_action = plan_t.variables[plan.action]
    filter_action = filter_t.variables[filter_t.ports[filt.action_port].variable]
    if plan_action.domain_key != filter_action.domain_key:
        raise ValueError(
            f"agent action domain mismatch: planning {plan_action.domain_key}, "
            f"perception {filter_action.domain_key}"
        )

    plan_bindings = {b.name: b for b in plan.beliefs}
    filter_bindings = {b.name: b for b in filt.beliefs}
    if set(plan_bindings) != set(filter_bindings):
        raise ValueError(
            "planning/perception boundary names differ: "
            f"{sorted(plan_bindings)} != {sorted(filter_bindings)}"
        )
    for name in sorted(plan_bindings):
        before, after = plan_bindings[name], filter_bindings[name]
        if (
            before.blocks != after.blocks
            or before.factorization is not after.factorization
            or before.episode_learnable != after.episode_learnable
        ):
            raise ValueError(
                f"boundary {name!r} differs between planning and perception: "
                "blocks, factorization and episode learnable must agree"
            )
        plan_prior = plan_t.variables[plan_t.factors[before.prior].edges["out"]]
        filter_prior = filter_t.variables[
            filter_t.factors[after.prior].edges["out"]
        ]
        filter_posterior = filter_t.variables[after.posterior]
        if not (
            plan_prior.domain_key
            == filter_prior.domain_key
            == filter_posterior.domain_key
        ):
            raise ValueError(
                f"boundary {name!r} changes domain across rollover: planning "
                f"{plan_prior.domain_key}, perception prior "
                f"{filter_prior.domain_key}, posterior "
                f"{filter_posterior.domain_key}"
            )


def _initial_belief(
    template: Template, arrays: dict[str, np.ndarray], binding: BeliefBinding
) -> np.ndarray:
    prior = template.factors[binding.prior]
    blob = prior.param_blobs["statistic"]
    if blob not in arrays:
        raise KeyError(
            f"agent belief {binding.name!r}: prior blob {blob!r} has no array"
        )
    return _normalize(
        np.asarray(arrays[blob]).copy(),
        f"agent belief {binding.name!r} prior {binding.prior!r}",
    )


def _normalize(value: np.ndarray, label: str) -> np.ndarray:
    """Normalize a categorical boundary without changing exact-zero support."""
    value = _check_belief(value, label)
    return value / value.sum()


def _check_belief(value: np.ndarray, label: str) -> np.ndarray:
    """Validate categorical mass while preserving checkpoint bits exactly."""
    value = np.asarray(value)
    if value.ndim != 1:
        raise ValueError(f"{label} must be a 1-D categorical belief, got {value.shape}")
    if not np.all(np.isfinite(value)) or np.any(value < 0):
        raise ValueError(f"{label} must be finite and non-negative")
    total = value.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"{label} must have positive finite mass")
    return value

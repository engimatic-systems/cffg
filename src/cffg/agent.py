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

import inspect
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from cffg._agent_checkpoint import (
    check_belief,
    checkpoint_agent,
    restore_agent,
    static_parameter_hash,
)
from cffg._agent_mu3 import PlanningRuntime, functional
from cffg.compiler import _log_of, compile_template
from cffg.core.ir import AgentSpec, BeliefBinding, Template
from cffg.core.plates import expand_plates
from cffg.core.serialize import structure_hash
from cffg.core.validate import validate
from cffg.engine import Engine, InferenceError, RunResult, RunStatus, State, _replace
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
        mu3: bool = False,
        mu3_key: jax.Array | None = None,
        mu3_samples: int = 256,
        mu3_min_ess_fraction: float = 0.1,
        mu3_refresh: str = "trial",
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
        self._jax_enable_x64 = bool(jax.config.jax_enable_x64)
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
        builder_accepts_mu3 = _accepts_keyword(builder, "mu3")
        if mu3 and not builder_accepts_mu3:
            raise ValueError(
                "mu3=True requires a rollout builder that accepts the mu3 keyword"
            )
        if mu3:
            plan_kwargs["mu3"] = True
        if learn:
            plan_kwargs["learn"] = learn
        plan_t, plan_arrays = builder(past=0, steps=horizon, **plan_kwargs)
        # A builder may be a partial with mu3=True. Perception never owns the
        # functional planning message, so override it explicitly here.
        filt_kwargs = {"learned_a": False}
        if builder_accepts_mu3:
            filt_kwargs["mu3"] = False
        if learn:
            filt_kwargs["learn"] = learn
        filt_t, filt_arrays = builder(past=1, steps=1, **filt_kwargs)
        plan_spec = _require_agent_spec(plan_t, "planning")
        filter_spec = _require_agent_spec(filt_t, "perception")
        validate(plan_t)
        validate(filt_t)
        _check_window_compatibility(plan_t, plan_spec, filt_t, filter_spec)
        self._plan_structure_hash = structure_hash(plan_t)
        self._filter_structure_hash = structure_hash(filt_t)
        grounded_plan = expand_plates(plan_t)
        grounded_filter = expand_plates(filt_t)

        self.plan = Engine(
            compile_template(grounded_plan, arrays=plan_arrays),
            mu3_samples=mu3_samples,
            mu3_min_ess_fraction=mu3_min_ess_fraction,
        )
        self.filter = Engine(compile_template(grounded_filter, arrays=filt_arrays))
        if self.filter.arena.mu3_groups:
            raise ValueError("perception graph must not declare a mu3 evaluator")
        if mu3 and not functional(self.plan):
            raise ValueError(
                "mu3=True requested functional planning, but the serialized "
                "planning constraints declare no evaluator"
            )
        self._planning = PlanningRuntime(mu3_key, refresh_policy=mu3_refresh)
        self._plan_parameter_hash = static_parameter_hash(
            grounded_plan,
            plan_arrays,
            _require_agent_spec(grounded_plan, "grounded planning"),
            goal_priors={
                factor
                for group in self.plan.arena.gfe_groups
                for factor in group.goal_priors
            },
        )
        self._filter_parameter_hash = static_parameter_hash(
            grounded_filter,
            filt_arrays,
            _require_agent_spec(grounded_filter, "grounded perception"),
            goal_priors=set(),
        )
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
        self._last_filter_result: RunResult | None = None
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
        result = self._planning.run(
            self.plan,
            st,
            self.n_sweeps,
            tol=1e-10,
            run_options=run_options,
        )
        st = result.require_converged()
        if any(status is not RunStatus.CONVERGED for status in result.mu3_statuses):
            raise InferenceError(result)
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
        result = self.filter.run(st, 60, tol=1e-12)
        st = result.require_converged()
        marginals = self.filter.marginals(st)
        next_beliefs = {
            name: _normalize(
                np.asarray(marginals[binding.posterior]),
                f"filter posterior {binding.posterior!r}",
            )
            for name, binding in self._filter_bindings.items()
        }
        if self.tick_learnables:
            self.learnables.accumulate(
                self.filter, result, only=self.tick_learnables
            )
        if self.episode_tick == 0 and self.episode_learnables:
            # q(z0) smoothed by this episode's first observation — the
            # sufficient statistic for where episodes start.
            self.learnables.accumulate(
                self.filter, result, only=self.episode_learnables
            )
        self.beliefs = next_beliefs
        self.tick += 1
        self.episode_tick += 1
        self._last_filter_result = result
        self._planning.invalidate()
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
        self._planning.invalidate()
        return self._belief_result()

    # -- persistence -------------------------------------------------------

    def checkpoint(self, path: str | Path) -> None:
        checkpoint_agent(self, path)

    def restore(self, path: str | Path) -> None:
        restore_agent(self, path)

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

    @property
    def functional_mu3(self) -> bool:
        """Whether serialized planning constraints select a functional evaluator."""
        return functional(self.plan)

    @property
    def last_plan_result(self) -> RunResult | None:
        return self._planning.result

    @property
    def last_filter_result(self) -> RunResult | None:
        return self._last_filter_result

    @property
    def planning_key_cursor(self) -> jax.Array | None:
        return self._planning.key_cursor

    @property
    def planning_epoch(self) -> int:
        return self._planning.epoch

    @property
    def planning_diagnostics(self) -> dict:
        if not self.functional_mu3 or self.last_plan_result is None:
            return {}
        return self.plan.mu3_diagnostics(self.last_plan_result.state)

    def refresh_mu3(self) -> None:
        """Declare the next act as an independent fixed-particle epoch."""
        if not self.functional_mu3:
            raise ValueError("this planning graph has no functional mu3 evaluator")
        self._planning.refresh()


def _require_agent_spec(template: Template, window: str) -> AgentSpec:
    if template.agent is None:
        raise ValueError(
            f"{window} template {template.name!r} does not declare an AgentSpec"
        )
    return template.agent


def _accepts_keyword(builder: Builder, name: str) -> bool:
    """Keep older strict builders compatible while supporting plan-only mu3."""
    try:
        parameters = inspect.signature(builder).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


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
    value = check_belief(value, label)
    return value / value.sum()

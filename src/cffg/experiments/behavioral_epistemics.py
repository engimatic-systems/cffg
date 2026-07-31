"""Behavioral calibration for observation epistemics on a binary cue task.

The task separates four first actions: exploit left, exploit right, inspect a
reward-relevant cue, or inspect a random distractor with the same uniform
observation marginal.  A two-slice rollout makes sensing useful only when the
agent replans after the first observation (the finite-horizon posterior over
controls is factorized, as documented in LIMITATIONS H5).

This module intentionally keeps its behavioral oracles outside the message
rules.  The Bayes sensing threshold comes from the simulator's reward table;
mutual information and realized information gain come from normalized joint
probabilities.  Neither reads engine messages or local GFE terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cffg.compiler import compile_template
from cffg.core import Template
from cffg.engine import Engine, RunStatus
from cffg.models.rollout import build_rollout

# Positions.  The latent state is position x binary reward context.
START, LEFT_ARM, RIGHT_ARM, CUE, DISTRACTOR, TERMINAL = range(6)
N_POSITION, N_CONTEXT = 6, 2

# Controls.  Outside an arm/terminal state, each control moves to its namesake
# position.  Arms terminate on the following slice, so a direct exploit earns
# at most one reward while a cue-first plan can still exploit on slice two.
GO_LEFT, GO_RIGHT, CHECK_CUE, CHECK_DISTRACTOR = range(4)
N_ACTION = 4
ACTION_POSITION = (LEFT_ARM, RIGHT_ARM, CUE, DISTRACTOR)

# Observations.  Position is implicit in the disjoint observation categories.
NEUTRAL, REWARD, LOSS, CUE_LEFT, CUE_RIGHT, DISTRACTOR_LEFT, DISTRACTOR_RIGHT = range(7)
N_OBSERVATION = 7


@dataclass(frozen=True)
class CueTask:
    """Parameters in log-preference utility units.

    ``cue_cost`` is paid by either sensing action.  ``reward_utility`` gives
    reward/loss utilities ``(+r, -r)``.  With a uniform context prior, the
    Bayes value of sensing before exploiting is

        (2 cue_quality - 1) (2 arm_quality - 1) reward_utility.
    """

    cue_quality: float = 0.9
    arm_quality: float = 1.0
    cue_cost: float = 0.25
    reward_utility: float = 1.5

    def __post_init__(self) -> None:
        for name in ("cue_quality", "arm_quality", "cue_cost", "reward_utility"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)}")
        for name in ("cue_quality", "arm_quality"):
            value = getattr(self, name)
            if not 0.5 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0.5, 1.0], got {value}")
        if self.cue_cost < 0:
            raise ValueError(f"cue_cost must be non-negative, got {self.cue_cost}")
        if self.reward_utility <= 0:
            raise ValueError(
                f"reward_utility must be positive, got {self.reward_utility}"
            )

    @property
    def bayes_sensing_threshold(self) -> float:
        """Largest cue cost for which adaptive sensing improves task return."""
        return (
            (2.0 * self.cue_quality - 1.0)
            * (2.0 * self.arm_quality - 1.0)
            * self.reward_utility
        )

    @property
    def bayes_expected_return(self) -> float:
        """Expected return of the better of direct and cue-first policies."""
        return max(0.0, self.bayes_sensing_threshold - self.cue_cost)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / weights.sum()


def goal_statistic(task: CueTask, *, utility_offset: float = 0.0) -> np.ndarray:
    """Normalized observation preferences.

    ``utility_offset`` exists to make additive-utility invariance executable:
    adding a constant to every utility must not change a policy.
    """
    utilities = np.array(
        [
            0.0,
            task.reward_utility,
            -task.reward_utility,
            -task.cue_cost,
            -task.cue_cost,
            -task.cue_cost,
            -task.cue_cost,
        ],
        dtype=np.float64,
    )
    return _softmax(utilities + utility_offset)


def initial_state() -> np.ndarray:
    d = np.zeros(N_POSITION * N_CONTEXT)
    d[START * N_CONTEXT : (START + 1) * N_CONTEXT] = 0.5
    return d


def transition_stack() -> np.ndarray:
    """Exact-zero deterministic controlled transitions."""
    n_state = N_POSITION * N_CONTEXT
    matrices = np.zeros((N_ACTION, n_state, n_state))
    for action, destination in enumerate(ACTION_POSITION):
        for position in range(N_POSITION):
            next_position = (
                TERMINAL
                if position in (LEFT_ARM, RIGHT_ARM, TERMINAL)
                else destination
            )
            for context in range(N_CONTEXT):
                source = position * N_CONTEXT + context
                target = next_position * N_CONTEXT + context
                matrices[action, target, source] = 1.0
    return matrices


def observation_matrix(task: CueTask) -> np.ndarray:
    """Likelihood with an informative cue and a reward-independent distractor."""
    n_state = N_POSITION * N_CONTEXT
    matrix = np.zeros((N_OBSERVATION, n_state))
    for position in range(N_POSITION):
        for context in range(N_CONTEXT):
            state = position * N_CONTEXT + context
            if position in (START, TERMINAL):
                matrix[NEUTRAL, state] = 1.0
            elif position == LEFT_ARM:
                p_reward = (
                    task.arm_quality if context == 0 else 1.0 - task.arm_quality
                )
                matrix[REWARD, state] = p_reward
                matrix[LOSS, state] = 1.0 - p_reward
            elif position == RIGHT_ARM:
                p_reward = (
                    task.arm_quality if context == 1 else 1.0 - task.arm_quality
                )
                matrix[REWARD, state] = p_reward
                matrix[LOSS, state] = 1.0 - p_reward
            elif position == CUE:
                p_left = (
                    task.cue_quality if context == 0 else 1.0 - task.cue_quality
                )
                matrix[CUE_LEFT, state] = p_left
                matrix[CUE_RIGHT, state] = 1.0 - p_left
            else:
                # Same uniform observation marginal as the cue, but independent
                # of the reward context: H(X) is matched while I(context; X)=0.
                matrix[DISTRACTOR_LEFT, state] = 0.5
                matrix[DISTRACTOR_RIGHT, state] = 0.5
    return matrix


def build_costly_cue(
    task: CueTask = CueTask(),
    *,
    steps: int = 2,
    past: int = 0,
    gfe: bool = True,
    utility_offset: float = 0.0,
) -> tuple[Template, dict[str, np.ndarray]]:
    """Build the serialized-source rollout for the costly-cue intervention."""
    return build_rollout(
        f"costly-cue[{past}|{steps}]",
        steps=steps,
        past=past,
        d=initial_state(),
        A=observation_matrix(task),
        c=goal_statistic(task, utility_offset=utility_offset),
        e=np.full(N_ACTION, 1.0 / N_ACTION),
        Bs=transition_stack(),
        gfe=gfe,
    )


def _context_mi(likelihood: np.ndarray, prior: np.ndarray) -> float:
    """I(context; observation), evaluated directly from a normalized joint."""
    joint = likelihood * prior[None, :]
    p_observation = joint.sum(axis=1, keepdims=True)
    independent = p_observation * prior[None, :]
    support = joint > 0
    return float(np.sum(joint[support] * np.log(joint[support] / independent[support])))


def candidate_mutual_information(task: CueTask) -> np.ndarray:
    """Posterior MI available at each candidate first action."""
    matrix = observation_matrix(task)
    prior = np.full(N_CONTEXT, 1.0 / N_CONTEXT)
    values = []
    for position in ACTION_POSITION:
        columns = slice(position * N_CONTEXT, (position + 1) * N_CONTEXT)
        values.append(_context_mi(matrix[:, columns], prior))
    return np.asarray(values)


def _posterior_update(
    prior: np.ndarray, likelihood: np.ndarray
) -> tuple[np.ndarray, float]:
    unnormalized = prior * likelihood
    evidence = unnormalized.sum()
    if not evidence > 0:
        raise ValueError("observation has zero support under the context belief")
    posterior = unnormalized / evidence
    support = posterior > 0
    information_gain = float(
        np.sum(posterior[support] * np.log(posterior[support] / prior[support]))
    )
    return posterior, information_gain


@dataclass(frozen=True)
class DecisionPlan:
    action_probabilities: np.ndarray
    context_belief: np.ndarray
    status: RunStatus
    free_energy: float
    iterations: int
    residual: float
    messages_finite: bool

    @property
    def action(self) -> int:
        return int(np.argmax(self.action_probabilities))


def _engine(task: CueTask, *, past: int, gfe: bool, utility_offset: float = 0.0) -> Engine:
    template, arrays = build_costly_cue(
        task, past=past, gfe=gfe, utility_offset=utility_offset
    )
    return Engine(compile_template(template, arrays=arrays))


def _solve(
    engine: Engine,
    *,
    past: int,
    data: dict[str, int] | None = None,
    n_sweeps: int = 400,
) -> DecisionPlan:
    state = engine.init_state()
    if data:
        state = engine.bind_data(state, data)
    run_options = (
        {}
        if engine.arena.gfe_groups or engine.arena.gfe_tm_groups
        else {"damp": 0.5}
    )
    result = engine.run(state, n_sweeps, tol=1e-10, **run_options)
    state = result.require_converged()
    marginals = engine.marginals(state)
    action = np.asarray(marginals[f"u{past + 1}"])
    context_variable = "z0" if past == 0 else f"zt{past}"
    context = np.asarray(marginals[context_variable]).reshape(
        N_POSITION, N_CONTEXT
    ).sum(axis=0)
    return DecisionPlan(
        action_probabilities=action,
        context_belief=context,
        status=result.status,
        free_energy=engine.bfe(state),
        iterations=result.iterations,
        residual=result.residual,
        messages_finite=bool(np.isfinite(np.asarray(state.messages)).all()),
    )


def plan_decision(
    task: CueTask = CueTask(),
    *,
    past: int = 0,
    gfe: bool = True,
    data: dict[str, int] | None = None,
    utility_offset: float = 0.0,
    n_sweeps: int = 400,
) -> DecisionPlan:
    """Compile and solve one planning posture with explicit termination data."""
    return _solve(
        _engine(task, past=past, gfe=gfe, utility_offset=utility_offset),
        past=past,
        data=data,
        n_sweeps=n_sweeps,
    )


class CueTaskEnv:
    """Ground-truth simulator with the same serialized likelihood/transition tables."""

    def __init__(self, task: CueTask, rng: np.random.Generator):
        self.task = task
        self.rng = rng
        self.matrix = observation_matrix(task)
        self.context = 0
        self.position = START

    def reset(self) -> None:
        self.context = int(self.rng.integers(N_CONTEXT))
        self.position = START

    def step(self, action: int) -> tuple[int, float]:
        if not 0 <= action < N_ACTION:
            raise ValueError(f"action {action} out of range [0, {N_ACTION})")
        if self.position in (LEFT_ARM, RIGHT_ARM, TERMINAL):
            self.position = TERMINAL
        else:
            self.position = ACTION_POSITION[action]
        state = self.position * N_CONTEXT + self.context
        observation = int(self.rng.choice(N_OBSERVATION, p=self.matrix[:, state]))
        if observation == REWARD:
            task_return = self.task.reward_utility
        elif observation == LOSS:
            task_return = -self.task.reward_utility
        elif self.position in (CUE, DISTRACTOR):
            task_return = -self.task.cue_cost
        else:
            task_return = 0.0
        return observation, task_return


@dataclass(frozen=True)
class BehavioralReport:
    """Behavioral outputs required by cff-4oq8, grouped for ablation review."""

    gfe: bool
    first_plan: DecisionPlan
    conditional_plans: dict[int, DecisionPlan]
    candidate_posterior_mi: np.ndarray
    realized_information_gain_by_seed: np.ndarray
    task_return_by_seed: np.ndarray
    regret_by_seed: np.ndarray

    @property
    def convergence_statuses(self) -> tuple[RunStatus, ...]:
        return (
            self.first_plan.status,
            *(plan.status for plan in self.conditional_plans.values()),
        )

    @property
    def free_energies(self) -> tuple[float, ...]:
        return (
            self.first_plan.free_energy,
            *(plan.free_energy for plan in self.conditional_plans.values()),
        )


def run_benchmark(
    task: CueTask = CueTask(),
    *,
    gfe: bool = True,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
    episodes_per_seed: int = 256,
) -> BehavioralReport:
    """Replan after the first observation and report MI, return, and status."""
    if not seeds:
        raise ValueError("seeds must not be empty")
    if episodes_per_seed <= 0:
        raise ValueError("episodes_per_seed must be positive")

    first = _solve(_engine(task, past=0, gfe=gfe), past=0)
    first_position = ACTION_POSITION[first.action]
    matrix = observation_matrix(task)
    columns = slice(
        first_position * N_CONTEXT, (first_position + 1) * N_CONTEXT
    )
    reachable = np.flatnonzero(matrix[:, columns].sum(axis=1) > 0)

    conditional_engine = _engine(task, past=1, gfe=gfe)
    conditional = {
        int(observation): _solve(
            conditional_engine,
            past=1,
            data={"act1": first.action, "obs1": int(observation)},
        )
        for observation in reachable
    }

    returns = []
    information_gains = []
    for seed in seeds:
        env = CueTaskEnv(task, np.random.default_rng(seed))
        seed_returns = []
        seed_information_gains = []
        for _ in range(episodes_per_seed):
            env.reset()
            prior = np.full(N_CONTEXT, 1.0 / N_CONTEXT)
            observation1, return1 = env.step(first.action)
            likelihood = matrix[
                observation1,
                env.position * N_CONTEXT : (env.position + 1) * N_CONTEXT,
            ]
            _, information_gain = _posterior_update(prior, likelihood)
            second_action = conditional[observation1].action
            _, return2 = env.step(second_action)
            seed_returns.append(return1 + return2)
            seed_information_gains.append(information_gain)
        returns.append(float(np.mean(seed_returns)))
        information_gains.append(float(np.mean(seed_information_gains)))

    task_return = np.asarray(returns)
    return BehavioralReport(
        gfe=gfe,
        first_plan=first,
        conditional_plans=conditional,
        candidate_posterior_mi=candidate_mutual_information(task),
        realized_information_gain_by_seed=np.asarray(information_gains),
        task_return_by_seed=task_return,
        regret_by_seed=task.bayes_expected_return - task_return,
    )

"""Locked cff-5ipi/v3 factored-corridor experiment for cff-2gev.

This module intentionally exposes no protocol-tuning arguments.  The model,
arms, seeds, windows, bootstrap, traces, timing, and outcome gates were fixed
in ``.tickets/cff-2gev.md`` before the v3 replacement experiment was run.
Changing a constant here after observing a result voids that preregistration.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jaxlib
import numpy as np

from cffg import rules
from cffg.agent import PlanningResult, RollingAgent
from cffg.models import TransitionNoveltySpec, build_factored_corridor

PROTOCOL = "cff-5ipi/v3"
LENGTH = 8
MODES = 2
ACTIONS = 3
HORIZON = 8
MAX_ACTIONS = 14
N_SWEEPS = 400
KNOWN_N_SWEEPS = 2_000
KNOWN_DAMP = 0.85
N_HALF = 4.0
SATURATION_WEIGHT = 1 / 16
SEEDS = tuple(range(16))
N_EPISODES = 64
EARLY = slice(0, 16)
LATE = slice(48, 64)
OBSERVABILITY = (1.0, 0.75, 0.5)
ARMS = ("known_plain", "learn_only", "unannealed", "annealed")
BOOTSTRAP_SEED = 20260729
N_BOOTSTRAP = 10_000

POSITION_LEARNABLE = "position_transition"
POSITION_PRIOR = np.r_[1.0, np.zeros(LENGTH - 1)]
MODE_PRIOR = np.full(MODES, 1 / MODES)
ACTION_PRIOR = np.full(ACTIONS, 1 / ACTIONS)
DIRICHLET_PRIOR = np.full(
    (MODES * ACTIONS, LENGTH, LENGTH), 1 / LENGTH
)

_STREAM_MODE = 1
_STREAM_SENSOR = 2
_STREAM_ACTION = 3


def observation_matrix(accuracy: float) -> np.ndarray:
    """Exact position plus a binary mode label at the locked accuracy."""
    matrix = np.zeros((LENGTH * MODES, LENGTH * MODES))
    for position in range(LENGTH):
        for mode in range(MODES):
            column = position * MODES + mode
            matrix[position * MODES + mode, column] = accuracy
            matrix[position * MODES + (1 - mode), column] = 1 - accuracy
    return matrix


def preference() -> np.ndarray:
    utility = np.zeros(LENGTH * MODES)
    utility[(LENGTH - 1) * MODES :] = 3.0
    weights = np.exp(utility - utility.max())
    return weights / weights.sum()


def active_columns() -> np.ndarray:
    """Public [mode, action, old-position] reference-policy mask."""
    active = np.zeros((MODES, ACTIONS, LENGTH), dtype=bool)
    active[0, 2, : LENGTH - 1] = True
    active[1, 0, : LENGTH - 1] = True
    return active


ACTIVE_PUBLIC = active_columns()
ACTIVE_FLAT = ACTIVE_PUBLIC.reshape(MODES * ACTIONS, LENGTH)


def true_position_transition() -> np.ndarray:
    transition = np.zeros(
        (MODES, ACTIONS, LENGTH, LENGTH), dtype=float
    )
    moves = (-1, 0, 1)
    for mode in range(MODES):
        direction = -1 if mode % 2 else 1
        for action, move in enumerate(moves):
            for position in range(LENGTH):
                target = np.clip(
                    position + direction * move, 0, LENGTH - 1
                )
                transition[mode, action, target, position] = 1.0
    return transition


TRUE_POSITION = true_position_transition().reshape(
    MODES * ACTIONS, LENGTH, LENGTH
)


def _seed(
    master: int, observability_index: int, episode: int, stream: int
) -> int:
    sequence = np.random.SeedSequence(
        [master, observability_index, episode, stream]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def mode_schedule(master: int, observability_index: int) -> np.ndarray:
    modes = np.repeat(np.arange(MODES), N_EPISODES // MODES)
    rng = np.random.default_rng(
        _seed(master, observability_index, 0, _STREAM_MODE)
    )
    rng.shuffle(modes)
    return modes


def _builder(accuracy: float, *, annealed: bool):
    schedule = (
        (
            TransitionNoveltySpec(
                POSITION_LEARNABLE,
                half_life=N_HALF,
                active_columns=ACTIVE_PUBLIC,
                saturation_weight=SATURATION_WEIGHT,
            ),
        )
        if annealed
        else ()
    )

    def build(*, past: int, steps: int, **kwargs):
        return build_factored_corridor(
            past=past,
            steps=steps,
            length=LENGTH,
            modes=MODES,
            position_prior=POSITION_PRIOR,
            mode_prior=MODE_PRIOR,
            observation_matrix=observation_matrix(accuracy),
            observation_source="state",
            preference=preference(),
            actions=ACTION_PRIOR,
            dtype=np.float64,
            gfe=True,
            transition_novelty=schedule,
            **kwargs,
        )

    return build


def make_agent(arm: str, accuracy: float) -> RollingAgent:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    learning = arm != "known_plain"
    epistemic = arm in ("unannealed", "annealed")
    return RollingAgent(
        _builder(accuracy, annealed=arm == "annealed"),
        horizon=HORIZON,
        learn=(POSITION_LEARNABLE,) if learning else (),
        priors=(
            {POSITION_LEARNABLE: DIRICHLET_PRIOR}
            if learning
            else None
        ),
        epistemic_b=epistemic,
        damp=KNOWN_DAMP if arm == "known_plain" else 0.5,
        n_sweeps=(
            KNOWN_N_SWEEPS if arm == "known_plain" else N_SWEEPS
        ),
        select="sample",
        seed=0,
    )


def _reset_agent(agent: RollingAgent) -> None:
    for name, initial in agent.learnables.initial_params.items():
        agent.learnables.params[name] = {
            key: value.copy() for key, value in initial.items()
        }
    agent.beliefs = {
        "position": POSITION_PRIOR.copy(),
        "mode": MODE_PRIOR.copy(),
    }
    agent.tick = 0
    agent.episode_tick = 0
    agent._pending_goal_priors = None


def _start_episode(agent: RollingAgent) -> None:
    agent.reset_episode()
    agent.beliefs["position"] = POSITION_PRIOR.copy()
    agent.beliefs["mode"] = MODE_PRIOR.copy()


def _environment_step(position: int, mode: int, action: int) -> int:
    selector = mode * ACTIONS + action
    return int(np.argmax(TRUE_POSITION[selector, :, position]))


def _transition_snapshot(
    agent: RollingAgent,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if POSITION_LEARNABLE not in agent.learnables.params:
        shape = (MODES * ACTIONS, LENGTH)
        return (
            np.full(shape, np.nan),
            np.full(shape, np.nan),
            TRUE_POSITION.copy(),
        )
    alpha = agent.learnables.params[POSITION_LEARNABLE]["alpha"]
    alpha0 = agent.learnables.initial_params[POSITION_LEARNABLE]["alpha"]
    evidence = (alpha - alpha0).sum(axis=1)
    weights = agent.learnables.novelty_weights(
        POSITION_LEARNABLE, N_HALF
    )
    mean = agent.learnables.expected(POSITION_LEARNABLE)
    return evidence, weights, mean


def _active_kl(agent: RollingAgent) -> float:
    if POSITION_LEARNABLE not in agent.learnables.params:
        return 0.0
    mean = agent.learnables.expected(POSITION_LEARNABLE)
    values = []
    for mode in range(MODES):
        action = 2 if mode == 0 else 0
        selector = mode * ACTIONS + action
        for position in range(LENGTH - 1):
            values.append(-np.log(mean[selector, position + 1, position]))
    return float(np.mean(values))


def _tm_node_energies(
    agent: RollingAgent, planned: PlanningResult
) -> np.ndarray:
    if planned.planner == "plain":
        return np.full(HORIZON, np.nan)
    engine = agent.plan
    state = planned.run.state
    terms = []
    for group in engine.arena.gfe_tm_groups:
        ko, ki, ku = group.cards
        md = state.messages[
            engine.arena.message_indices(group.gather[:, 0], ko)
        ]
        mi = state.messages[
            engine.arena.message_indices(group.gather[:, 1], ki)
        ]
        me = state.messages[
            engine.arena.message_indices(group.gather[:, 2], ku)
        ]
        b_key = group.name + (":counts" if group.learned else ":B")
        terms.extend(
            np.asarray(
                rules.gfe_tm_energy(
                    mi,
                    me,
                    md,
                    state.params[b_key],
                    learned=group.learned,
                    novelty_weight=(
                        state.params[group.novelty_key]
                        if group.novelty_key is not None
                        else None
                    ),
                )
            ).tolist()
        )
    out = np.full(HORIZON, np.nan)
    out[: min(len(terms), HORIZON)] = terms[:HORIZON]
    return out


def _empty_trace() -> dict[str, np.ndarray]:
    step = (N_EPISODES, MAX_ACTIONS)
    return {
        "valid": np.zeros(step, dtype=bool),
        "policy": np.full((*step, ACTIONS), np.nan, dtype=np.float32),
        "action": np.full(step, -1, dtype=np.int8),
        "true_position": np.full(step, -1, dtype=np.int8),
        "true_mode": np.full(step, -1, dtype=np.int8),
        "observation": np.full(step, -1, dtype=np.int16),
        "position_belief": np.full(
            (*step, LENGTH), np.nan, dtype=np.float32
        ),
        "mode_belief": np.full(
            (*step, MODES), np.nan, dtype=np.float32
        ),
        "evidence": np.full(
            (*step, MODES * ACTIONS, LENGTH),
            np.nan,
            dtype=np.float32,
        ),
        "novelty_weight": np.full(
            (*step, MODES * ACTIONS, LENGTH),
            np.nan,
            dtype=np.float32,
        ),
        "transition_mean": np.full(
            (*step, MODES * ACTIONS, LENGTH, LENGTH),
            np.nan,
            dtype=np.float32,
        ),
        "active_saturated": np.zeros(step, dtype=bool),
        "planner": np.full(step, -1, dtype=np.int8),
        "plan_free_energy": np.full(step, np.nan),
        "filter_free_energy": np.full(step, np.nan),
        "tm_node_energy": np.full((*step, HORIZON), np.nan),
        "plan_iterations": np.zeros(step, dtype=np.int16),
        "filter_iterations": np.zeros(step, dtype=np.int16),
        "plan_final_delta": np.full(step, np.nan),
        "filter_final_delta": np.full(step, np.nan),
        "plan_status": np.full(step, -1, dtype=np.int8),
        "filter_status": np.full(step, -1, dtype=np.int8),
        "terminal_reason": np.full(N_EPISODES, -1, dtype=np.int8),
    }


def _run_cell(
    agent: RollingAgent,
    *,
    arm: str,
    accuracy: float,
    observability_index: int,
    master_seed: int,
    output: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    _reset_agent(agent)
    modes = mode_schedule(master_seed, observability_index)
    sensor = observation_matrix(accuracy)
    trace = _empty_trace()
    success = np.zeros(N_EPISODES, dtype=bool)
    steps = np.zeros(N_EPISODES, dtype=np.int8)
    task_return = np.zeros(N_EPISODES)
    model_kl = np.zeros(N_EPISODES)
    active_occupancy = np.zeros(N_EPISODES, dtype=np.int16)
    outside_occupancy = np.zeros(N_EPISODES, dtype=np.int16)
    switch_episode = N_EPISODES + 1

    for episode, mode in enumerate(modes):
        _start_episode(agent)
        # cff-5ipi/v1 permits drop-back only at an episode boundary. Latch
        # the choice here even if the final active column crosses its
        # threshold during this episode.
        episode_plain = (
            arm == "annealed" and agent.using_plain_planner
        )
        if episode_plain:
            switch_episode = min(switch_episode, episode + 1)
        agent.reset_action_rng(
            _seed(
                master_seed,
                observability_index,
                episode,
                _STREAM_ACTION,
            )
        )
        sensor_rng = np.random.default_rng(
            _seed(
                master_seed,
                observability_index,
                episode,
                _STREAM_SENSOR,
            )
        )
        position = 0
        for step in range(MAX_ACTIONS):
            planned = agent.plan_action(
                use_plain=episode_plain if arm == "annealed" else None
            )
            action = planned.action
            old_position = position
            position = _environment_step(position, int(mode), action)
            state_column = position * MODES + int(mode)
            observation = int(
                sensor_rng.choice(
                    LENGTH * MODES, p=sensor[:, state_column]
                )
            )
            filtered = agent.filter_observation(action, observation)
            evidence, weights, mean = _transition_snapshot(agent)

            selector = int(mode) * ACTIONS + action
            if ACTIVE_FLAT[selector, old_position]:
                active_occupancy[episode] += 1
            else:
                outside_occupancy[episode] += 1

            index = (episode, step)
            trace["valid"][index] = True
            trace["policy"][index] = planned.marginal
            trace["action"][index] = action
            trace["true_position"][index] = position
            trace["true_mode"][index] = mode
            trace["observation"][index] = observation
            trace["position_belief"][index] = agent.beliefs["position"]
            trace["mode_belief"][index] = agent.beliefs["mode"]
            trace["evidence"][index] = evidence
            trace["novelty_weight"][index] = weights
            trace["transition_mean"][index] = mean
            trace["active_saturated"][index] = agent.novelty_saturated()
            trace["planner"][index] = (
                1 if planned.planner == "plain" else 0
            )
            trace["plan_free_energy"][index] = planned.free_energy
            trace["filter_free_energy"][index] = filtered.free_energy
            trace["tm_node_energy"][index] = _tm_node_energies(
                agent, planned
            )
            trace["plan_iterations"][index] = planned.run.iterations
            trace["filter_iterations"][index] = filtered.run.iterations
            trace["plan_final_delta"][index] = planned.run.residual
            trace["filter_final_delta"][index] = filtered.run.residual
            # Every non-converged result raises before this point. Retain the
            # explicit public status code in the valid trace nevertheless.
            trace["plan_status"][index] = 0
            trace["filter_status"][index] = 0

            if not (
                np.isfinite(planned.free_energy)
                and np.isfinite(filtered.free_energy)
                and np.isfinite(planned.run.residual)
                and np.isfinite(filtered.run.residual)
                and np.all(np.isfinite(planned.marginal))
                and np.all(
                    np.isfinite(np.asarray(planned.run.state.messages))
                )
                and np.all(
                    np.isfinite(np.asarray(filtered.run.state.messages))
                )
                and (
                    planned.planner == "plain"
                    or np.all(
                        np.isfinite(trace["tm_node_energy"][index])
                    )
                )
            ):
                raise FloatingPointError(
                    f"{PROTOCOL}: non-finite diagnostic in "
                    f"{arm}/r={accuracy}/seed={master_seed}/"
                    f"episode={episode + 1}/step={step + 1}"
                )
            if position == LENGTH - 1:
                success[episode] = True
                trace["terminal_reason"][episode] = 0
                steps[episode] = step + 1
                break
        else:
            trace["terminal_reason"][episode] = 1
            steps[episode] = MAX_ACTIONS

        task_return[episode] = (
            float(success[episode]) - steps[episode] / MAX_ACTIONS
        )
        model_kl[episode] = _active_kl(agent)

    filename = (
        f"trace_r{int(round(accuracy * 100)):03d}_"
        f"{arm}_seed{master_seed:02d}.npz"
    )
    np.savez_compressed(
        output / filename,
        **trace,
        protocol=np.asarray(PROTOCOL),
        arm=np.asarray(arm),
        accuracy=np.asarray(accuracy),
        master_seed=np.asarray(master_seed),
        configured_n_sweeps=np.asarray(agent.n_sweeps),
        configured_damp=np.asarray(agent.damp),
        mode_schedule=modes,
        status_vocabulary=np.asarray(
            ["converged", "budget_exhausted", "diverged", "infeasible"]
        ),
        terminal_vocabulary=np.asarray(["success", "budget"]),
        planner_vocabulary=np.asarray(["epistemic_b", "plain"]),
    )
    metrics = {
        "success": success,
        "steps": steps,
        "task_return": task_return,
        "model_kl": model_kl,
        "active_occupancy": active_occupancy,
        "outside_occupancy": outside_occupancy,
    }
    snapshot = {
        "params": {
            name: {
                key: value.copy()
                for key, value in params.items()
            }
            for name, params in agent.learnables.params.items()
        },
        "beliefs": {
            name: value.copy() for name, value in agent.beliefs.items()
        },
        "tick": agent.tick,
        "episode_tick": agent.episode_tick,
        "switch_episode": switch_episode,
    }
    return metrics, snapshot


def _restore_snapshot(
    agent: RollingAgent, snapshot: dict[str, object]
) -> None:
    agent.learnables.params = {
        name: {
            key: np.asarray(value).copy()
            for key, value in params.items()
        }
        for name, params in snapshot["params"].items()
    }
    agent.beliefs = {
        name: np.asarray(value).copy()
        for name, value in snapshot["beliefs"].items()
    }
    agent.tick = int(snapshot["tick"])
    agent.episode_tick = int(snapshot["episode_tick"])


def _direct_plan(agent: RollingAgent, *, plain: bool) -> None:
    engine = agent.plain_plan if plain else agent.plan
    if engine is None:
        raise RuntimeError("timing requested an unavailable plain planner")
    state0 = agent._with_params(engine)
    run = engine.run(
        state0, agent.n_sweeps, tol=1e-10, damp=agent.damp
    )
    state = run.require_converged()
    policy = np.asarray(
        engine.marginals(state)[agent._plan_spec.action]
    )
    int(agent.rng.choice(agent.n_ctrl, p=policy / policy.sum()))
    float(engine.bfe(state))


def _timed(callable_) -> int:
    started = time.perf_counter_ns()
    callable_()
    return time.perf_counter_ns() - started


def _run_timing(
    agent: RollingAgent,
    snapshots: dict[int, dict[str, object]],
) -> np.ndarray:
    # [seed, planner variant, repetition], where the variants are ordinary
    # direct, annealed dispatch, and forced GFE-TM.
    timing = np.full((len(SEEDS), 3, 25), np.nan)
    for seed in SEEDS:
        snapshot = snapshots[seed]
        _restore_snapshot(agent, snapshot)
        if not agent.using_plain_planner:
            continue
        _direct_plan(agent, plain=True)
        _direct_plan(agent, plain=False)
        calls = (
            lambda: _direct_plan(agent, plain=True),
            lambda: agent.plan_action(),
            lambda: _direct_plan(agent, plain=False),
        )
        for repetition in range(25):
            for index in np.roll(np.arange(3), -(repetition % 3)):
                timing[seed, index, repetition] = _timed(calls[index])
    return timing


def _interval(samples: np.ndarray) -> list[float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return [float(lo), float(hi)]


def _bootstrap_ratio(a: np.ndarray, b: np.ndarray) -> dict[str, object]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_pairs = len(a)
    indices = rng.integers(
        0, n_pairs, size=(N_BOOTSTRAP, n_pairs)
    )
    samples = a[indices].mean(axis=1) / b[indices].mean(axis=1)
    return {
        "point": float(a.mean() / b.mean()),
        "ci95": _interval(samples),
    }


def _bootstrap_difference(
    a: np.ndarray, b: np.ndarray
) -> dict[str, object]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_pairs = len(a)
    indices = rng.integers(
        0, n_pairs, size=(N_BOOTSTRAP, n_pairs)
    )
    samples = (a[indices] - b[indices]).mean(axis=1)
    return {
        "point": float((a - b).mean()),
        "ci95": _interval(samples),
    }


def analyze(
    metrics: dict[str, np.ndarray], timing: np.ndarray
) -> dict[str, object]:
    success = metrics["success"]
    task_return = metrics["task_return"]
    model_kl = metrics["model_kl"]
    switch_episode = metrics["switch_episode"]
    arm_index = {name: index for index, name in enumerate(ARMS)}
    ann = arm_index["annealed"]
    learn = arm_index["learn_only"]
    raw = arm_index["unannealed"]

    medians = np.nanmedian(timing, axis=2)
    timing_valid = np.isfinite(medians[:, :2]).all(axis=1)
    timing_ratio = (
        _bootstrap_ratio(
            medians[timing_valid, 1] / 1e6,
            medians[timing_valid, 0] / 1e6,
        )
        if timing_valid.sum() >= 15
        else {"point": float("nan"), "ci95": [float("nan")] * 2}
    )
    timing_summary = {
        "switched_seed_count": int(timing_valid.sum()),
        "per_seed_median_ms": medians.tolist(),
        "median_ms": np.nanmedian(medians, axis=0).tolist(),
        "mad_ms": np.nanmedian(
            np.abs(medians - np.nanmedian(medians, axis=0)), axis=0
        ).tolist(),
        "annealed_plain_ratio": timing_ratio,
    }

    levels: dict[str, object] = {}
    for obs_index, accuracy in enumerate(OBSERVABILITY):
        early_kl = model_kl[obs_index, :, :, EARLY].mean(axis=2)
        late_success = success[obs_index, :, :, LATE].mean(axis=2)
        late_return = task_return[obs_index, :, :, LATE].mean(axis=2)
        kl_ratio = _bootstrap_ratio(early_kl[ann], early_kl[learn])
        success_noninferiority = _bootstrap_difference(
            late_success[ann], late_success[learn]
        )
        return_noninferiority = _bootstrap_difference(
            late_return[ann], late_return[learn]
        )
        unannealed_gain = _bootstrap_difference(
            late_success[ann], late_success[raw]
        )
        switched = int(
            np.sum(switch_episode[obs_index, ann] <= 48)
        )
        gates = {
            "early_kl": kl_ratio["ci95"][1] <= 0.90,
            "late_success_noninferior": (
                success_noninferiority["ci95"][0] >= -0.05
            ),
            "late_return_noninferior": (
                return_noninferiority["ci95"][0] >= -0.05
            ),
            "beats_unannealed": (
                unannealed_gain["point"] >= 0.10
                and unannealed_gain["ci95"][0] > 0
            ),
            "switch_coverage": switched >= 15,
            "plain_cost": timing_ratio["ci95"][1] <= 1.15,
        }
        episode_means = {
            arm: {
                "success": success[
                    obs_index, index
                ].mean(axis=0).tolist(),
                "task_return": task_return[
                    obs_index, index
                ].mean(axis=0).tolist(),
                "model_kl": model_kl[
                    obs_index, index
                ].mean(axis=0).tolist(),
            }
            for index, arm in enumerate(ARMS)
        }
        levels[str(accuracy)] = {
            "early_kl_ratio": kl_ratio,
            "late_success_minus_learn_only": success_noninferiority,
            "late_return_minus_learn_only": return_noninferiority,
            "late_success_minus_unannealed": unannealed_gain,
            "switch_by_episode_48": switched,
            "gates": gates,
            "arm_means": {
                arm: {
                    "early_kl": float(early_kl[index].mean()),
                    "late_success": float(late_success[index].mean()),
                    "late_return": float(late_return[index].mean()),
                    "late_kl": float(
                        model_kl[
                            obs_index, index, :, LATE
                        ].mean()
                    ),
                    "mean_switch_episode": float(
                        switch_episode[obs_index, index].mean()
                    ),
                    "active_occupancy": int(
                        metrics["active_occupancy"][
                            obs_index, index
                        ].sum()
                    ),
                    "outside_occupancy": int(
                        metrics["outside_occupancy"][
                            obs_index, index
                        ].sum()
                    ),
                    "successful_steps": float(
                        metrics["steps"][
                            obs_index, index, :, LATE
                        ][success[obs_index, index, :, LATE]].mean()
                    )
                    if np.any(success[obs_index, index, :, LATE])
                    else float("nan"),
                }
                for index, arm in enumerate(ARMS)
            },
            "episode_means": episode_means,
        }

    primary = levels["1.0"]["gates"]
    if all(primary.values()):
        secondary_passes = all(
            all(levels[str(accuracy)]["gates"].values())
            for accuracy in OBSERVABILITY[1:]
        )
        outcome = (
            "useful annealed transition exploration"
            if secondary_passes
            else "bound-sensitive useful exploration"
        )
    else:
        task_cost_names = (
            "late_success_noninferior",
            "late_return_noninferior",
            "switch_coverage",
            "plain_cost",
        )
        task_cost = all(
            primary[key]
            for key in task_cost_names
        )
        level = levels["1.0"]
        gain = levels["1.0"]["late_success_minus_unannealed"]
        point_gates = {
            "early_kl": level["early_kl_ratio"]["point"] <= 0.90,
            "late_success_noninferior": (
                level["late_success_minus_learn_only"]["point"] >= -0.05
            ),
            "late_return_noninferior": (
                level["late_return_minus_learn_only"]["point"] >= -0.05
            ),
            "beats_unannealed": gain["point"] >= 0.10,
            "switch_coverage": (
                level["switch_by_episode_48"] >= 15
            ),
            "plain_cost": (
                timing_ratio["point"] <= 1.15
            ),
        }
        task_cost_points = all(
            point_gates[key] for key in task_cost_names
        )
        raw_equivalent = (
            gain["point"] < 0.10 and gain["ci95"][0] >= -0.05
        )
        if not task_cost_points:
            outcome = "annealing does not recover the required behavior"
        elif task_cost and raw_equivalent:
            outcome = "no observed task need for annealing"
        elif (
            task_cost
            and primary["beats_unannealed"]
            and not point_gates["early_kl"]
        ):
            outcome = (
                "annealing recovers goal performance/cost without "
                "evidence of useful exploration"
            )
        else:
            # At this point the recovery thresholds pass in point estimate,
            # but at least one locked interval gate does not. The
            # preregistration names that result ambiguous rather than pass.
            outcome = "ambiguous at the preregistered budget"

    return {
        "protocol": PROTOCOL,
        "outcome": outcome,
        "levels": levels,
        "timing": timing_summary,
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib_xla": jaxlib.__version__,
            "numpy": np.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "device_kinds": [
                device.device_kind for device in jax.devices()
            ],
            "platform_versions": sorted(
                {
                    str(device.client.platform_version)
                    for device in jax.devices()
                }
            ),
        },
    }


def run_experiment(output: Path) -> dict[str, object]:
    # v1 requested float64 arrays but let JAX silently truncate them. v3
    # explicitly installs and verifies the preregistered dtype before any
    # engine is constructed or compiled.
    jax.config.update("jax_enable_x64", True)
    if not jax.config.x64_enabled:  # pragma: no cover - JAX owns this flag
        raise RuntimeError(f"{PROTOCOL}: JAX x64 mode could not be enabled")

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"{output} is not empty; {PROTOCOL} never resumes or overwrites "
            "a partial run"
        )
    output.mkdir(parents=True, exist_ok=True)

    shape = (
        len(OBSERVABILITY),
        len(ARMS),
        len(SEEDS),
        N_EPISODES,
    )
    metrics = {
        "success": np.zeros(shape, dtype=bool),
        "steps": np.zeros(shape, dtype=np.int8),
        "task_return": np.zeros(shape),
        "model_kl": np.zeros(shape),
        "active_occupancy": np.zeros(shape, dtype=np.int16),
        "outside_occupancy": np.zeros(shape, dtype=np.int16),
        "switch_episode": np.full(
            shape[:-1], N_EPISODES + 1, dtype=np.int16
        ),
    }
    timing_agent: RollingAgent | None = None
    timing_snapshots: dict[int, dict[str, object]] = {}

    for obs_index, accuracy in enumerate(OBSERVABILITY):
        for arm_index, arm in enumerate(ARMS):
            agent = make_agent(arm, accuracy)
            if np.dtype(agent.plan.init_state().messages.dtype) != np.float64:
                raise RuntimeError(
                    f"{PROTOCOL}: {arm}/r={accuracy} compiled without float64"
                )
            for seed in SEEDS:
                cell, snapshot = _run_cell(
                    agent,
                    arm=arm,
                    accuracy=accuracy,
                    observability_index=obs_index,
                    master_seed=seed,
                    output=output,
                )
                for name in (
                    "success",
                    "steps",
                    "task_return",
                    "model_kl",
                    "active_occupancy",
                    "outside_occupancy",
                ):
                    metrics[name][obs_index, arm_index, seed] = cell[name]
                metrics["switch_episode"][
                    obs_index, arm_index, seed
                ] = snapshot["switch_episode"]
                if accuracy == 1.0 and arm == "annealed":
                    timing_agent = agent
                    timing_snapshots[seed] = snapshot
                print(
                    f"{PROTOCOL} r={accuracy:.2f} arm={arm} "
                    f"seed={seed:02d} complete",
                    flush=True,
                )

    if timing_agent is None or set(timing_snapshots) != set(SEEDS):
        raise RuntimeError("missing exact-sensor annealed timing checkpoints")
    timing = _run_timing(timing_agent, timing_snapshots)
    np.savez_compressed(output / "metrics.npz", **metrics, timing_ns=timing)
    summary = analyze(metrics, timing)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Run locked {PROTOCOL} experiment"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new or empty directory for raw traces and summary",
    )
    args = parser.parse_args()
    summary = run_experiment(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()

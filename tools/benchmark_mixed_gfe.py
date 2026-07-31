"""Cold/warm mixed-GFE benchmark for the cff-lv2w acceptance gates.

Run one case per fresh process so ``cold_seconds`` includes the first JAX
compile. The first repeated measurement is discarded and the remaining
samples are reported individually plus by median.
"""

from __future__ import annotations

import argparse
import functools
import json
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cffg.agent import RollingAgent
from cffg.compiler import compile_template
from cffg.engine import Engine, RunResult
from cffg.models import build_factored_corridor
from cffg.models.gridworld import build_gridworld
from cffg.models.rollout import build_rollout
from cffg.models.tmaze import build_tmaze


@dataclass(frozen=True)
class Counts:
    outer_maps: int
    trials: int
    linear_iterations: int
    solver_compilations: int

    @classmethod
    def from_results(cls, results: list[RunResult]) -> Counts:
        return cls(
            sum(getattr(result, "outer_map_evaluations", 0) for result in results),
            sum(getattr(result, "trial_evaluations", 0) for result in results),
            sum(getattr(result, "linear_iterations", 0) for result in results),
            sum(getattr(result, "solver_compilations", 0) for result in results),
        )


Runner = Callable[[], list[RunResult]]


def _sync(results: list[RunResult]) -> None:
    for result in results:
        jax.block_until_ready(result.state.messages)


def _tmaze(*, gfe: bool = True, picard: bool = False) -> Runner:
    template, arrays = build_tmaze(gfe=gfe)
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.init_state()

    def run() -> list[RunResult]:
        options = (
            {"damp": 0.5, "outer_solver": "picard"} if picard else {}
        )
        return [engine.run(state, 400, tol=1e-13, **options)]

    return run


def _d1(*, picard: bool = False) -> Runner:
    states, actions = 4, 2
    sensor = np.full((states, states), 0.1 / (states - 1))
    np.fill_diagonal(sensor, 0.9)
    counts = np.stack(
        [
            np.full((states, states), 60.0 / states),
            np.ones((states, states)),
        ]
    )
    template, arrays = build_rollout(
        "benchmark-d1",
        steps=2,
        d=np.full(states, 1.0 / states),
        A=sensor,
        c=np.full(states, 1.0 / states),
        e=np.full(actions, 1.0 / actions),
        Bs=np.full((actions, states, states), 1.0 / states),
        gfe=False,
        epistemic_b=True,
    )
    arrays["Bc"] = counts
    engine = Engine(compile_template(template, arrays=arrays))
    state = engine.init_state()

    def run() -> list[RunResult]:
        options = (
            {"damp": 0.5, "outer_solver": "picard"} if picard else {}
        )
        return [engine.run(state, 80, tol=1e-12, **options)]

    return run


def _rolling(*, picard: bool = False) -> Runner:
    agent = RollingAgent(
        functools.partial(build_gridworld, k=6),
        horizon=3,
        n_sweeps=400,
        seed=0,
    )
    beliefs = agent.beliefs["state"].copy()

    def run() -> list[RunResult]:
        results = []
        for tick in range(10):
            agent.beliefs["state"] = np.roll(beliefs, tick % len(beliefs))
            state = agent._with_params(agent.plan)
            options = (
                {"damp": 0.5, "outer_solver": "picard"}
                if picard
                else {}
            )
            results.append(agent.plan.run(state, 400, tol=1e-10, **options))
        return results

    return run


def _horizon8() -> Runner:
    length, modes, actions, horizon = 8, 2, 3, 8
    position_prior = np.r_[1.0, np.zeros(length - 1)]
    mode_prior = np.full(modes, 1.0 / modes)
    action_prior = np.full(actions, 1.0 / actions)
    transition_prior = np.full(
        (modes * actions, length, length), 1.0 / length
    )
    sensor = np.eye(length * modes)
    utility = np.zeros(length * modes)
    utility[(length - 1) * modes :] = 3.0
    preference = np.exp(utility - utility.max())
    preference /= preference.sum()

    def builder(*, past, steps, **kwargs):
        return build_factored_corridor(
            past=past,
            steps=steps,
            length=length,
            modes=modes,
            position_prior=position_prior,
            mode_prior=mode_prior,
            observation_matrix=sensor,
            observation_source="state",
            preference=preference,
            actions=action_prior,
            dtype=np.float64,
            gfe=True,
            **kwargs,
        )

    agent = RollingAgent(
        builder,
        horizon,
        learn=("position_transition",),
        priors={"position_transition": transition_prior},
        damp=0.5,
        n_sweeps=400,
        select="sample",
        seed=0,
    )
    alpha = transition_prior.copy()
    alpha[5, 0, 0] += 1.0
    agent.learnables.params["position_transition"]["alpha"] = alpha
    agent.beliefs = {
        "position": position_prior.copy(),
        "mode": np.array([0.0, 1.0]),
    }
    engine = agent.plan
    state = agent._with_params(engine)

    def run() -> list[RunResult]:
        return [engine.run(state, 400, tol=1e-10)]

    return run


CASES: dict[str, Callable[[], Runner]] = {
    "tmaze": _tmaze,
    "tmaze_picard": functools.partial(_tmaze, picard=True),
    "d1": _d1,
    "d1_picard": functools.partial(_d1, picard=True),
    "rolling10": _rolling,
    "rolling10_picard": functools.partial(_rolling, picard=True),
    "horizon8": _horizon8,
    "bfe": functools.partial(_tmaze, gfe=False),
}


def _measure(run: Runner) -> tuple[float, Counts, list[RunResult]]:
    start = time.perf_counter()
    results = run()
    _sync(results)
    return time.perf_counter() - start, Counts.from_results(results), results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions < 5:
        parser.error("--repetitions must be at least 5")

    run = CASES[args.case]()
    cold_seconds, cold_counts, cold_results = _measure(run)
    discarded_seconds, _, discarded_results = _measure(run)
    warm = [_measure(run) for _ in range(args.repetitions)]
    all_results = [*cold_results, *discarded_results]
    for _, _, results in warm:
        all_results.extend(results)
    if any(result.status.value != "converged" for result in all_results):
        statuses = [result.status.value for result in all_results]
        raise RuntimeError(f"benchmark inference did not converge: {statuses}")

    warm_seconds = [seconds for seconds, _, _ in warm]
    print(
        json.dumps(
            {
                "case": args.case,
                "backend": jax.default_backend(),
                "device": str(jax.devices()[0]),
                "jax": jax.__version__,
                "python": platform.python_version(),
                "x64": bool(jax.config.jax_enable_x64),
                "cold_seconds": cold_seconds,
                "cold_counts": asdict(cold_counts),
                "discarded_warmup_seconds": discarded_seconds,
                "warm_seconds": warm_seconds,
                "warm_median_seconds": statistics.median(warm_seconds),
                "warm_counts": [asdict(counts) for _, counts, _ in warm],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Pair production indexed and dense Colony runs without changing the Hallway inputs."""

import argparse
from dataclasses import dataclass, fields
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from unittest.mock import patch

import jax
import numpy as np

import cffg
from benchmark_ordinary_iteration import (
    DAMPING,
    MEMBERS,
    SWEEPS,
    hallway,
    hardware,
    inputs_fingerprint,
    measured,
    profile,
    results_record,
)
from cffg.colony import Colony
from cffg.compiled import NEG
from cffg.compiled_iteration import CompiledIteration, controller_for
from cffg.compiler import compile_template
from cffg.core import BlobRef, Broadcast, FactorSpec, PlateSpec, PortRole, PortSpec, Template, VariableSpec
from cffg.core.serialize import structure_hash
from cffg.engine import Engine
from cffg.iteration_tolerance import tolerance
from cffg.models.rollout import build_rollout
from cffg.runtime_types import _replace


@dataclass(frozen=True)
class Case:
    template: Template
    arrays: dict
    members: tuple[dict, ...]
    sweeps: int


def cases():
    template, arrays, priors = hallway()

    def members(values):
        return tuple({**values, "d": row} for row in priors)

    yield "hallway", Case(template, arrays, members(arrays), SWEEPS)
    card = len(priors[0])
    diffuse = np.mean([np.roll(np.eye(card), shift, axis=0) for shift in range(card // 4)], axis=0)
    for name, background in (("dense", 1 / card), ("quarter_support", diffuse)):
        values = {**arrays, "Bs": .5 * arrays["Bs"] + .5 * background}
        yield name, Case(template, values, members(values), 12)

    prior = np.array([.2, .3, .5])
    small, small_arrays = build_rollout(
        "small-colony-control", steps=2, d=prior, A=np.full((2, 3), .5),
        c=np.array([.4, .6]), e=np.full(2, .5),
        Bs=np.stack((np.eye(3), np.roll(np.eye(3), 1, axis=0))), gfe=False,
    )
    yield "small", Case(small, small_arrays, tuple(
        {**small_arrays, "d": np.roll(prior, i % 3)} for i in range(MEMBERS)
    ), SWEEPS)

    fragment = Template(
        "member", variables={"s": VariableSpec("s", 3)},
        factors={"prior": FactorSpec("prior", "categorical", {"out": "s"},
                                     param_blobs={"statistic": "w"})},
        ports={"s": PortSpec("s", "s", PortRole.GLUE)}, blobs={"w": BlobRef("w", "w.npy")},
    )
    count = 64
    balanced = Template("balanced-colony-control", plates={
        "p": PlateSpec(fragment, (count,), (Broadcast("s", "hub"),), per_copy_blobs=("w",)),
    })
    values = tuple({
        f"p/{leaf}/w": np.roll(prior, (leaf + lane) % 3) for leaf in range(count)
    } for lane in range(MEMBERS))
    yield "balanced_broadcast", Case(balanced, values[0], values, SWEEPS)


class Arm:
    def __init__(self, case, name):
        self.name = name
        self.colony = None
        self.outputs = {}
        self.timings = {}
        if name.startswith("colony_"):
            self.setup_s, self.colony = measured(lambda: self._spawn(case))
            self.engine, = self.colony.engines.values()
            self.states = tuple(row.state for row in self.colony.instances.values())
            if name == "colony_dense":
                stock = controller_for(self.engine)
                self.engine._compiled_iteration = CompiledIteration(
                    self.engine, stock.hooks, allow_indexed=False,
                )
        else:
            self.setup_s, (self.engine, self.states) = measured(lambda: self._initialize(case))

    @staticmethod
    def _spawn(case):
        colony = Colony()
        for arrays in case.members:
            colony.spawn(case.template, arrays)
        return colony

    def _initialize(self, case):
        engine = Engine(compile_template(case.template, arrays=case.arrays))
        states = [engine.init_state()]
        if self.name == "sequential":
            for arrays in case.members[1:]:
                local = compile_template(case.template, arrays=arrays)
                state = _replace(states[0], params={
                    key: jax.numpy.asarray(value) for key, value in local.log_params.items()
                })
                states.append(state)
        return engine, jax.block_until_ready(tuple(states))

    def reset(self):
        if self.colony is not None:
            for row, state in zip(self.colony.instances.values(), self.states, strict=True):
                row.state = state

    def run(self, budget):
        if self.colony is not None:
            results = tuple(self.colony.run(budget, damp=DAMPING, tol=None).values())
            jax.block_until_ready(tuple(result.state for result in results))
            return results
        results = []
        for state in self.states:
            result = self.engine.run(state, budget, tol=None, damp=DAMPING, harden_last=budget)
            jax.block_until_ready(result.state)
            results.append(result)
        return tuple(results)

    def dispatch(self, budget):
        if self.colony is None:
            return None
        controller = controller_for(self.engine)
        self.reset()
        with patch.object(controller, "_execute", wraps=controller._execute) as stock:
            results_record(self.run(budget), budget)
        return "dense" if stock.call_count else "indexed"

    def scratch(self, budget):
        controller = controller_for(self.engine)
        if self.name == "colony_indexed":
            runner = controller._indexed_batch(self.states)
            if runner is not None:
                controller = runner
        inputs = jax.tree.map(lambda *xs: jax.numpy.stack(xs), *self.states)
        lowered = controller.batch.lower(
            inputs, tolerance(inputs.messages.dtype, None), DAMPING, n_iters=budget,
            harden_last=0, defer_hard=False,
        )
        return lowered.compile().memory_analysis().temp_size_in_bytes


def compare(reference, candidate, decoder):
    worst = {}
    for old, new in zip(reference, candidate, strict=True):
        for field in fields(old):
            if field.name in ("state", "free_energies", "message_deltas", "residual"):
                continue
            expected, actual = getattr(old, field.name), getattr(new, field.name)
            if isinstance(expected, float) and np.isnan(expected) and np.isnan(actual):
                continue
            if expected != actual:
                raise ValueError(f"public result field differs: {field.name}")
        for field in ("free_energies", "message_deltas", "residual", "messages"):
            a, b = (getattr(value.state if field == "messages" else value, field)
                    for value in (old, new))
            a, b = np.asarray(a), np.asarray(b)
            if field == "messages":
                np.testing.assert_array_equal(a == NEG, b == NEG)
            finite = np.isfinite(a)
            np.testing.assert_array_equal(b[~finite], a[~finite], err_msg=field)
            if not np.isfinite(b[finite]).all():
                raise ValueError(f"{field} became nonfinite")
            a, b = a[finite], b[finite]
            error = float(np.max(np.abs(a - b) / (1 + np.abs(a)), initial=0.0))
            if error > 1e-10:
                raise ValueError(f"{field} scaled difference {error:.4g}")
            worst[field] = max(worst.get(field, 0), error)
        old_marginals, new_marginals = (decoder.marginals(run.state) for run in (old, new))
        assert old_marginals.keys() == new_marginals.keys()
        for edge, expected in old_marginals.items():
            error = float(np.max(np.abs(expected - new_marginals[edge])))
            if not np.isfinite(error) or error > 1e-10:
                raise ValueError(f"{edge} posterior difference {error:.4g}")
            worst["posterior"] = max(worst.get("posterior", 0), error)
    return worst


def timing(name, case, arms, repetitions):
    budgets = (1, case.sweeps) if name == "hallway" else (case.sweeps,)
    for budget in budgets:
        for arm in arms:
            arm.reset()
            cold, result = measured(lambda: arm.run(budget))
            results_record(result, budget)
            arm.timings[str(budget)] = {"cold_s": cold, "samples_s": []}
            arm.reset()
            arm.run(budget)
        for rep in range(repetitions):
            for arm in arms if rep % 2 == 0 else arms[::-1]:
                arm.reset()
                gc.collect()
                elapsed, result = measured(lambda: arm.run(budget))
                results_record(result, budget)
                arm.timings[str(budget)]["samples_s"].append(elapsed)
                arm.outputs[budget] = result
    for arm in arms:
        for row in arm.timings.values():
            row["median_s"] = statistics.median(row["samples_s"])
    return budgets


def trace_hallway(arm, budgets):
    close_trace = jax.jit(lambda value: value + 1)
    value = jax.numpy.asarray(0)
    close_trace(value).block_until_ready()
    traces = {}
    for budget in budgets:
        arm.reset()
        elapsed, results, counts = profile(
            jax, lambda: arm.run(budget), lambda: close_trace(value).block_until_ready(),
            require_complete=False,
        )
        results_record(results, budget)
        traces[str(budget)] = {**counts, "profiled_wall_s": elapsed}
    return traces


def run_case(name, case, repetitions, traces):
    names = ("single", "sequential") if name == "hallway" else ()
    arms = [Arm(case, mode) for mode in (*names, "colony_dense", "colony_indexed")]
    budgets = timing(name, case, arms, repetitions)
    stock, indexed = arms[-2:]
    parity = {}
    for budget in budgets:
        reference = arms[1] if name == "hallway" else stock
        parity[str(budget)] = {
            arm.name: compare(reference.outputs[budget][:len(arm.states)], arm.outputs[budget], stock.engine)
            for arm in arms if arm is not reference
        }
    results = {arm.name: {
        "setup_s": arm.setup_s, "timings": arm.timings,
        "dispatch": {str(budget): arm.dispatch(budget) for budget in budgets},
    } for arm in arms}
    if name == "hallway" and traces:
        for arm in (stock, indexed):
            results[arm.name]["submissions"] = trace_hallway(arm, budgets)
            results[arm.name]["scratch_bytes"] = arm.scratch(case.sweeps)
    budget = str(case.sweeps)
    medians = {arm.name: arm.timings[budget]["median_s"] for arm in arms}
    print(name, medians, results[indexed.name]["dispatch"], flush=True)
    return {"arms": results, "parity": parity, "members": len(case.members), "sweeps": case.sweeps,
            "structure_sha256": structure_hash(case.template),
            "inputs_sha256": inputs_fingerprint(case.arrays, np.stack([row["d"] for row in case.members]))
            if name == "hallway" else None,
            "dense_over_indexed": medians[stock.name] / medians[indexed.name],
            "sequential_over_indexed": medians["sequential"] / medians[indexed.name]
            if name == "hallway" else None}


def pristine_single(checkout, repetitions):
    script = """
import json
from pathlib import Path
from statistics import median
import subprocess
import sys

import cffg
from benchmark_ordinary_iteration import SWEEPS, make_runner, measured, results_record

assert Path(cffg.__file__).resolve().parent == Path.cwd() / "src/cffg"
reset, run, setup, fixture = make_runner("single")
cold, results = measured(lambda: run(SWEEPS))
results_record(results, SWEEPS)
run(SWEEPS)
samples = []
for _ in range(int(sys.argv[1])):
    reset()
    seconds, results = measured(lambda: run(SWEEPS))
    results_record(results, SWEEPS)
    samples.append(seconds)
print(json.dumps({"source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "cold_s": cold, "samples_s": samples, "median_s": median(samples),
                  "setup": setup, "fixture": fixture}))
"""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, (checkout / "src", checkout / "tools")))}
    return json.loads(subprocess.check_output(
        [sys.executable, "-c", script, str(repetitions)], cwd=checkout, env=env, text=True,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--baseline", type=Path, required=True, help="unchanged checkout for the singleton control")
    parser.add_argument("--without-profiler", action="store_true", help="correctness and wall-clock smoke")
    args = parser.parse_args()
    if not jax.config.x64_enabled or jax.default_backend() != "cpu" or args.repetitions < 1:
        parser.error("use CPU x64 and a positive repetition count")
    root = Path(__file__).resolve().parents[1]
    if Path(cffg.__file__).resolve().parent != root / "src/cffg":
        raise RuntimeError("benchmark imported a different checkout")
    baseline = args.baseline.resolve()
    for checkout in (root, baseline):
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout):
            raise RuntimeError(f"dirty timing checkout: {checkout}")
    for path in ("uv.lock", "tools/benchmark_ordinary_iteration.py"):
        if (baseline / path).read_bytes() != (root / path).read_bytes():
            raise RuntimeError(f"baseline {path} differs")
    report = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frozen_driver_sha256": hashlib.sha256((root / "tools/benchmark_ordinary_iteration.py").read_bytes()).hexdigest(),
        "lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        "hardware": hardware(), "jax": jax.__version__, "damp": DAMPING, "tol": None,
        "pristine_single": pristine_single(baseline, args.repetitions),
        "cases": {name: run_case(name, case, args.repetitions, not args.without_profiler)
                  for name, case in cases()},
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    rows = report["cases"]
    if report["pristine_single"]["fixture"]["inputs_sha256"] != rows["hallway"]["inputs_sha256"]:
        raise SystemExit("singleton reference input differs from the frozen batch")
    observed = rows["hallway"]["arms"]["colony_indexed"]["dispatch"]
    if rows["hallway"]["sequential_over_indexed"] < 8 or set(observed.values()) != {"indexed"}:
        raise SystemExit("Hallway target or indexed dispatch missed")
    for name, row in rows.items():
        if name != "hallway" and set(row["arms"]["colony_indexed"]["dispatch"].values()) != {"dense"}:
            raise SystemExit(f"negative control selected the indexed path: {name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare public ordinary Engine/Colony runs on a fixed Hallway workload.

Run with uv run --frozen --no-sync python tools/benchmark_ordinary_iteration.py
--output FILE [--checkout PATH]. Use --baseline BASELINE.json on a later revision
to compare with the frozen host loop. The measured checkout supplies cffg via
PYTHONPATH. Each arm starts in a fresh CPU process with persistent JAX caches
disabled. The checkout stays clean during the run.

July's Hallway experiment was a GFE graph and compared engine.run with a
handwritten vmapped sweep. This ordinary adaptation retains its dimensions,
matrices, seeded prior draws, and damping. It fixes the budget to 60 sweeps
and uses two status-bearing public entry points.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter


MEMBERS = 32
SWEEPS = 60
DAMPING = 0.5
SEED = 0
ARMS = ("single", "sequential", "colony")
CPU_DISPATCH = "CommonPjRtLoadedExecutable::ExecuteHelperOnSingleDevice"
CPU_THUNK = "ThunkExecutor::Execute (wait for completion)"
TRACE_BODY = "ordinary_inference"
TRACE_END = "ordinary_trace_end"
HISTORICAL = {
    "source": "cffg-experiments/long-horizon-pomdp/results.json:phase_c",
    "source_sha256": "21e7424c0d7261a96af942890cba063debf322e5e97fa11fb7f358071efa962f",
    "sequential_seconds": 33.255144709255546,
    "batched_seconds": 21.53129008412361,
    "speedup": 1.5445031198468073,
    "protocol": "Hallway L15 H10, 32 priors, GFE, 120 sweeps, damp=.5, handwritten filter_vmap",
}


def hallway():
    """Build the ordinary counterpart of the archived L=15, H=10 Hallway."""
    import numpy as np
    from cffg.models.rollout import build_rollout

    cells, horizon = 15, 10
    states, observations, actions = 4 * cells, 17, 5
    transitions = np.zeros((actions, states, states))
    emission = np.zeros((observations, states))
    for source in range(states):
        cell, direction = divmod(source, 4)
        goal = cell == cells - 1
        forward = cell + (1 if direction == 1 else -1 if direction == 3 else 0)
        forward = min(max(forward, 0), cells - 1)
        intended = (source, 4 * forward + direction, 4 * cell + (direction - 1) % 4,
                    4 * cell + (direction + 1) % 4, 4 * cell + (direction + 2) % 4)
        for action, target in enumerate(intended):
            transitions[action, source if goal else target, source] += .8
            transitions[action, source, source] += 1 - .8
        walls = (True, goal, True, cell == 0)
        for reading in range(16):
            likelihood = 1.
            for bit in range(4):
                correct = bool(reading & (1 << bit)) == walls[(direction + bit) % 4]
                likelihood *= .95 if correct else .05
            emission[reading, source] = likelihood * ((1 - .95) if goal else 1.)
        emission[-1, source] = .95 if goal else 0.
    preference = np.exp(np.r_[np.full(16, -4.), 0.])
    preference /= preference.sum()
    rng = np.random.default_rng(SEED)
    priors = rng.random((MEMBERS, states)) + 1e-3
    priors[:, -4:] = 0.
    priors /= priors.sum(axis=1, keepdims=True)
    template, arrays = build_rollout("hallwayL15[0|10]", steps=horizon, d=priors[0],
                                     A=emission, c=preference, e=np.full(actions, 1./actions),
                                     Bs=transitions, gfe=False)
    return template, arrays, priors


def seconds(values):
    return {"samples": values, "median": statistics.median(values),
            "min": min(values), "max": max(values)}


class Compiles(logging.Handler):
    """Capture JAX-reported host lowering and XLA compile durations by name."""
    pattern = re.compile(r"Finished (XLA compilation|jaxpr to MLIR module conversion) of (.+) in ([0-9.eE+-]+) sec")

    def __init__(self):
        super().__init__()
        self.events = []

    def emit(self, record):
        match = self.pattern.search(record.getMessage())
        if match:
            phase, target, duration = match.groups()
            self.events.append({"phase": phase, "target": target, "seconds": float(duration)})


@contextmanager
def compile_log(jax):
    logger = logging.getLogger("jax")
    handler = Compiles()
    logger.addHandler(handler)
    try:
        with jax.log_compiles(True):
            yield handler.events
    finally:
        logger.removeHandler(handler)


def trace_metrics(directory, *, require_complete):
    paths = list(Path(directory).rglob("perfetto_trace.json.gz"))
    if len(paths) != 1:
        raise RuntimeError("expected one JAX CPU trace")
    with gzip.open(paths[0], "rt") as stream:
        trace = json.load(stream)
    spans = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    bodies = [span for span in spans if span["name"] == TRACE_BODY]
    ends = [span for span in spans if span["name"] == TRACE_END]

    def during(frame, name):
        return [span for span in spans if span["name"] == name
                and frame["ts"] <= span["ts"] < frame["ts"] + frame["dur"]]

    complete = len(bodies) == len(ends) == 1 and len(during(ends[0], CPU_DISPATCH)) == 1
    executions = during(bodies[0], CPU_DISPATCH) if complete else []
    if not executions:
        if require_complete:
            raise RuntimeError("CPU executable trace is incomplete or incompatible with this runtime")
        return {"complete": False}
    thunks = during(bodies[0], CPU_THUNK)
    return {"complete": True, "cpu_executable_submissions": len(executions),
            "execute_helper_span_sum_s": sum(e["dur"] for e in executions) / 1e6,
            "synchronous_cpu_thunk_spans": len(thunks),
            "synchronous_cpu_thunk_span_sum_s": sum(e["dur"] for e in thunks) / 1e6}


def measured(call):
    start = perf_counter()
    result = call()
    return perf_counter() - start, result


def inputs_fingerprint(arrays, priors):
    h = hashlib.sha256()
    for name, array in sorted({**arrays, "member_priors": priors}.items()):
        h.update(str((name, array.dtype.str, array.shape)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def make_runner(arm):
    import jax
    from cffg.colony import Colony
    from cffg.compiler import compile_template
    from cffg.core.serialize import structure_hash
    from cffg.engine import Engine, _replace

    template, arrays, priors = hallway()
    member_arrays = [{**arrays, "d": row} for row in priors]
    setup = {}
    if arm == "colony":
        colony = Colony()

        def spawn():
            ids = [colony.spawn(template, row) for row in member_arrays]
            states = tuple(colony.instances[i].state for i in ids)
            jax.block_until_ready(states)
            return ids, states

        setup["spawn_including_template_compile_s"], (ids, states) = measured(spawn)
        if len(colony.engines) != 1:
            raise RuntimeError("Colony members did not form one structure group")
        engine, = colony.engines.values()

        def reset():
            for iid, initial in zip(ids, states, strict=True):
                colony.instances[iid].state = initial

        def run(budget):
            results = colony.run(budget, damp=DAMPING, tol=None)
            jax.block_until_ready(tuple(result.state for result in results.values()))
            return tuple(results[i] for i in ids)
    else:
        setup["template_compile_s"], arena = measured(lambda: compile_template(template, arrays=arrays))
        setup["engine_construction_s"], engine = measured(lambda: Engine(arena))

        def initialize():
            # Apply the same public parameter payload and initial messages as
            # Colony.spawn, without allocating 32 execution engines.
            states = [engine.init_state()]
            for params in member_arrays[1:] if arm == "sequential" else ():
                local = compile_template(template, arrays=params)
                states.append(_replace(states[0], params={k: jax.numpy.asarray(v) for k, v in local.log_params.items()}))
            return jax.block_until_ready(tuple(states))

        setup["prepare_initial_states_s"], states = measured(initialize)

        def reset():
            pass

        def run(budget):
            results = []
            for state in states:
                result = engine.run(state, budget, tol=None, damp=DAMPING, harden_last=budget)
                jax.block_until_ready(result.state)
                results.append(result)
            return tuple(results)

    arena = engine.arena
    if (arena.gfe_groups or arena.gfe_tm_groups or arena.mu3_groups
            or engine.has_adaptive_gaussian or len(arena.delta_pairs)):
        raise RuntimeError("fixture no longer selects ordinary iterative execution")
    metadata = {"slots": int(arena.n_slots), "schedule": engine.schedule.kind,
                "structure_sha256": structure_hash(template), "inputs_sha256": inputs_fingerprint(arrays, priors),
                "dtype": str(states[0].messages.dtype), "members": len(states)}
    return reset, run, setup, metadata


def results_record(results, budget):
    import numpy as np
    members = []
    states = []
    for result in results:
        if (result.status.value != "budget_exhausted" or result.iterations != budget
                or result.failure is not None and result.failure.reason != "iteration_budget"):
            raise RuntimeError(f"unexpected {budget}-sweep result: {result.status}, {result.iterations}, {result.failure}")
        members.append({"status": result.status.value, "iterations": result.iterations,
                        "trace_lengths": [len(result.message_deltas), len(result.free_energies)],
                        "diagnostic": None if result.failure is None else asdict(result.failure),
                        "residual": float(result.residual), "final_free_energy": result.free_energies[-1]})
        states.append(np.asarray(result.state.messages))
    return members, np.stack(states)


def profile(jax, call, probe, *, require_complete=True):
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 1
    options.enable_hlo_proto = False
    with tempfile.TemporaryDirectory(prefix="cffg-ordinary-trace-") as directory:
        with jax.profiler.trace(directory, create_perfetto_trace=True, profiler_options=options):
            with jax.profiler.TraceAnnotation(TRACE_BODY):
                elapsed, results = measured(call)
            # If the CPU executable log stops recording before the end, the
            # known executable cannot be recovered inside the closing span.
            with jax.profiler.TraceAnnotation(TRACE_END):
                probe()
        return elapsed, results, trace_metrics(directory, require_complete=require_complete)


def worker(arm, output, warmup, repetitions, trace_budget):
    import cffg
    import jax
    import jaxlib
    import numpy as np

    expected = Path.cwd() / "src/cffg"
    if Path(cffg.__file__).resolve().parent != expected.resolve():
        raise RuntimeError(f"wrong measured cffg import: {cffg.__file__}")
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise RuntimeError("expected CPU with JAX float64 enabled")
    # The trace and log cover the first call. Their overhead is excluded from
    # the uninstrumented steady-state samples below.
    reset, run, setup, fixture = make_runner(arm)
    increment = jax.jit(lambda x: x + 1)
    probe_input = jax.numpy.asarray(0)

    def probe():
        increment(probe_input).block_until_ready()

    probe()
    with compile_log(jax) as events:
        first, initial, cold_trace = profile(jax, lambda: run(SWEEPS), probe, require_complete=False)
    del initial
    for _ in range(warmup):
        reset()
        run(SWEEPS)
    samples, results = [], None
    for _ in range(repetitions):
        reset()
        gc.collect()
        elapsed, results = measured(lambda: run(SWEEPS))
        samples.append(elapsed)
    members, messages = results_record(results, SWEEPS)

    submissions = {}
    for budget in (1, trace_budget):
        if budget != SWEEPS:
            reset()
            run(budget)  # warm each static budget before counting
        reset()
        elapsed, profiled, counts = profile(jax, lambda: run(budget), probe, require_complete=False)
        results_record(profiled, budget)
        submissions[str(budget)] = {**counts, "profiled_wall_s": elapsed}

    np.save(output.with_suffix(".npy"), messages, allow_pickle=False)
    artifact = {
        "arm": arm, "setup": setup, "fixture": fixture, "members": members,
        "first_call_s": first, "first_call_trace": cold_trace, "cold_compile_events": events,
        "first_call_xla_compile_s": sum(e["seconds"] for e in events if e["phase"] == "XLA compilation"),
        "warm_s": seconds(samples), "submissions_by_sweep_budget": submissions,
        "runtime": {"python": platform.python_version(), "jax": jax.__version__, "jaxlib": jaxlib.__version__,
                    "numpy": np.__version__, "device": str(jax.devices()[0]), "backend": jax.default_backend()},
    }
    output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")


def command(*args, cwd):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def hardware():
    detail = {"machine": platform.machine(), "platform": platform.platform(), "logical_cpus": os.cpu_count()}
    if sys.platform == "darwin":
        for field, key in (("model", "hw.model"), ("cpu", "machdep.cpu.brand_string"),
                           ("memory_bytes", "hw.memsize"), ("performance_cores", "hw.perflevel0.logicalcpu"),
                           ("efficiency_cores", "hw.perflevel1.logicalcpu")):
            value = command("sysctl", "-n", key, cwd=Path.cwd())
            detail[field] = int(value) if value.isdigit() else value
    return detail


def max_scaled_difference(left, right):
    import numpy as np
    return float(np.max(np.abs(left - right) / (1 + np.abs(left))))


def compare(before, after):
    matches = ("benchmark_sha256", "lock_sha256", "controls", "hardware")
    for key in matches:
        if before[key] != after[key]:
            raise ValueError(f"baseline and candidate differ in {key}")
    output = {}
    max_fe_error = 0.
    for arm in ARMS:
        old, new = before["arms"][arm], after["arms"][arm]
        if old["runtime"] != new["runtime"] or old["fixture"] != new["fixture"]:
            raise ValueError(f"incompatible baseline runtime or fixture: {arm}")
        for reference, candidate in zip(old["members"], new["members"], strict=True):
            for field in ("status", "iterations", "trace_lengths", "diagnostic"):
                if reference[field] != candidate[field]:
                    raise ValueError(f"baseline and candidate {field} differ in {arm}")
            expected = reference["final_free_energy"]
            error = abs(candidate["final_free_energy"] - expected) / (1 + abs(expected))
            max_fe_error = max(max_fe_error, error)
        output[arm + "_warm_speedup"] = old["warm_s"]["median"] / new["warm_s"]["median"]
    if max_fe_error > 1e-10:
        raise ValueError(f"baseline and candidate free energy differ: {max_fe_error}")
    return {"baseline_sha": before["commit"], "max_scaled_final_free_energy_difference": max_fe_error, **output,
            "single_no_slower_than_frozen_host": output["single_warm_speedup"] >= 1.}


def submission_parity(traces):
    if not all(trace["complete"] for trace in traces.values()):
        return None
    return len({trace["cpu_executable_submissions"] for trace in traces.values()}) == 1


def run_study(args):
    import numpy as np
    repo = args.checkout.resolve()
    if repo.joinpath("uv.lock").read_bytes() != Path(__file__).resolve().parents[1].joinpath("uv.lock").read_bytes():
        raise ValueError("measured checkout and launcher dependency locks differ")
    commit = command("git", "rev-parse", "HEAD", cwd=repo)
    script = Path(__file__).resolve()
    script_digest = hashlib.sha256(script.read_bytes()).hexdigest()

    def ensure_snapshot():
        runtime_status = command("git", "status", "--porcelain", "--untracked-files=all",
                                 "--", "src", "uv.lock", "pyproject.toml", cwd=repo)
        if command("git", "rev-parse", "HEAD", cwd=repo) != commit or runtime_status:
            raise ValueError(f"runtime sources or dependencies differ from {commit}: {runtime_status}")
        if hashlib.sha256(script.read_bytes()).hexdigest() != script_digest:
            raise ValueError("benchmark source changed during the study")

    ensure_snapshot()
    env = {**os.environ, "PYTHONPATH": str(repo / "src"), "JAX_PLATFORMS": "cpu",
           "JAX_ENABLE_X64": "true", "JAX_ENABLE_COMPILATION_CACHE": "false"}
    arms, messages = {}, {}
    with tempfile.TemporaryDirectory(prefix="cffg-ordinary-study-") as directory:
        for arm in ARMS:
            path = Path(directory) / f"{arm}.json"
            subprocess.run([sys.executable, str(script), "--worker", arm,
                            "--output", str(path), "--warmup", str(args.warmup),
                            "--repetitions", str(args.repetitions), "--trace-budget", str(args.trace_budget)],
                           env=env, cwd=repo, check=True)
            arms[arm] = json.loads(path.read_text())
            messages[arm] = np.load(path.with_suffix(".npy"), allow_pickle=False)
            print(f"{arm}: {arms[arm]['warm_s']['median']:.6f} s warm median", flush=True)
    ensure_snapshot()
    max_gap = max_scaled_difference(messages["sequential"], messages["colony"])
    single_gap = max_scaled_difference(messages["sequential"][:1], messages["single"])
    if not max(max_gap, single_gap) <= 1e-10:
        raise RuntimeError(f"ordinary state parity failed: {max_gap=}, {single_gap=}")
    seq = arms["sequential"]["warm_s"]["median"]
    batch = arms["colony"]["warm_s"]["median"]
    report = {
        "commit": commit, "checkout": str(repo), "benchmark_sha256": script_digest,
        "lock_sha256": hashlib.sha256((repo / "uv.lock").read_bytes()).hexdigest(),
        "hardware": hardware(), "controls": {"members": MEMBERS, "sweeps": SWEEPS, "damp": DAMPING,
            "tol": None, "harden_last": SWEEPS, "seed": SEED, "warmup": args.warmup, "repetitions": args.repetitions},
        "historical": HISTORICAL, "arms": arms,
        "colony_over_sequential_throughput_ratio": seq / batch,
        "members_per_second": {"sequential": MEMBERS / seq, "colony": MEMBERS / batch},
        "max_scaled_message_difference": {"single": single_gap, "colony": max_gap},
        "timing_note": args.timing_note,
        "submission_count_definition": CPU_DISPATCH,
        "submission_trace_budgets": [1, args.trace_budget],
        "cold_timing_boundary": "first public 60-sweep call with host profiler and compile logging enabled",
        "warm_timing_boundary": "synchronized public inference from initial states with no instrumentation",
        "submissions_independent_of_budget": {arm: submission_parity(a["submissions_by_sweep_budget"])
                                               for arm, a in arms.items()},
    }
    if args.baseline is not None:
        report["paired_revision_comparison"] = compare(json.loads(args.baseline.read_text()), report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--checkout", type=Path, default=Path(__file__).resolve().parents[1])
    cli.add_argument("--baseline", type=Path)
    cli.add_argument("--warmup", type=int, default=2)
    cli.add_argument("--repetitions", type=int, default=5)
    cli.add_argument("--trace-budget", type=int, default=SWEEPS,
                     help="second profiler budget, independent of the 60-sweep wall measurements")
    cli.add_argument("--timing-note", help="measurement conditions to include in the evidence")
    cli.add_argument("--worker", choices=ARMS, help=argparse.SUPPRESS)
    args = cli.parse_args()
    if args.warmup < 1 or args.repetitions < 1:
        cli.error("warmup and repetitions must be positive")
    if not 1 < args.trace_budget <= SWEEPS:
        cli.error(f"trace budget must be between 2 and {SWEEPS}")
    if args.worker:
        worker(args.worker, args.output, args.warmup, args.repetitions, args.trace_budget)
    else:
        run_study(args)


if __name__ == "__main__":
    main()

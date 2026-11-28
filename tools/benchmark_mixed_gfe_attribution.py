"""Fresh-process cold and warm measurements for the retained mixed-GFE cases."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from functools import wraps
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

from benchmark_ordinary_iteration import compile_log, hardware, profile


ARM_MODES = ("host", "compiled")
DEFAULT_CASES = ("tmaze", "d1")
CASES = (*DEFAULT_CASES, "rolling10", "horizon8", "bfe")
HISTORICAL = ("600a775", "931758d")


def command(cwd, *arguments):
    return subprocess.check_output(arguments, cwd=cwd, text=True).strip()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


def result_fields(result):
    result_fields = {"iterations": result.iterations, "status": result.status.value}
    for name in ("outer_map_evaluations", "trial_evaluations", "linear_iterations", "solver_compilations",
                 "outer_residual", "outer_gradient_norm", "inner_residual"):
        result_fields[name] = finite(getattr(result, name, None))
    for name in ("inner_status", "termination_reason"):
        result_fields[name] = getattr(getattr(result, name, None), "value", None)
    result_fields["final_bfe"] = finite(result.free_energies[-1]) if result.free_energies else None
    failure = getattr(result, "failure", None)
    result_fields["failure"] = asdict(failure) if failure else None
    return result_fields


def time_result(jax, run):
    start = time.perf_counter()
    results = run()
    returned = time.perf_counter()
    jax.block_until_ready(tuple(r.state for r in results))
    ready = time.perf_counter()
    return {
        "seconds": ready - start,
        "public_call_s": returned - start,
        "explicit_final_block_s": ready - returned,
        "results": [result_fields(r) for r in results],
    }


def hook_clock(stack, owner, name, label, counts):
    """Time an existing host method without imposing extra synchronization."""
    original = getattr(owner, name, None)
    if original is None:
        return
    static = isinstance(inspect.getattr_static(owner, name), staticmethod)

    @wraps(original)
    def measure(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            counts.setdefault(label, []).append(time.perf_counter() - start)

    stack.enter_context(patch.object(owner, name, staticmethod(measure) if static else measure))


def topology(template, arrays):
    """Identify the numerical inputs independent of version-specific IR defaults."""
    import numpy as np

    factors = [
        (name, factor.type, sorted(factor.edges.items()),
         sorted(factor.param_blobs.items()), sorted(factor.param_edges.items()))
        for name, factor in sorted(template.factors.items())
    ]
    dimensions = [(name, v.cardinality, v.dim) for name, v in sorted(template.variables.items())]
    constraints = [(c.kind.value, c.target, c.detail) for c in template.constraints]
    ports = [(name, p.variable, p.role.value) for name, p in sorted(template.ports.items())]
    data = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        a = np.asarray(array)
        data.update(repr((name, a.shape, a.dtype.str)).encode())
        data.update(a.tobytes())
    return {"factor_graph_sha256": sha256(json.dumps([factors, dimensions, constraints, ports]).encode()),
            "numeric_inputs_sha256": data.hexdigest(), "factors": len(factors)}


def worker(case, mode, repetitions, output, attribution):
    import cffg
    import jax
    import jaxlib
    import numpy as np

    import benchmark_mixed_gfe as fixture
    from cffg.engine import Engine

    source = Path.cwd()
    if Path(cffg.__file__).resolve().parent != source/"src/cffg":
        raise RuntimeError(f"unexpected CFFG code root: {cffg.__file__}")
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise ValueError("requested benchmark requires CPU and x64")
    opt_in = mode == "compiled"
    if opt_in and "outer_solver" not in inspect.signature(Engine.run).parameters:
        raise ValueError("compiled opt-in unavailable at this revision")

    original_run = Engine.run
    prepare = Engine.init_state
    compile_fixture = fixture.compile_template
    run_controls = []
    model_inputs = []

    @wraps(original_run)
    def selected(engine, state, budget, **kwargs):
        if opt_in:
            kwargs["outer_solver"] = "compiled_trust_region"
        invocation = {"budget": budget, **kwargs}
        if invocation not in run_controls:
            run_controls.append(invocation)
        return original_run(engine, state, budget, **kwargs)

    @wraps(prepare)
    def prepared(engine):
        return jax.block_until_ready(prepare(engine))

    @wraps(compile_fixture)
    def compilation(template, *, arrays):
        model_inputs.append(topology(template, arrays))
        return compile_fixture(template, arrays=arrays)

    report = {"case": case, "mode": mode, "attribution_worker": attribution,
              "commit": command(source, "git", "rev-parse", "HEAD"),
              "runtime": {"python": platform.python_version(), "jax": jax.__version__,
                          "jaxlib": jaxlib.__version__, "numpy": np.__version__,
                          "x64": True, "device": str(jax.devices()[0]), "backend": "cpu"}}
    with (
        patch.object(Engine, "run", selected),
        patch.object(Engine, "init_state", prepared),
        patch.object(fixture, "compile_template", compilation),
    ):
        start = time.perf_counter()
        run = fixture.CASES[case]()
        report["construction_seconds"] = time.perf_counter() - start
        report["fixture_inputs"] = model_inputs
        if attribution:
            from cffg import outer

            hooks = {}
            increment = jax.jit(lambda x: x + 1)
            probe_input = jax.numpy.asarray(0)

            def probe():
                increment(probe_input).block_until_ready()

            probe()
            with ExitStack() as stack:
                hook_clock(stack, outer, "build_outer_chart", "chart_construction", hooks)
                hook_clock(stack, Engine, "bfe", "host_bfe", hooks)
                hook_clock(stack, Engine, "_state_is_finite", "host_state_finite", hooks)
                hook_clock(stack, Engine, "_has_zero_support", "host_zero_support", hooks)
                hook_clock(stack, getattr(outer, "OuterChart", None), "support_changed",
                           "host_chart_support", hooks)
                with compile_log(jax) as events:
                    _, cold, trace = profile(jax, lambda: time_result(jax, run), probe,
                                             require_complete=False)
            report["trace"] = trace
            report["host_phases"] = {name: {"calls": len(spans), "total_s": sum(spans)}
                                     for name, spans in hooks.items()}
            report["cold"] = {**cold, "compiles": events}
        else:
            with compile_log(jax) as events:
                cold = time_result(jax, run)
            report["cold"] = {**cold, "compiles": events}
            report["discarded_warmup"] = time_result(jax, run)
            report["warm"] = [time_result(jax, run) for _ in range(repetitions)]
    report["run_controls"] = run_controls
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


def stats(rows, warm_repetitions):
    cold = [r["cold"]["seconds"] for r in rows]
    warm = [item["seconds"] for row in rows for item in row["warm"]]
    if len(warm) != len(cold)*warm_repetitions:
        raise ValueError("incorrect warm sample inventory")
    first = rows[0]
    for row in rows[1:]:
        for key in ("commit", "case", "mode", "runtime", "fixture_inputs", "run_controls"):
            if row[key] != first[key]:
                raise ValueError(f"workers differ in {key}")
    fields = [x for row in rows for step in [row["cold"], *row["warm"]] for x in step["results"]]
    statuses = sorted({x["status"] for x in fields})
    residuals = [x["outer_residual"] for x in fields if x["outer_residual"] is not None]
    host_compile_s = [sum(event["seconds"] for event in row["cold"]["compiles"]
                          if event["phase"] == "XLA compilation") for row in rows]
    return {
        "commit": first["commit"], "case": first["case"], "mode": first["mode"],
        "cold_median_s": statistics.median(cold), "warm_median_s": statistics.median(warm),
        "cold_samples": len(cold), "warm_samples": len(warm), "statuses": statuses,
        "max_reported_outer_residual": max(residuals, default=None),
        "cold_median_xla_compile_s": statistics.median(host_compile_s),
        "construction_median_s": statistics.median(row["construction_seconds"] for row in rows),
        "outcome_comparable": statuses == ["converged"], "fixture_inputs": first["fixture_inputs"],
        "run_controls": first["run_controls"],
    }


def stack_targets(revisions, cases):
    for label, checkout in revisions:
        for case in cases:
            yield label, checkout, case, "host"
            if label == "current" and case != "bfe":
                yield label, checkout, case, "compiled"


def study(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    source_receipt = {p.name: sha256(p.read_bytes()) for p in (
        script, script.with_name("benchmark_mixed_gfe.py"), script.with_name("benchmark_ordinary_iteration.py")
    )}
    revisions = [("current", args.checkout.resolve())]
    revisions.extend((head, path.resolve()) for head, path in zip(HISTORICAL, args.references, strict=True))
    locked_heads = {label: command(repo, "git", "rev-parse", "HEAD") for label, repo in revisions}
    for label, repo in revisions:
        if label != "current" and not locked_heads[label].startswith(label):
            raise ValueError(f"wrong revision at {repo}")
        if command(repo, "git", "status", "--porcelain", "--", "src", "uv.lock"):
            raise ValueError(f"unclean runtime revision at {repo}")
    report = {"controls": {"cases": args.cases, "cold_samples": args.cold_samples,
                            "warm_samples_per_process": args.warm_samples, "discarded_warmups": 1,
                            "persistent_jax_cache": False, "cpu_trace_attribution": args.attribute},
              "source_sha256": source_receipt, "hardware": hardware(), "commits": locked_heads, "arms": {}}
    # Each sample gets its own process. Interleave revisions so a thermal trend
    # does not affect one revision exclusively.
    targets = list(stack_targets(revisions, args.cases))

    def launch(target, purpose, sample):
        label, repo, case, mode = target
        stem = f"{label}-{case}-{mode}-{purpose}-{sample:02d}"
        result = output/f"{stem}.json"
        env = {**os.environ, "PYTHONPATH": str(repo/"src"), "JAX_PLATFORMS": "cpu",
               "JAX_ENABLE_X64": "true", "JAX_ENABLE_COMPILATION_CACHE": "false"}
        arguments = [sys.executable, str(script), "--worker", case, "--mode", mode,
                     "--warm-samples", str(args.warm_samples), "--output", str(result)]
        if purpose == "trace":
            arguments.append("--attribute")
        with (output/f"{stem}.log").open("x") as log:
            subprocess.run(arguments, env=env, cwd=repo, stdout=log, stderr=log, check=True)
        data = json.loads(result.read_text())
        if data["commit"] != locked_heads[label]:
            raise RuntimeError(f"measured source changed during {stem}")
        return data

    timing = {target: [] for target in targets}
    for sample in range(args.cold_samples):
        for target in targets:
            timing[target].append(launch(target, "timing", sample))
            print(f"{target[0]} {target[2]} {target[3]} sample {sample+1}/{args.cold_samples}", flush=True)
    for target, samples in timing.items():
        label, _, case, mode = target
        arm = stats(samples, args.warm_samples)
        if args.attribute and label in ("current", "600a775"):
            trace = launch(target, "trace", 0)
            arm["trace"] = trace["trace"]
            arm["host_phases"] = trace["host_phases"]
        report["arms"][f"{label}/{case}/{mode}"] = arm
    matched = {}
    for case in args.cases:
        arms = [a for a in report["arms"].values() if a["case"] == case]
        input_matches = all(a["fixture_inputs"] == arms[0]["fixture_inputs"] for a in arms)
        request_only = lambda a: [{k: v for k, v in row.items() if k != "outer_solver"}
                                  for row in a["run_controls"]]
        controls_match = all(request_only(a) == request_only(arms[0]) for a in arms)
        matched[case] = {"available_topology_matches": input_matches,
                         "explicit_controls_match": controls_match}
    if source_receipt != {name: sha256((script.parent/name).read_bytes()) for name in source_receipt}:
        raise RuntimeError("benchmark source changed during measurement")
    report["matched"] = matched
    with (output/"summary.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--checkout", type=Path, default=Path(__file__).resolve().parents[1])
    cli.add_argument("--references", type=Path, nargs=2, metavar=("PROOF", "PRE_PROOF"))
    cli.add_argument("--cases", nargs="+", choices=CASES, default=DEFAULT_CASES)
    cli.add_argument("--cold-samples", type=int, default=5)
    cli.add_argument("--warm-samples", type=int, default=5)
    cli.add_argument("--attribute", action="store_true")
    cli.add_argument("--worker", choices=CASES, help=argparse.SUPPRESS)
    cli.add_argument("--mode", choices=ARM_MODES, default="host", help=argparse.SUPPRESS)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.cold_samples < 1 or args.warm_samples < 1:
        cli.error("sample counts must be positive")
    if args.worker:
        worker(args.worker, args.mode, args.warm_samples, args.output, args.attribute)
    elif args.references is None:
        cli.error("--references required for a matched study")
    elif len(set(args.cases)) != len(args.cases):
        cli.error("case names must be unique")
    else:
        study(args)


if __name__ == "__main__":
    main()

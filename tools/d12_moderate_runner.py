"""Opt-in A3WZ evidence runner. No moderate work happens at import or in ``plan``.

Inspect: uv run python tools/d12_moderate_runner.py plan
After review and authorization, use a NEW directory outside the checkout:
  uv run python tools/d12_moderate_runner.py run --phase solves --output /tmp/d12-moderate-001
  uv run python tools/d12_moderate_runner.py run --phase storage --output /tmp/d12-storage-001
Validate: uv run python tools/d12_moderate_runner.py validate /tmp/d12-moderate-001
Separate phase invocations cannot reuse executables from terminated children.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

from d12_compile_scaling import (
    checked_arrays,
    child_environment,
    compile_program,
    digest,
    native_jax,
    supervise,
    write_json,
)
from d12_moderate_plan import ROOT, SCHEMA, MODES, identities, load_spec


def runtime_identity(jax):
    import jaxlib
    import numpy as np

    clock = time.get_clock_info("perf_counter")
    return dict(
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        machine=platform.machine(),
        jax=jax.__version__,
        jaxlib=jaxlib.__version__,
        numpy=np.__version__,
        backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()],
        clock=dict(
            implementation=clock.implementation,
            monotonic=clock.monotonic,
            adjustable=clock.adjustable,
            resolution=clock.resolution,
        ),
        environment={
            k: os.environ.get(k)
            for k in (
                "JAX_ENABLE_X64",
                "JAX_EXPLICIT_X64_DTYPES",
                "JAX_PLATFORMS",
                "JAX_ENABLE_COMPILATION_CACHE",
                "JAX_COMPILATION_CACHE_DIR",
                "XLA_FLAGS",
            )
        },
        config={
            k: v.name if hasattr(v, "name") else v
            for k, v in jax.config.values.items()
            if "cache" in k or "x64" in k
        },
    )


def timed_call(executable, arguments, compilation, checkpoint, stage):
    """One authorized call, including synchronization, never a warm-up or repeat."""
    import jax

    checked_arrays(arguments)
    jax.block_until_ready(arguments)
    compilation["calls"]["execute"] += 1
    checkpoint(stage)
    start = time.perf_counter_ns()
    result = jax.block_until_ready(executable(*arguments))
    seconds = (time.perf_counter_ns() - start) / 1e9
    checked_arrays(result)
    return result, seconds


def save_state(folder, name, coordinate, state):
    from d12_moderate_payload import write_snapshot
    from d12_moderate_workload import output_values
    import jax

    start = time.perf_counter_ns()
    values = jax.block_until_ready(output_values(coordinate, state))
    checked_arrays(values)
    snapshot = write_snapshot(folder, name, values)
    return snapshot, (time.perf_counter_ns() - start) / 1e9


def run_solves(horizon, folder, report, checkpoint, spec):
    import jax
    from cffg import hybrid_joint_solver as solver, hybrid_joint_storage as storage
    from d12_compile_fixture import context
    from d12_moderate_payload import write_snapshot
    from d12_moderate_workload import initialize, raw_coordinate

    coordinate, parameters, sites = context(horizon)
    inputs = dict(parameters=parameters, sites=sites, metadata=coordinate.metadata())
    checked_arrays(inputs)
    report["inputs"] = write_snapshot(folder, "inputs", inputs)
    report["cases"] = []
    report["initialization_compilation"] = {}
    report["solve_compilation"] = {}
    xi = jax.numpy.asarray(0, jax.numpy.float32)
    mode = jax.numpy.asarray(0, jax.numpy.int32)
    init = compile_program(
        lambda p, s, xi, mode: initialize(coordinate, p, s, xi, mode),
        (parameters, sites, xi, mode),
        report["initialization_compilation"],
        checkpoint,
    )
    function = storage.programs(coordinate, solver.Budgets(**spec["budgets"]))[
        "solve_from_coordinate"
    ]
    solve = compile_program(
        function,
        (parameters, sites, raw_coordinate(coordinate, xi)),
        report["solve_compilation"],
        checkpoint,
    )
    for case in (c for c in spec["cases"] if c["T"] == horizon):
        record = dict(case=case)
        report["cases"].append(record)
        xi = jax.numpy.asarray(case["xi"], jax.numpy.float32)
        mode = jax.numpy.asarray(MODES.index(case["messages"]), jax.numpy.int32)
        state, record["initialization_seconds"] = timed_call(
            init[1],
            (parameters, sites, xi, mode),
            report["initialization_compilation"],
            checkpoint,
            "initialize",
        )
        record["initial"], initial_export = save_state(
            folder, case["id"] + "-initial", coordinate, state
        )
        record["initial_status"] = dict(
            outcome=int(state.status),
            reason=int(state.reason),
            numerical=int(state.evaluation.status),
        )
        checkpoint("initial_saved")
        if int(state.status) != solver.Outcome.RUNNING:
            raise ValueError(
                f"initialization failed for {case['id']}; no optimizer call authorized for invalid input"
            )
        result, record["solve_seconds"] = timed_call(
            solve[1],
            (parameters, sites, state.coordinate),
            report["solve_compilation"],
            checkpoint,
            "solve",
        )
        record["terminal"], terminal_export = save_state(
            folder, case["id"] + "-terminal", coordinate, result
        )
        record["export_seconds"] = initial_export + terminal_export
        record["terminal_status"] = dict(
            outcome=int(result.status),
            reason=int(result.reason),
            numerical=int(result.evaluation.status),
        )
        checkpoint("terminal_saved")
        if int(result.status) != solver.Outcome.CONVERGED:
            raise ValueError(
                f"{case['id']} did not converge; preserve first failure and stop without retry"
            )
    report["retained_objects"] = len(init) + len(solve)
    report["reuse"] = (
        "Six distinct authorized starts share the same retained solve executable; initialization has its own executable"
    )


def run_storage(case, folder, report, checkpoint, spec):
    from cffg import hybrid_joint_solver as solver, hybrid_joint_storage as storage
    from d12_compile_fixture import context
    from d12_moderate_workload import logical_inventory, compile_arguments

    coordinate, parameters, sites = context(case["T"])
    report["case"] = case
    report["logical"] = logical_inventory(coordinate, parameters, sites)
    report["fixed_metadata"] = checked_arrays(coordinate.metadata())
    report["compilation"] = {}
    function = storage.programs(coordinate, solver.Budgets(**case["budgets"]))[
        case["program"]
    ]
    retained = compile_program(
        function,
        compile_arguments(coordinate, parameters, sites, case["program"]),
        report["compilation"],
        checkpoint,
    )
    report["retained_objects"] = len(retained)
    report["reuse"] = (
        "Zero executable calls. Memory-only cap variants never run an optimizer."
    )


def run_inventory(report, spec):
    """Keep wide abstract tracing out of each compiled variant's process peak."""
    from d12_compile_fixture import context
    from d12_moderate_workload import wide_context, logical_inventory

    report["shape_inventory"] = []
    for shape in spec["storage"]["logical_shapes"]:
        ctx = (
            context(shape["T"])
            if shape["K"] == 4
            else wide_context(shape["T"], shape["K"], shape["d"])
        )
        report["shape_inventory"].append(
            dict(shape=shape, inventory=logical_inventory(*ctx))
        )
    report["complete_program_calls"] = dict(lower=0, compile=0, execute=0)


def worker(phase, job_id, folder):
    spec = load_spec()
    report = dict(
        schema=SCHEMA,
        phase=phase,
        identity=identities(),
        pid=os.getpid(),
        status="started",
        started_utc=datetime.now(timezone.utc).isoformat(),
    )

    def checkpoint(stage):
        report.update(stage=stage, stage_started_ns=time.monotonic_ns())
        write_json(folder / "worker.json", report)

    checkpoint("setup")
    try:
        jax = native_jax()
        import cffg
        from cffg import hybrid_joint_solver as solver

        if Path(cffg.__file__).resolve().parent != ROOT / "src/cffg":
            raise RuntimeError("cffg imported from another checkout")
        if (
            report["identity"]["git_status"]
            or asdict(solver.Budgets()) != spec["budgets"]
        ):
            raise ValueError(
                "qualification requires clean committed sources and frozen defaults"
            )
        report["runtime"] = runtime_identity(jax)
        if phase == "solves":
            run_solves(int(job_id.removeprefix("t")), folder, report, checkpoint, spec)
        elif job_id == "inventory":
            run_inventory(report, spec)
        else:
            case = next(c for c in spec["storage"]["cases"] if c["id"] == job_id)
            run_storage(case, folder, report, checkpoint, spec)
        if identities() != report["identity"]:
            raise RuntimeError("source/repository state changed during measurement")
        report["status"] = "complete"
        checkpoint("complete")
        return 0
    except BaseException as error:
        report.update(
            status="failed",
            failure=dict(
                type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            ),
        )
        checkpoint("failed")
        traceback.print_exc()
        return 1


def run(phase, output):
    spec = load_spec()
    identity = identities()
    if identity["git_status"]:
        raise ValueError(
            "commit implementation/specification and complete review before qualification"
        )
    output = output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("use a new evidence directory outside the checkout")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        schema=SCHEMA, phase=phase, identity=identity, status="started", jobs=[]
    )
    write_json(output / "spec.json", spec)
    write_json(output / "report.json", report)
    jobs = (
        [f"t{t}" for t in (32, 100)]
        if phase == "solves"
        else ["inventory"] + [c["id"] for c in spec["storage"]["cases"]]
    )
    limits = dict(spec["limits"])
    if phase == "storage":
        limits["worker_seconds"] = limits["storage_worker_seconds"]
    try:
        for job in jobs:
            folder = output / job
            folder.mkdir()
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--phase",
                phase,
                "--job",
                job,
                "--output",
                str(folder),
            ]
            process = supervise(command, folder, limits, child_environment())
            entry = dict(id=job, process_sha256=digest(folder / "process.json"))
            report["jobs"].append(entry)
            if (folder / "worker.json").exists():
                entry["worker_sha256"] = digest(folder / "worker.json")
            if process["exit_code"] != 0 or process["stop"] is not None:
                raise RuntimeError(
                    f"{job} failed; partial evidence is preserved; remaining jobs were not started"
                )
            write_json(output / "report.json", report)
        if identities() != identity:
            raise RuntimeError("source/repository state changed during measurement")
        report["status"] = "complete"
        write_json(output / "report.json", report)
        from d12_moderate_report import validate_report

        result = validate_report(output)
        write_json(output / "validation.json", result)
        return 0
    except BaseException as error:
        report.update(
            status="failed", failure=dict(type=type(error).__name__, message=str(error))
        )
        write_json(output / "report.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    for name in ("run", "worker"):
        command = commands.add_parser(name)
        command.add_argument("--phase", choices=("solves", "storage"), required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "worker":
            command.add_argument("--job", required=True)
    commands.add_parser("validate").add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(load_spec(), indent=2))
        return 0
    if args.command == "validate":
        from d12_moderate_report import validate_report

        print(json.dumps(validate_report(args.directory), indent=2))
        return 0
    if args.command == "worker":
        return worker(args.phase, args.job, args.output)
    return run(args.phase, args.output)


if __name__ == "__main__":
    sys.exit(main())

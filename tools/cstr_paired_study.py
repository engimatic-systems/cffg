"""Run the ten held-out PCGym seed pairs sequentially in fresh processes.

Run from the checked-out library:
    uv run --frozen python tools/cstr_paired_study.py run --output NEW_DIRECTORY

The separate replay and report commands consume that output without changing it.
"""
import argparse
from dataclasses import asdict
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cstr_reactor_pilot as pilot


RUNTIME_FILES = ("tools/cstr_reactor_pilot.py", "tools/cstr_paired_study.py",
                 "tools/cstr_host_comparator.py", "tools/benchmark_cstr_chain.py",
                 "tools/cstr_runtime_fixture.py", "uv.lock")


def run_arm(output, experiment, seed, arm, order):
    launch = float(os.environ["CSTR_STUDY_LAUNCH_MONOTONIC"])
    output.mkdir(parents=True, exist_ok=False)
    make_env, params, info = pilot.load_runtime(experiment)
    for source in RUNTIME_FILES:
        if arm == "directed" and source == "tools/cstr_host_comparator.py":
            continue
        path = pilot.ROOT / source
        info["file_sha256"][str(path)] = pilot.sha(path)
    setup = time.perf_counter()
    if arm == "directed":
        _, arrays = pilot.fixture.model()
        controller = pilot.decide
        arm_model = arrays
    else:
        from cstr_host_comparator import create

        controller = create(experiment, seed).decide
        arrays = {}
        import experiment_physics as host
        arm_model = dict(configuration=asdict(host.CONFIG), constants="frozen physical prior mean",
                         initialization="shift the previous accepted plan", rk4_substeps=8)
        for module in ("agent", "graphs", "physics_exp", "setpoint_exp"):
            path = experiment / "common" / f"{module}.py"
            info["file_sha256"][str(path)] = pilot.sha(path)
    setup_s = time.perf_counter() - setup
    document = dict(seed=seed, arm=arm, order=order, parameters=params, protocol=pilot.PROTOCOL,
                    configuration=pilot.fixture.CONFIGURATION, runtime=info, model=arm_model,
                    launch_epoch=float(os.environ["CSTR_STUDY_LAUNCH_EPOCH"]),
                    launch_boundary="before uv", setup_s=setup_s)
    # Persist inputs before each trajectory. File writing stays outside the per-decision clock.
    pilot.save(output / "manifest.json", document)
    summary = pilot.episode(output, make_env, params, arrays, launch, seed=seed, controller=controller)
    pilot.save(output / "episode.json", summary)
    return 0 if summary["complete"] else 1


def run_study(output, experiment):
    dirty = pilot.git("status", "--porcelain", "--", "src", *RUNTIME_FILES)
    if dirty:
        raise RuntimeError(f"Commit the measured runtime sources before starting a study: {dirty}")
    output.mkdir(parents=True, exist_ok=False)
    seeds = pilot.PROTOCOL["held_out_paired_seeds"]
    order_by_seed = {seed: ["directed", "host"] if seed % 2 else ["host", "directed"]
                     for seed in seeds}
    (output / "study.json").write_text(json.dumps(dict(
        protocol=pilot.PROTOCOL, order_by_seed=order_by_seed, revision=pilot.git("rev-parse", "HEAD"),
        runtime_sources={p: pilot.sha(pilot.ROOT / p) for p in RUNTIME_FILES},
        started_epoch=time.time()), indent=2) + "\n")
    finished = []
    for seed, arms in order_by_seed.items():
        for order, arm in enumerate(arms):
            directory = output / f"seed-{seed:02d}" / arm
            directory.parent.mkdir(exist_ok=True)
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            environment.update(PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                               VECLIB_MAXIMUM_THREADS="1", MKL_NUM_THREADS="1",
                               JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1", JAX_ENABLE_COMPILATION_CACHE="false")
            command = ["uv", "run", "--frozen", "--no-sync", "python", str(Path(__file__).resolve()),
                       "arm", "--output", str(directory), "--experiment", str(experiment),
                       "--seed", str(seed), "--arm", arm, "--order", str(order)]
            with (directory.parent / f"{arm}.log").open("x") as log:
                start = time.perf_counter()
                environment.update(CSTR_STUDY_LAUNCH_MONOTONIC=str(start),
                                   CSTR_STUDY_LAUNCH_EPOCH=str(time.time()))
                process = subprocess.run(command, cwd=pilot.ROOT, env=environment, stdout=log,
                                         stderr=subprocess.STDOUT)
            finished.append(dict(seed=seed, arm=arm, exit_code=process.returncode,
                                 wall_s=time.perf_counter() - start))
            print(f"Seed {seed} {arm}: exit {process.returncode} in {finished[-1]['wall_s']:.2f}s", flush=True)
    (output / "processes.json").write_text(json.dumps(finished, indent=2) + "\n")
    return int(any(arm["exit_code"] for arm in finished))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    run, arm = sub.add_parser("run"), sub.add_parser("arm")
    for branch in (run, arm):
        branch.add_argument("--output", type=Path, required=True)
        branch.add_argument("--experiment", type=Path, default=pilot.EXPERIMENT)
    arm.add_argument("--seed", type=int, choices=range(11), required=True)
    arm.add_argument("--arm", choices=("directed", "host"), required=True)
    arm.add_argument("--order", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    output, experiment = args.output.resolve(), args.experiment.resolve()
    if args.mode == "arm":
        return run_arm(output, experiment, args.seed, args.arm, args.order)
    return run_study(output, experiment)


if __name__ == "__main__":
    sys.exit(main())

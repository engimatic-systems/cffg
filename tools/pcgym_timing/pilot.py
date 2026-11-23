"""Development timing pilot. Run each phase in a fresh, single-threaded CPU process.

Run with uv run --frozen python tools/pcgym_timing/pilot.py --help.
The output directory contains private peer installations and the saved truth fixture.
"""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import math
import subprocess
import sys
from time import perf_counter
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = Path.home() / "experiments/cffg-testing/pc-gym/00/cffg"
ARMS = ("cffg", "rxinfer", "numpyro", "gtsam")
PHASES = ("oracle", "short", "long", "forecast")
SEED = 0
DEADLINE = 1.


def jsonable(x):
    if isinstance(x, Path):
        return str(x)
    if hasattr(x, "tolist"):
        return x.tolist()
    raise TypeError(str(type(x)))


def diagnostic(value):
    """Keep nonfinite failure diagnostics legible in strict JSON evidence."""
    if isinstance(value, dict):
        return {k: diagnostic(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic(v) for v in value]
    if hasattr(value, "tolist"):
        return diagnostic(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def save(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, default=jsonable, allow_nan=False, indent=2) + "\n")
    temp.replace(path)


def parameters(base, long):
    if not long:
        return base
    # The first 60 samples reproduce the original schedule. Longer holds and
    # repeated setpoint changes exercise both steady operation and transients.
    first = list(base["SP"]["Ca"])
    motif = sum(([level] * count for level, count in (
        (.85, 60), (.90, 10), (.87, 30), (.90, 50), (.85, 20), (.87, 10))), [])
    levels = first + motif * 3
    assert len(levels) == 600
    return {**base, "N": len(levels), "tsim": base["tsim"] * len(levels) / base["N"],
            "SP": {"Ca": levels}}


@contextmanager
def plant_rng(stream):
    import numpy as np
    ambient = np.random.get_state()
    np.random.set_state(stream.get_state())
    try:
        yield
    finally:
        stream.set_state(np.random.get_state())
        np.random.set_state(ambient)


def rng_digest(stream):
    name, data, pos, cache, value = stream.get_state()
    return sha256(data.tobytes() + repr((name, pos, cache, value)).encode()).hexdigest()


def latency(samples):
    import numpy as np
    return {name: float(value) for name, value in zip(
        ("mean", "median", "p95", "max"),
        (np.mean(samples), np.median(samples), np.quantile(samples, .95), np.max(samples)))}


def episode(adapter, make_env, params, raw, window, seed=SEED):
    import numpy as np
    stream = np.random.RandomState(seed)
    with plant_rng(stream):
        env = make_env(params)
        obs, _ = env.reset(seed=seed)
    last = 0.
    history, rows = [], []
    started = perf_counter()
    with raw.open("x") as out:
        for tick in range(params["N"] - 1):
            start = perf_counter()
            data = window(params, history, obs, tick)["data"]
            begin_solve = perf_counter()
            action, msg = None, None
            native = {}
            try:
                decision = adapter.solve(data)
                native = {k:v for k,v in decision.items() if k not in ("mean","cov","action")}
                status = decision["status"]
                if status == "ok":
                    action = float(decision["action"])
                    if not np.isfinite(action):
                        status, action = "nonfinite", None
            except Exception as exc:
                status, msg = "error", f"{type(exc).__name__}: {exc}"
            solved = perf_counter()
            proposed = np.clip(action, -1., 1.) if status == "ok" else last
            command = np.array([proposed], dtype=float)
            ready = perf_counter()
            late = ready - start > DEADLINE
            if late or status != "ok":
                command[0] = last
            last = float(command[0])
            with plant_rng(stream):
                next_obs, reward, done, truncated, _ = env.step(command)
            reward = float(np.asarray(reward).item())
            truth = np.array(env.state[:2], copy=True)
            target = params["SP"]["Ca"][env.t]
            if not np.isfinite([*truth, *next_obs[:2], reward]).all() or env.t != tick + 1:
                raise ArithmeticError(f"Invalid plant output at tick {tick}")
            row = dict(tick=tick, obs=obs[:2], target=target, next_true_state=truth, reward=reward,
                       error_Ca=float(truth[0]-target), action=action, applied=last, late=late,
                       held=late or status != "ok", status=status, error=msg, native=diagnostic(native),
                       solve_s=solved-begin_solve, decision_s=ready-start)
            out.write(json.dumps(row, default=jsonable, allow_nan=False) + "\n")
            out.flush()
            rows.append(row)
            history.append((obs.copy(), last))
            obs = next_obs
            if done or truncated:
                break
    expected = params["N"] - 1
    if len(rows) != expected or not (done or truncated):
        raise RuntimeError(f"Unexpected termination at {len(rows)}/{expected}")
    env.close()
    return dict(n=expected, complete=True, seed=seed, episode_s=perf_counter()-started,
                return_sum=sum(r["reward"] for r in rows), rmse_Ca=float(np.sqrt(np.mean(
                    [r["error_Ca"]**2 for r in rows]))), missed=sum(r["late"] for r in rows),
                holds=sum(r["held"] for r in rows), failures=sum(r["status"] != "ok" for r in rows),
                latency_s=latency([r["decision_s"] for r in rows]), rng_digest=rng_digest(stream),
                first_decision_s=rows[0]["decision_s"])


def configure(args):
    # Set before importing any numerical runtime, including a Julia subprocess.
    os.environ.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1", JAX_ENABLE_COMPILATION_CACHE="false",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                      MKL_NUM_THREADS="1", JULIA_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    sys.path[:0] = [str(ROOT / "src"), str(HERE.parent)]
    if args.arm in ("gtsam", "numpyro"):
        extra = args.peer_root / ("gtsam/site" if args.arm == "gtsam" else "numpyro-site")
        sys.path.append(str(extra))


def worker(args, destination):
    configure(args)
    import cstr_reactor_pilot as pilot
    make_env, base, runtime = pilot.load_runtime(args.experiment)
    import cstr_runtime_fixture as fixture
    _, arrays = fixture.model()
    launch = float(os.environ["PCGYM_TIMING_START"])
    metadata = dict(runtime=runtime, phase=args.phase, arm=args.arm, seed=args.seed,
                    start_to_imports_s=perf_counter()-launch)
    files = [p for p in HERE.iterdir() if p.suffix in (".py", ".jl", ".cpp", ".sh")]
    artifacts = {
        "numpyro": ("numpyro-arm.lock", "numpyro-extra.lock"),
        "gtsam": ("gtsam/requirements.txt", "gtsam/build.lock"),
        "rxinfer": ("rxinfer/Project.toml", "rxinfer/Manifest.toml"),
    }
    metadata["dependency_sha256"] = {
        name: sha256((args.peer_root / name).read_bytes()).hexdigest()
        for name in artifacts.get(args.arm, ()) if (args.peer_root / name).exists()
    }
    if args.arm == "gtsam":
        metadata["native_binary_sha256"] = {
            p.name: sha256(p.read_bytes()).hexdigest()
            for p in (args.peer_root / "gtsam" / "site").glob("_gtsam_cstr*.so")
        }
    metadata["source_sha256"] = {p.name: sha256(p.read_bytes()).hexdigest() for p in files}
    manifest = destination / "manifest.json"
    save(manifest, metadata)
    if args.phase == "truth":
        from forecast import prepare
        prepare(make_env, parameters(base, True), arrays, args.output / "forecast-truth.json",
                study_seed=args.seed)
        return dict(complete=True)
    module = importlib.import_module(args.arm + "_adapter")
    for name in ("numpyro", "gtsam"):
        if args.arm == name:
            metadata["peer_version"] = importlib.metadata.version(name)
    save(manifest, metadata)
    if args.phase == "oracle":
        from qualification import qualify
        rows = qualify(module.Adapter, arrays)
        return dict(complete=all(r["passed"] for r in rows), cases=rows)
    before = perf_counter()
    adapter = module.Adapter(arrays, fixture.CONFIGURATION)
    setup = perf_counter()-before
    if hasattr(adapter, "metadata"):
        metadata["adapter"] = diagnostic(adapter.metadata)
        save(manifest, metadata)
    try:
        if args.phase == "forecast":
            from forecast import run
            result = run(adapter, arrays, args.output / "forecast-truth.json", study_seed=args.seed)
        else:
            before = perf_counter()
            for hist in (True, False):
                warm = adapter.solve(fixture.observations(change=2, history=hist, goal=.19))
                if warm["status"] != "ok":
                    raise RuntimeError(f"Synthetic warmup failed: {warm}")
            warmup = perf_counter()-before
            before = perf_counter()
            result = episode(adapter, make_env, parameters(base, args.phase == "long"),
                             destination / "ticks.jsonl", pilot.window, args.seed or SEED)
            result.update(warmup_s=warmup, previous_stage_to_episode_s=before-launch)
        result.update(adapter_setup_s=setup)
        return result
    finally:
        if hasattr(adapter, "close"):
            adapter.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("truth", *PHASES))
    p.add_argument("--arm", choices=ARMS, default="cffg")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--peer-root", type=Path, help="Pinned library installations and dependency locks")
    p.add_argument("--seed", type=int, help="Paired study seed. Omit for the original development pilot.")
    p.add_argument("--experiment", type=Path, default=EXPERIMENT)
    p.add_argument("--run-name", help="New run directory name, e.g. numpyro-oracle-2")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.peer_root = (args.peer_root or args.output).resolve()
    destination = args.output / (args.run_name or f"{args.arm}-{args.phase}")
    if args.worker:
        try:
            answer = worker(args, destination)
            save(destination / "result.json", answer)
            return int(answer.get("complete") is False or answer.get("status") == "failed")
        except Exception:
            save(destination / "error.json", dict(error=traceback.format_exc()))
            raise
    destination.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    with (destination / "worker.log").open("x") as log:
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
                              env={**os.environ, "PCGYM_TIMING_START":str(start)}, stdout=log,
                              stderr=subprocess.STDOUT)
    elapsed = perf_counter()-start
    save(destination / "process.json", dict(wall_s=elapsed, code=proc.returncode))
    print(f"{args.arm} {args.phase}: exit {proc.returncode}, {elapsed:.2f}s, {destination}", flush=True)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())

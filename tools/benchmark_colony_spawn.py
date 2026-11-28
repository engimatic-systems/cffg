"""Measure validated parameter-only Colony spawning on a retained CSTR layout.

The affine transition is synthetic. The model shape, observations and parameter
families come from the retained P4/H15 reactor fixture. Run with a clean baseline
checkout. Each arm uses the candidate's locked Python environment.
"""

from argparse import ArgumentParser, SUPPRESS
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from time import perf_counter


HERE = Path(__file__).resolve()
SIZES = (1, 40, 400)


def head(directory):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()


def fixture():
    import numpy as np

    from cffg.core import BlobRef
    from cstr_runtime_fixture import model, observations

    template, arrays = model()
    template.name = "affine-cstr-layout-spawn"
    template.constraints.clear()
    arrays["transition"] = np.diag([.9, .85, .98, .97])
    arrays["transition_offset"] = np.zeros(4)
    for name in ("transition", "transition_offset"):
        template.blobs[name] = BlobRef(name, f"blobs/{name}.npy")
    for name, factor in template.factors.items():
        if factor.type == "gaussian_nonlinear":
            template.factors[name] = replace(
                factor, type="gaussian_linear", function=None,
                param_blobs={"matrix": "transition", "offset": "transition_offset",
                             "cov": factor.param_blobs["cov"]},
            )
    inputs = []
    for i in range(max(SIZES)):
        varied = {**arrays, "m0": np.array([i/1000, -.1, 0., .02])}
        data = observations(change=(i % 5) * .1, history=True, goal=.1)
        inputs.append((varied, data))
    return template, inputs


def timed(call):
    start = perf_counter()
    result = call()
    return perf_counter() - start, result


def worker(repetitions):
    import jax
    import numpy as np

    from cffg import compiler
    from cffg.colony import Colony
    from cffg.engine import RunStatus

    expected = Path.cwd() / "src" / "cffg" / "compiler.py"
    if Path(compiler.__file__).resolve() != expected.resolve():
        raise RuntimeError("benchmark imported a different checkout")
    jax.block_until_ready(jax.numpy.zeros(()))
    template, inputs = fixture()
    counts = dict(topology=0, assembly=0)

    def counted(key, function):
        def measured(*args, **kwargs):
            counts[key] += 1
            return function(*args, **kwargs)
        return measured

    compiler.build_topology = counted("topology", compiler.build_topology)
    compiler.assemble_arena = counted("assembly", compiler.assemble_arena)
    rows = {}
    for size in SIZES:
        samples = []
        for _ in range(repetitions):
            colony = Colony()
            before = counts.copy()

            def spawn(rows):
                ids = [colony.spawn(template, arrays) for arrays, _ in rows]
                jax.block_until_ready([colony.instances[i].state for i in ids])
                return ids

            first, first_id = timed(lambda: spawn(inputs[:1]))
            cached, later_ids = timed(lambda: spawn(inputs[1:size])) if size > 1 else (0., [])
            ids = first_id + later_ids
            compilation = {name: counts[name] - value for name, value in before.items()}

            def bind():
                for i, (_, data) in zip(ids, inputs[:size], strict=True):
                    colony.bind(i, data)
                jax.block_until_ready([colony.instances[i].state for i in ids])

            binding, _ = timed(bind)
            engine = next(iter(colony.engines.values()))
            if len(colony.engines) != 1 or engine.schedule.kind != "two_pass":
                raise RuntimeError("benchmark changed structure grouping or exact dispatch")

            # Warm exact executable on the first member without replacing its state.
            def solve(state):
                result = engine.run(state, 1, tol=1e-6)
                jax.block_until_ready(result.state)
                return result

            warmup, _ = timed(lambda: solve(colony.instances[ids[0]].state))

            def infer():
                results = colony.run(1, tol=1e-6)
                jax.block_until_ready([r.state for r in results.values()])
                return results

            inference, results = timed(infer)
            digest = hashlib.sha256()
            for result in results.values():
                if result.status is not RunStatus.CONVERGED or result.iterations != 1:
                    raise ArithmeticError(f"expected a certified exact result at {size}")
                for arr in (result.state.g_eta, result.state.g_lam, result.free_energies,
                            result.message_deltas, result.residual):
                    digest.update(np.asarray(arr).tobytes())
            samples.append({"first_spawn_s": first, "cached_spawns_s": cached,
                            "bind_s": binding, "first_inference_warmup_s": warmup,
                            "warm_inference_s": inference, "compile_counts": compilation,
                            "result_sha256": digest.hexdigest()})
        times = {key: median(row[key] for row in samples) for key in samples[0] if key.endswith("_s")}
        rows[str(size)] = {"median": times, "samples": samples}

    print(json.dumps({"head": head(Path.cwd()), "jax": jax.__version__, "device": str(jax.devices()[0]),
                      "numpy": np.__version__, "rows": rows}, allow_nan=False))


def run(checkout, repetitions):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(checkout/"src"), JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
    raw = subprocess.check_output(
        [sys.executable, str(HERE), "--worker", "--repetitions", str(repetitions)],
        cwd=checkout, env=env, text=True,
    )
    return json.loads(raw)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=SUPPRESS)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.worker:
        worker(args.repetitions)
        return
    if not args.baseline or not args.output:
        parser.error("--baseline and --output are required")
    checkouts = {"baseline": args.baseline.resolve(), "candidate": HERE.parents[1]}
    for checkout in checkouts.values():
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True)
        if dirty.strip():
            raise RuntimeError(f"benchmark source checkout must be clean: {checkout}")

    report = {name: run(checkout, args.repetitions) for name, checkout in checkouts.items()}
    report["runtime"] = {
        "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "python": platform.python_version(), "jax_platform": "cpu", "x64": True,
        "fixture": "synthetic affine transitions on the retained 4D CSTR P4/H15 layout",
        "source_sha256": hashlib.sha256(HERE.read_bytes()).hexdigest(),
        "lock_sha256": hashlib.sha256((HERE.parents[1] / "uv.lock").read_bytes()).hexdigest(),
    }
    for size in SIZES:
        old, new = (report[role]["rows"][str(size)]["samples"] for role in checkouts)
        if {s["result_sha256"] for s in old+new} != {old[0]["result_sha256"]}:
            raise ArithmeticError(f"numerical results differ at batch size {size}")
        for samples, expected in ((old, size), (new, 1)):
            if any(s["compile_counts"] != {"topology": expected, "assembly": expected}
                   for s in samples):
                raise ArithmeticError(f"unexpected structural work at batch size {size}")
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

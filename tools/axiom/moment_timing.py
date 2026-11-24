"""Compare synchronized Engine.gaussian_node_stats against a pinned revision on CPU.

Run with --candidate COMMIT --output NEW_DIRECTORY in an isolated timing slot.
Use --plan to inspect the protocol without starting JAX.
"""
import argparse
import contextlib
import hashlib
import io
import itertools
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path


BASE = "8d46be45f0bba11971f2dcd32db78d0137f3128f"
ARMS = ("old_default", "current_default", "centered")
SHAPES = list(itertools.product((32, 500, 5000), (2, 6)))
CALLS = 5
MATERIAL = 1.20
ENV = {"JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "0", "JAX_DISABLE_JIT": "0",
       "JAX_ENABLE_COMPILATION_CACHE": "0", "OPENBLAS_NUM_THREADS": "1",
       "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "OMP_NUM_THREADS": "1",
       "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"}


def orders():
    # Every arm occupies every position four times per shape.
    rng = random.Random(7104)
    result = []
    for _ in range(2):
        permutations = list(itertools.permutations(ARMS))
        rng.shuffle(permutations)
        result.extend(permutations)
    return result


def protocol():
    return dict(baseline=BASE, arms=ARMS, shapes=SHAPES, gaussian_mixture_members=1,
                dtype="float32", environment=ENV, block_order=orders(), calls_per_block=CALLS,
                warmup_calls_after_first=3, material_median_ratio=MATERIAL,
                interval="percentile bootstrap median of 12 paired block-mean ratios, 10000 resamples, seed 1047",
                endpoints="full public host readout of sensor and mixture; no outer JIT, graph setup or inference inside clock",
                bytes="sum of returned array nbytes; logical host-visible payload, not physical CPU bus traffic")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def worker(arm, k, d):
    imported = time.perf_counter()
    import jax
    import numpy as np
    from moment_probe import Engine, compile_template, metadata, model

    runtime = metadata()
    assert Path(runtime["source_directory"]) == Path(os.environ["PYTHONPATH"]) / "cffg"
    assert jax.default_backend() == "cpu" and not jax.config.x64_enabled
    import_s = time.perf_counter() - imported
    started = time.perf_counter()
    idx = np.arange(k)
    mean = ((idx[:, None] % 17 - 8) / 16 + np.arange(d)[None, :] / 8)
    covariance = (1.25*np.eye(d) + .0625) * (1 + .125*(idx % 5)[:, None, None])
    prior = 1 + idx % 7
    arrays = {key: np.array(value, np.float32) for key, value in
              dict(m=mean, V=covariance, p=prior/prior.sum(), y=np.linspace(-.5, .5, d), R=.75*np.eye(d)).items()}
    engine = Engine(compile_template(model(arrays), arrays=arrays))
    state = jax.block_until_ready(engine.init_state())
    setup_s = time.perf_counter() - started
    started = time.perf_counter()
    state = jax.block_until_ready(engine.step(state))
    step_s = time.perf_counter() - started
    read_stats = engine.gaussian_node_stats
    kwargs = {"centered_mixtures": True} if arm == "centered" else {}

    def read():
        # The public API returns NumPy arrays. Materialize every leaf even if
        # a future version changes its return type to asynchronous device arrays.
        out = read_stats(state, **kwargs)
        return {node: {key: np.asarray(value) for key, value in record.items()}
                for node, record in out.items()}

    started = time.perf_counter()
    first = read()
    first_s = time.perf_counter() - started
    assert set(first) == {"object", "sensor"} and all(row["proper"] for row in first.values())
    assert len(engine.arena.mix_groups) == 1 and len(engine.arena.mix_groups[0].members) == 1
    weight = first["object"]["responsibility"]
    assert weight.shape == (k,) and weight.dtype == np.float32 and np.all(weight > 0)
    np.testing.assert_allclose(weight.sum(), 1, rtol=2e-6)
    keys = (("component_mean", "component_covariance") if arm == "centered"
            else ("weighted_mean", "weighted_second"))
    for key, shape in zip(keys, ((k, d), (k, d, d)), strict=True):
        value = first["object"][key]
        assert value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all()
    mixture_bytes = sum(a.nbytes for a in first["object"].values())
    assert mixture_bytes == (k + k*d + k*d*d)*4 + 1
    payload = {node: {key: dict(shape=value.shape, dtype=str(value.dtype), bytes=value.nbytes,
                                 sha256=sha(value.tobytes())) for key, value in row.items()}
               for node, row in first.items()}
    initial = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(state):
        array = np.asarray(leaf)
        initial.update(repr((array.shape, array.dtype.str)).encode())
        initial.update(array.tobytes())
    for _ in range(3):
        read()
    print(json.dumps(dict(runtime=runtime, import_and_device_s=import_s, setup_s=setup_s, step_s=step_s,
                          first_read_s=first_s, state_sha256=initial.hexdigest(), payload=payload,
                          mixture_bytes=mixture_bytes, total_bytes=sum(v.nbytes for row in first.values()
                                                                       for v in row.values()))), flush=True)
    for line in sys.stdin:
        if line.strip() != "block":
            raise ValueError("Unexpected worker command")
        durations = []
        for _ in range(CALLS):
            start = time.perf_counter_ns()
            result = read()
            durations.append(time.perf_counter_ns() - start)
            del result
        print(json.dumps(durations), flush=True)


def summarize(blocks):
    def interval(values):
        generator = random.Random(1047)
        draws = sorted(statistics.median(generator.choices(values, k=len(values))) for _ in range(10000))
        return [draws[249], draws[9749]]

    means = {arm: [statistics.mean(row[arm]) / 1e6 for row in blocks] for arm in ARMS}
    latency = {arm: dict(median_ms=statistics.median(values), range_ms=[min(values), max(values)])
               for arm, values in means.items()}
    ratios = {}
    for a, b in (("centered", "old_default"), ("centered", "current_default"), ("current_default", "old_default")):
        paired = [x/y for x, y in zip(means[a], means[b], strict=True)]
        ci = interval(paired)
        ratio = statistics.median(paired)
        ratios[a + "_over_" + b] = dict(median=ratio, block_range=[min(paired), max(paired)],
                                         ci95=ci, material=ratio > MATERIAL, ci_above_bound=ci[0] > MATERIAL)
    return dict(latency=latency, paired=ratios)


def snapshot(revision, root):
    tree = subprocess.check_output(["git", "archive", revision, "src/cffg", "uv.lock"])
    path = root / revision
    with tarfile.open(fileobj=io.BytesIO(tree)) as tar:
        tar.extractall(path, filter="data")
    return path


def response(proc):
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"Worker terminated with status {proc.wait()}")
    return json.loads(line)


@contextlib.contextmanager
def launch(root, arm, k, d, source):
    env = dict(os.environ, **ENV, PYTHONPATH=str(source / "src"))
    command = [sys.executable, str(root / "moment_timing.py"), "--worker", arm, str(k), str(d)]
    start = time.perf_counter()
    with subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as proc:
        try:
            ready = response(proc)
            ready["launch_to_warmed_ready_s"] = time.perf_counter() - start
            yield proc, ready
        finally:
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError(f"{arm} worker failed")


def run(args):
    revision = subprocess.check_output(["git", "rev-parse", "--verify", args.candidate + "^{commit}"], text=True).strip()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for script in ("moment_timing.py", "moment_probe.py"):
        shutil.copyfile(Path(__file__).with_name(script), root / script)
    sources = {rev: snapshot(rev, root) for rev in (BASE, revision)}
    lock = sha((sources[revision] / "uv.lock").read_bytes())
    if sha((sources[BASE] / "uv.lock").read_bytes()) != lock:
        raise ValueError("Compared revisions have different dependency locks")
    result = dict(protocol=protocol(), candidate=revision, uv_lock_sha256=lock,
                  timing_script_sha256=sha((root / "moment_timing.py").read_bytes()),
                  start_utc=datetime.now(timezone.utc).isoformat(), results=[])
    (root / "protocol.json").write_text(json.dumps(result, indent=2))
    for index, (k, d) in enumerate(SHAPES):
        processes, records = {}, {}
        launch_order = list(itertools.permutations(ARMS))[index % 6]
        with contextlib.ExitStack() as stack:
            for arm in launch_order:
                source = sources[BASE if arm == "old_default" else revision]
                processes[arm], records[arm] = stack.enter_context(launch(root, arm, k, d, source))
            old, compat, centered = (records[arm] for arm in ARMS)
            for field in ("state_sha256", "mixture_bytes", "total_bytes"):
                assert old[field] == compat[field] == centered[field], (k, d, field)
            assert old["payload"] == compat["payload"], (k, d, "legacy outputs changed")
            assert old["payload"]["sensor"] == centered["payload"]["sensor"]
            for key in ("responsibility", "proper"):
                assert old["payload"]["object"][key] == centered["payload"]["object"][key]
            blocks = []
            for order in orders():
                row = {}
                for arm in order:
                    proc = processes[arm]
                    proc.stdin.write("block\n")
                    proc.stdin.flush()
                    row[arm] = response(proc)
                blocks.append(row)
        result["results"].append(dict(k=k, d=d, launch_order=launch_order, workers=records,
                                      blocks_ns=blocks, **summarize(blocks)))
        print(f"K={k} d={d} complete", flush=True)
        (root / "partial.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    result["end_utc"] = datetime.now(timezone.utc).isoformat()
    (root / "timing.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (root / "partial.json").unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--worker", nargs=3, help=argparse.SUPPRESS)
    parser.add_argument("--candidate")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        arm, k, d = args.worker
        worker(arm, int(k), int(d))
    elif args.plan:
        print(json.dumps(protocol(), indent=2))
    elif args.candidate and args.output:
        run(args)
    else:
        parser.error("supply --plan or both --candidate and --output")


if __name__ == "__main__":
    main()

"""Paired mixed-learning time and checkpoint cost on two pinned CPU source trees.

Use 'plan' to inspect the frozen protocol. 'run CANDIDATE NEW_DIRECTORY' starts
measurement. Source snapshots beneath that directory isolate the two workers.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tarfile
import time


BASE = "fd8bfd164cbb274b201d5b41f79419058e99eb11"
SHAPES = [(n, k) for k in (32, 500, 5000) for n in (1, 32)]
STEPS = 8
BLOCKS = 12
ENV = dict(
    JAX_PLATFORMS="cpu",
    JAX_ENABLE_X64="0",
    JAX_ENABLE_COMPILATION_CACHE="0",
    JAX_DISABLE_JIT="0",
    OPENBLAS_NUM_THREADS="1",
    VECLIB_MAXIMUM_THREADS="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)


def plan():
    return dict(
        baseline=BASE,
        shapes=SHAPES,
        dimension=6,
        categorical_cardinalities=[3, 5],
        qualification_dtypes=["float32", "float64"],
        timed_dtype="float32",
        environment=ENV,
        initial_training_ticks=32,
        measured_ticks=STEPS,
        blocks=BLOCKS,
        warmups=3,
        design="Paired alternating old/new order, initial order rotated by shape. One idle process per arm. Fresh workers for each shape.",
        carry="Each block starts from an initialized bank learned on 32 distinct batches. All eight subsequent ticks score and update the preceding bank.",
        device="Device-resident JIT scan; final bank, last public score/weights and all validity flags synchronized; no host output conversion.",
        host="Eight host-originating NumPy batch calls to a jitted public score+update; all scores, weights, valid flags and posterior leaves materialized each time.",
        checkpoints="NPZ deflated files from learned banks. Public snapshot + buffered filesystem write; read + public restore + sync reported separately. Stored/uncompressed file sizes also recorded.",
        archive_baselines="Old/new at their actual learned states and old serializer on the candidate's identical visible values. Optional exact retained separately.",
        cost_flag="Paired median new/old duration >=1.10; required compact file bytes >=1.10x. Report useful improvements plus 95% percentile whole-block bootstrap intervals; exact storage supplementary.",
        data="Seeded distinct full SPD priors and fractional assignments; source float32. No held-out or AXIOM agent data.",
    )


def sequence(n, k, dtype="f4"):
    import numpy as np
    from cffg.mixed_learning import MixedBank, MixedBatch

    rng = np.random.default_rng(44107 + k)
    d = 6
    base = rng.normal(0, 0.35, (k, d))
    lower = np.tril(rng.normal(0, 0.14, (k, d, d)))
    lower[:, np.arange(d), np.arange(d)] += 1.5
    psi = lower @ lower.transpose(0, 2, 1)
    priors = [base, rng.uniform(1, 5, k), psi, rng.uniform(8, 14, k), rng.uniform(0.5, 4, k)]
    heads = tuple(rng.uniform(0.5, 5, (k, c)).astype(dtype) for c in (3, 5))
    b = MixedBank(*(a.astype(dtype) for a in priors), heads, np.arange(k) % 8 != 7)
    ticks = 32 + STEPS
    t = np.arange(ticks * n).reshape(ticks, n)
    rng = np.random.default_rng(52094 + n)
    mean = (rng.normal(0, 0.45, (ticks, n, d)) + 0.3 * np.sin(t[:, :, None] * 0.1 + np.arange(d))).astype(
        dtype
    )
    cov = ((0.125 * np.eye(d) + 0.03125) * (1 + t[:, :, None, None] % 3)).astype(dtype)

    def labels(p, c):
        return np.where(t % p == 1, -1, rng.integers(0, c, t.shape)).astype("i4")

    return b, MixedBatch(mean, cov, t % 7 != 4, (labels(4, 3), labels(7, 5)))


def worker(route, n, k, directory):
    started = time.perf_counter()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from cffg.mixed_learning import MixedBank, score, update
    from online_precision_probe import runtime, visible, payload, read_file, same_tree

    meta = runtime()
    assert Path(meta["source_root"]) == Path(os.environ["PYTHONPATH"]) / "cffg"
    assert jax.default_backend() == "cpu" and not jax.config.x64_enabled
    import_s = time.perf_counter() - started
    b, xt = sequence(n, k)

    def ashost(x):
        return jax.tree.map(np.asarray, x)

    def asdevice(x):
        return jax.tree.map(jnp.asarray, x)

    def next_bank(b, x):
        s, w, sv = score(b, x)
        post, uv = update(b, x, w)
        return post, s, w, sv & uv

    one = jax.jit(next_bank)

    def fused(b, x):
        # Expose the last readout as well as the entire final bank without
        # allocating the N*K score/weight outputs for every internal tick.
        def body(carry, xx):
            post, s, w, v = next_bank(carry[0], xx)
            return (post, s, w, carry[-1] & v), None

        blank = jnp.zeros((n, k), dtype=b.mu.dtype)
        return jax.lax.scan(body, (b, blank, blank, jnp.asarray(True)), x)[0]

    scan = jax.jit(fused)

    def initialize(b, x):
        def body(bank, xx):
            post, _, _, ok = next_bank(bank, xx)
            return post, ok

        return jax.lax.scan(body, b, x)

    started = time.perf_counter()
    trained, valid = jax.block_until_ready(
        jax.jit(initialize)(asdevice(b), asdevice(jax.tree.map(lambda a: a[:32], xt)))
    )
    assert np.all(valid)
    training_s = time.perf_counter() - started
    x = jax.tree.map(lambda a: a[32:], xt)
    dx = asdevice(x)
    host_inputs = [jax.tree.map(lambda a: a[t], x) for t in range(STEPS)]
    hb = ashost(trained)
    np.savez(directory / f"{route}-visible.npz", **visible(trained))

    def recurrent():
        return jax.block_until_ready(scan(trained, dx))

    def boundary():
        carry = hb
        valid = True
        for xx in host_inputs:
            carry, s, w, v = ashost(one(carry, xx))
            valid &= bool(v)
        return carry, s, w, valid

    functions = dict(scan=recurrent, public=boundary)
    cold = {}
    endpoints = {}
    for name, fn in functions.items():
        start = time.perf_counter_ns()
        result = fn()
        cold[name] = (time.perf_counter_ns() - start) / 1e6
        assert bool(result[-1])
        endpoints[name] = visible(result[0])
    # Fused/eight single calls may differ at the last rounding bit.
    for key in endpoints["scan"]:
        np.testing.assert_allclose(
            endpoints["scan"][key],
            endpoints["public"][key],
            rtol=32 * np.finfo("f4").eps,
            atol=32 * np.finfo("f4").eps,
            err_msg=key,
        )
    for _ in range(3):
        for fn in functions.values():
            assert bool(fn()[-1])
    fields = visible(trained)

    def nbytes(a):
        return sum(np.asarray(v).nbytes for v in jax.tree.leaves(a))

    receipt = dict(
        runtime=meta,
        import_s=import_s,
        prelearn_compile_and_run_s=training_s,
        cold_block_ms=cold,
        model_bytes=nbytes(trained),
        visible_bytes=nbytes(fields),
        n=n,
        k=k,
        fixture_sha256=hash_inputs(
            (b.mu, b.psi, b.kappa, b.nu, b.alpha, *b.categories, b.active, *jax.tree.leaves(xt))
        ),
        compact_payload_sha256=hash_mapping(payload(trained, "compact")),
    )
    print(json.dumps(receipt), flush=True)
    banks = {"compact": (trained, "compact")}
    if route == "new":
        banks["exact"] = (trained, "exact")

    def archive(label, record=False):
        measured, mode = banks[label]
        path = directory / f"{route}-{label}.npz"
        started = time.perf_counter_ns()
        mapping = payload(measured, mode)
        np.savez_compressed(path, **mapping)
        written = time.perf_counter_ns()
        restored = jax.block_until_ready(read_file(path))
        read = time.perf_counter_ns()
        if record:
            restored_visible = visible(restored)
            for key, value in visible(measured).items():
                np.testing.assert_array_equal(restored_visible[key], value)
            if mode == "exact":
                assert same_tree(measured, restored)
            with io.BytesIO() as raw:
                np.savez(raw, **mapping)
                size = raw.tell()
            return dict(
                file_bytes=path.stat().st_size,
                stored_npz_bytes=size,
                payload_bytes=nbytes(mapping),
                file_sha256=digest(path),
                payload_sha256=hash_mapping(mapping),
            )
        return dict(write_ms=(written - started) / 1e6, read_ms=(read - written) / 1e6)

    for line in sys.stdin:
        command = line.strip()
        if command == "checkpoint_setup":
            if route == "old":
                with np.load(directory / "new-visible.npz", allow_pickle=False) as a:
                    bank = MixedBank(
                        *(a[key] for key in ("mu", "kappa", "psi", "nu", "alpha")),
                        tuple(a[f"category_{i}"] for i in range(2)),
                        a["active"],
                    )
                banks["same_visible"] = (asdevice(bank), "compact")
            result = {label: archive(label, record=True) for label in banks}
        elif command in functions:
            start = time.perf_counter_ns()
            end = functions[command]()
            interval = (time.perf_counter_ns() - start) / 1e6
            assert bool(end[-1])
            result = interval / STEPS
        else:
            result = archive(command)
        print(json.dumps(result), flush=True)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hash_inputs(values):
    import numpy as np

    h = hashlib.sha256()
    for a in values:
        a = np.asarray(a)
        h.update(repr((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def hash_mapping(m):
    return hash_inputs(m[key] for key in sorted(m))


def summary(a, b):
    vals = [x / y for x, y in zip(a, b, strict=True)]
    rng = random.Random(3407)
    draws = sorted(statistics.median(rng.choices(vals, k=len(vals))) for _ in range(10000))
    ci = [draws[249], draws[9749]]
    mid = statistics.median(vals)
    return dict(
        old_median=statistics.median(b),
        new_median=statistics.median(a),
        ratio=mid,
        ci95=ci,
        range=[min(vals), max(vals)],
        threshold_crossed=mid >= 1.10,
        ci_above_threshold=ci[0] >= 1.10,
    )


def read(p):
    line = p.stdout.readline()
    if not line:
        raise RuntimeError(f"worker ended: {p.wait()}")
    return json.loads(line)


def ask(p, cmd):
    p.stdin.write(cmd + "\n")
    p.stdin.flush()
    return read(p)


@contextmanager
def start(root, rev, route, n, k, cell):
    env = dict(os.environ, **ENV, PYTHONPATH=str(root / rev / "src"))
    argv = [
        sys.executable,
        str(root / "online_precision_timing.py"),
        "worker",
        route,
        str(n),
        str(k),
        str(cell),
    ]
    with subprocess.Popen(argv, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as p:
        try:
            yield p, read(p)
        finally:
            p.stdin.close()
            if p.wait():
                raise RuntimeError(f"{route} worker failed")


def snapshot(rev, root):
    dat = subprocess.check_output(["git", "archive", rev, "src/cffg", "uv.lock"])
    with tarfile.open(fileobj=io.BytesIO(dat)) as a:
        a.extractall(root / rev, filter="data")


def run(candidate, root):
    candidate = subprocess.check_output(
        ["git", "rev-parse", "--verify", candidate + "^{commit}"], text=True
    ).strip()
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ("online_precision_probe.py", "online_precision_timing.py"):
        shutil.copyfile(Path(__file__).with_name(name), root / name)
    for rev in (BASE, candidate):
        snapshot(rev, root)
    assert digest(root / BASE / "uv.lock") == digest(root / candidate / "uv.lock")
    rec = dict(
        protocol=plan(),
        candidate=candidate,
        driver_sha256=digest(root / "online_precision_timing.py"),
        lock_sha256=digest(root / BASE / "uv.lock"),
        platform=platform.platform(),
        clock=vars(time.get_clock_info("perf_counter")),
        start_unix=time.time(),
        results=[],
    )
    (root / "protocol.json").write_text(json.dumps(rec, indent=2))
    for idx, (n, k) in enumerate(SHAPES):
        cell = root / f"n{n}-k{k}"
        cell.mkdir()
        routes = ["old", "new"][:: 1 if idx % 2 == 0 else -1]
        with ExitStack() as stack:
            workers, ready = {}, {}
            for route in routes:
                rev = BASE if route == "old" else candidate
                workers[route], ready[route] = stack.enter_context(start(root, rev, route, n, k, cell))
            assert ready["old"]["fixture_sha256"] == ready["new"]["fixture_sha256"]
            blocks = []
            for i in range(BLOCKS):
                row = {}
                for metric in ["scan", "public"][:: 1 if i % 2 == 0 else -1]:
                    row[metric] = {r: ask(workers[r], metric) for r in routes[:: 1 if i % 2 == 0 else -1]}
                blocks.append(row)
            info = {r: ask(workers[r], "checkpoint_setup") for r in routes}
            archive = []
            for i in range(BLOCKS):
                row = {}
                for r in routes[:: 1 if i % 2 == 0 else -1]:
                    labels = list(info[r])
                    row[r] = {label: ask(workers[r], label) for label in labels[:: 1 if i % 2 == 0 else -1]}
                archive.append(row)
        effects = {}
        for metric in ("scan", "public"):
            effects[metric + "_ms_per_tick"] = summary(
                [r[metric]["new"] for r in blocks], [r[metric]["old"] for r in blocks]
            )
        for label in ("compact", "same_visible"):
            for op in ("write_ms", "read_ms"):
                effects["compact_over_old_" + label + "_" + op] = summary(
                    [r["new"]["compact"][op] for r in archive], [r["old"][label][op] for r in archive]
                )
        effects["compact_file_new_over_old"] = (
            info["new"]["compact"]["file_bytes"] / info["old"]["compact"]["file_bytes"]
        )
        effects["compact_file_same_visible_ratio"] = (
            info["new"]["compact"]["file_bytes"] / info["old"]["same_visible"]["file_bytes"]
        )
        effects["exact_file_over_compact"] = (
            info["new"]["exact"]["file_bytes"] / info["new"]["compact"]["file_bytes"]
        )
        rec["results"].append(
            dict(
                n=n,
                k=k,
                workers=ready,
                blocks=blocks,
                archive_blocks=archive,
                checkpoints=info,
                effects=effects,
            )
        )
        print(f"N={n} K={k} finished", flush=True)
        (root / "partial.json").write_text(json.dumps(rec, indent=2, allow_nan=False))
    rec["finish_unix"] = time.time()
    (root / "timing.json").write_text(json.dumps(rec, indent=2, allow_nan=False))
    (root / "partial.json").unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    cmd = sub.add_parser("run")
    cmd.add_argument("candidate")
    cmd.add_argument("output", type=Path)
    cmd = sub.add_parser("worker")
    cmd.add_argument("route", choices=("old", "new"))
    cmd.add_argument("n", type=int)
    cmd.add_argument("k", type=int)
    cmd.add_argument("cell", type=Path)
    a = parser.parse_args()
    if a.cmd == "plan":
        print(json.dumps(plan(), indent=2))
    elif a.cmd == "worker":
        worker(a.route, a.n, a.k, a.cell)
    else:
        run(a.candidate, a.output)


if __name__ == "__main__":
    main()

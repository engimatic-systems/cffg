"""Causal native-agent worker for the frozen changing-corridor comparison."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import jax
import numpy as np
from scipy.special import digamma

from cffg.agent import RollingAgent
from cffg.bundle import load_bundle, save_bundle
from cffg.compiler import compile_template
from cffg.core.serialize import structure_hash
from cffg.engine import Engine
from cffg.learn import Learnables
from cffg.models import build_factored_corridor
from cffg.runtime_types import InferenceError

OWNER = "position_transition"
PROTOCOL = Path(__file__).with_name("changing_corridor_protocol.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, allow_nan=False, indent=2)
        handle.write("\n")


def revision():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROTOCOL.parent, text=True
    ).strip()


def model(config, arm, planning):
    raw = np.full((2, 3, 2, 2), config["reset_alpha"], dtype=np.float64)
    template, arrays = build_factored_corridor(
        past=0 if planning else 1,
        steps=1,
        length=2,
        modes=2,
        position_prior=np.array([1.0, 0.0]),
        mode_prior=np.array([1.0, 0.0]),
        position_transition=raw / raw.sum(axis=2, keepdims=True),
        mode_transition=np.array(config["mode_transition"]),
        actions=np.array(config["action_prior"]),
        goal_utility=config["goal_utility"],
        gfe=True,
        learn=(OWNER,),
        epistemic_b=planning and arm != "no_epistemics",
        transition_counts={OWNER: raw},
        transition_hazards={} if arm == "stationary" else {OWNER: config["hazard"]},
    )
    return template, arrays, {OWNER: raw.reshape((6, 2, 2))}


def prepare(root):
    config = json.loads(PROTOCOL.read_text())
    root.mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir()
    (root / "protocol.json").write_bytes(PROTOCOL.read_bytes())
    entries = {}
    for arm in config["arms"]:
        entries[arm] = {}
        for kind in ("plan", "filter"):
            template, arrays, priors = model(config, arm, kind == "plan")
            learner = Learnables(
                Engine(compile_template(template, arrays=arrays)), priors=priors
            )
            path = root / "models" / f"{arm}-{kind}.cffgz"
            save_bundle(path, template, arrays, learnables=learner)
            saved = load_bundle(path)
            if structure_hash(saved.template) != structure_hash(template):
                raise ValueError("bundle structure changed on reload")
            np.testing.assert_array_equal(
                saved.learnables.initial_params[OWNER]["alpha"], priors[OWNER]
            )
            entries[arm][kind] = dict(
                file=str(path.relative_to(root)),
                sha256=digest(path),
                structure_hash=structure_hash(template),
            )
    write_json(
        root / "models.json", dict(arms=entries, protocol_sha256=digest(PROTOCOL))
    )


def native_agent(root, config, models, arm, seed):
    archives = {}
    for kind, entry in models["arms"][arm].items():
        path = root / entry["file"]
        if digest(path) != entry["sha256"]:
            raise ValueError("model archive digest mismatch")
        saved = load_bundle(path)
        if structure_hash(saved.template) != entry["structure_hash"]:
            raise ValueError("model template mismatch")
        archives[kind] = saved

    def builder(*, past, steps, **options):
        if steps != 1 or past not in (0, 1):
            raise ValueError("undeclared study window")
        archive = archives["filter" if past else "plan"]
        return archive.template, archive.arrays

    priors = archives["plan"].learnables.initial_params[OWNER]["alpha"]
    np.testing.assert_array_equal(
        priors, archives["filter"].learnables.initial_params[OWNER]["alpha"]
    )
    return RollingAgent(
        builder,
        config["horizon"],
        learn=(OWNER,),
        priors={OWNER: priors},
        epistemic_b=arm != "no_epistemics",
        select="sample",
        seed=seed,
        n_sweeps=config["planning"]["sweeps"],
    )


def rss_reader():
    if sys.platform == "darwin":
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")

        def resident():
            buffer = ctypes.create_string_buffer(128)
            size = lib.proc_pidinfo(os.getpid(), 4, 0, buffer, len(buffer))
            if size < 16:
                raise RuntimeError("proc_pidinfo could not read resident size")
            return int.from_bytes(buffer.raw[8:16], sys.byteorder)

        return resident
    page = os.sysconf("SC_PAGE_SIZE")
    return lambda: int(Path("/proc/self/statm").read_text().split()[1]) * page


class Measurements:
    def __init__(self):
        self.read = rss_reader()
        self.lock = threading.Lock()
        self.active = None
        self.done = threading.Event()
        self.worker = threading.Thread(target=self.sample, daemon=True)
        self.worker.start()

    def sample(self):
        while not self.done.wait(0.005):
            with self.lock:
                if self.active is not None:
                    self.active["peak"] = max(self.active["peak"], self.read())

    @contextmanager
    def phase(self):
        with self.lock:
            record = {"peak": self.read()}
            self.active = record
        started = time.perf_counter()
        try:
            yield record
        finally:
            record["seconds"] = time.perf_counter() - started
            with self.lock:
                record["peak"] = max(record["peak"], self.read())
                self.active = None

    def close(self):
        self.done.set()
        self.worker.join()


def finite(value):
    return float(value) if np.isfinite(value) else None


def receipt(result):
    return dict(
        status=result.status.value,
        reason=result.termination_reason.value,
        inner_status=result.inner_status.value,
        outer_residual=finite(result.outer_residual),
        free_energy=finite(result.free_energies[-1]) if result.free_energies else None,
        iterations=result.iterations,
        message_residual=finite(result.residual),
    )


def certified(record, planning=False):
    return (
        record["status"] == record["inner_status"] == "converged"
        and record["free_energy"] is not None
        and (
            not planning
            or (
                record["reason"] == "root_converged"
                and record["outer_residual"] is not None
            )
        )
    )


def novelty(column):
    total = column.sum()
    mean = column / total
    return float(
        np.sum(mean * (digamma(column + 1) - digamma(total + 1) - np.log(mean)))
    )


def successors(pos, mode, x, m, action, uniforms):
    # The joint law factors conditional on the old state and applied action.
    return int(uniforms[0] >= pos[m, action, 0, x]), int(
        uniforms[1] >= mode[action, 0, m]
    )


def worker(root, arm, world, seed, development):
    config = json.loads((root / "protocol.json").read_text())
    models = json.loads((root / "models.json").read_text())
    if digest(root / "protocol.json") != digest(PROTOCOL) or models[
        "protocol_sha256"
    ] != digest(PROTOCOL):
        raise ValueError("protocol source or model manifest mismatch")
    seeds = [config["development_seed"]] if development else config["seeds"]
    if seed not in seeds or arm not in config["arms"] or world not in config["worlds"]:
        raise ValueError("cell outside registered inventory")
    fingerprint = dict(
        source_commit=revision(),
        protocol_sha256=digest(PROTOCOL),
        models_sha256=digest(root / "models.json"),
    )
    if not development:
        freeze = json.loads((root / "freeze.json").read_text())
        if any(freeze[k] != v for k, v in fingerprint.items()):
            raise ValueError("frozen revision or hashes differ")
    out = (root / "development" if development else root) / "cells" / str(seed)
    out.mkdir(exist_ok=True, parents=True)
    stem = out / f"{arm}-{world}"
    entry = models["arms"][arm]
    result = dict(
        seed=seed,
        world=world,
        arm=arm,
        completed_steps=0,
        status="failed",
        reason=None,
        **fingerprint,
        plan_hash=entry["plan"]["structure_hash"],
        filter_hash=entry["filter"]["structure_hash"],
        hardware=platform.machine(),
        jax=jax.__version__,
        x64=jax.config.x64_enabled,
    )
    if stem.with_suffix(".json").exists():
        raise FileExistsError(stem)
    pos, changed, mode = (
        np.asarray(config[k])
        for k in ("old_position", "new_position", "mode_transition")
    )
    warm = config["warmup"]
    post = config["development_followup" if development else "followup"]
    # Reserve the full physical uniform tape before the first action.
    uniforms = np.random.default_rng(np.random.SeedSequence([seed, 871])).random(
        (warm + post, 2)
    )
    recent = []
    visits = 0
    x, m = config["initial_state"]
    phase = "setup"
    attempt = {}
    meter = Measurements()
    try:
        with stem.with_suffix(".jsonl").open("x", buffering=1) as tape:
            try:
                learner = native_agent(root, config, models, arm, seed)
                bank = learner.learnables
                original = bank.accumulate
                update_measure = {}

                def measured_update(*args, **kwargs):
                    with meter.phase() as sample:
                        original(*args, **kwargs)
                    update_measure.update(sample)

                bank.accumulate = measured_update
                for t in range(warm + post):
                    attempt = {"t": t, "x": x, "m": m}
                    if t == warm:
                        if (
                            visits < config["saturation_min_visits"]
                            or max(recent[-config["saturation_window"] :])
                            > config["saturation_max_d1"]
                        ):
                            result.update(
                                status="unsaturated",
                                reason="prespecified warmup diagnostic",
                            )
                            break
                    alpha = bank.params[OWNER]["alpha"].copy()
                    forecast = bank.predicted_transition_counts(OWNER)
                    plan = None
                    plan_measure = {}
                    if t < warm:
                        u = 2 * (t % 2)
                        recent.append(novelty(forecast[2, :, 0]))
                        visits += int((m, x, u) == (0, 0, 2))
                    else:
                        phase = "planning"
                        with meter.phase() as plan_measure:
                            u = learner.act()
                        plan = receipt(learner._planning.result)
                        attempt["plan"] = plan
                        if not certified(plan, True):
                            raise ValueError("planning root qualification failed")
                    law = changed if t >= warm and world == "change" else pos
                    nx, nm = successors(law, mode, x, m, u, uniforms[t])
                    attempt.update(u=u, x_next=nx, m_next=nm)
                    phase = "filtering"
                    update_measure.clear()
                    start = time.perf_counter()
                    learner.observe(u, 2 * nx + nm)
                    observe_s = time.perf_counter() - start
                    filt = receipt(learner._last_filter_result)
                    attempt["filter"] = filt
                    if not certified(filt):
                        raise ValueError("filter qualification failed")
                    row = dict(
                        t=t,
                        x=x,
                        m=m,
                        u=u,
                        x_next=nx,
                        m_next=nm,
                        alpha=alpha.tolist(),
                        forecast=forecast.tolist(),
                        after=bank.params[OWNER]["alpha"].tolist(),
                        plan=plan,
                        filter=filt,
                        plan_s=plan_measure.get("seconds"),
                        observe_s=observe_s,
                        update_s=update_measure["seconds"],
                        plan_peak_rss_bytes=plan_measure.get("peak"),
                        update_peak_rss_bytes=update_measure["peak"],
                    )
                    tape.write(
                        json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"
                    )
                    result["completed_steps"] = t + 1
                    x, m = nx, nm
                else:
                    result["status"] = "complete"
            except Exception as error:
                if isinstance(error, InferenceError):
                    attempt["native"] = receipt(error.result)
                result.update(
                    reason=f"{type(error).__name__}: {error}",
                    failure=dict(phase=phase, **attempt),
                )
    finally:
        meter.close()
    result.update(
        warm_visits=visits, warm_last_d1=recent[-config["saturation_window"] :]
    )
    write_json(stem.with_suffix(".json"), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "development", "cell"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm")
    parser.add_argument("--world")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "cpu":
        parser.error("protocol requires CPU execution")
    if args.command == "prepare":
        prepare(args.root)
    else:
        res = worker(
            args.root, args.arm, args.world, args.seed, args.command == "development"
        )
        print(
            json.dumps(
                {
                    k: res[k]
                    for k in (
                        "arm",
                        "world",
                        "seed",
                        "status",
                        "reason",
                        "completed_steps",
                    )
                }
            )
        )

"""One experiment per fresh process. Results are pinned to its input bundle and tape."""

from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import jax
import numpy as np

from cffg import Engine, compile_template
from cffg.core.ir import PortRole
from cffg.core.plates import expand_plates
from .jobs import write
from .models import Document, json_bytes


def finite(value):
    return float(value) if np.isfinite(value) else None


def contract(template):
    t = expand_plates(template)
    return {
        name: {
            "family": v.family,
            "dimension": v.dim,
            "cardinality": v.cardinality,
            "ef": v.ef,
        }
        for name, p in t.ports.items()
        if p.role == PortRole.DATA
        for v in (t.variables[p.variable],)
    }


def random_observations(ports, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for name, p in sorted(ports.items()):
        if p["family"] == "gaussian":
            out[name] = rng.normal(size=p["dimension"]).tolist()
        elif p["family"] == "discrete":
            out[name] = int(rng.integers(p["cardinality"]))
        elif p["ef"] == "poisson":
            out[name] = int(rng.poisson(3))
        elif p["ef"] == "bernoulli":
            out[name] = int(rng.integers(2))
        else:
            raise ValueError(f"Supply observations for {name} ({p['ef']})")
    return out


def validate_tape(ports, obs):
    if not isinstance(obs, dict) or obs.keys() != ports.keys():
        raise ValueError(
            "Observation names must exactly match DATA ports: " + ", ".join(ports)
        )
    for name, value in obs.items():
        if value is None:
            continue
        a = np.asarray(value)
        if a.dtype.kind not in "fibu" or not np.isfinite(a).all():
            raise ValueError(f"{name}: expected finite numeric observations")
        p = ports[name]
        if p["family"] == "gaussian" and a.shape != (p["dimension"],):
            raise ValueError(f"{name}: expected a vector of length {p['dimension']}")
        if p["family"] == "discrete" and (
            a.ndim or a.dtype.kind not in "iu" or not 0 <= int(a) < p["cardinality"]
        ):
            raise ValueError(
                f"{name}: expected a categorical index in 0..{p['cardinality'] - 1}"
            )


def work(folder):
    spec = json.loads((folder / "request.json").read_text())

    def emit(kind, **fields):
        event = {"type": kind, "time": time.time(), **fields}
        with (folder / "events.jsonl").open("ab") as stream:
            stream.write(json.dumps(event, allow_nan=False).encode() + b"\n")

    def finish(result):
        write(folder / "result.json", result)
        emit(
            "finish",
            **{
                k: v
                for k, v in result.items()
                if k not in ("source", "ground_truth", "observations", "beliefs")
            },
        )

    emit("phase", message="Loading model revision")
    durations = defaultdict(float)

    def timing(name, duration, **kw):
        if name.startswith("/jax/core/compile/"):
            durations[name.rsplit("/", 1)[-1]] += duration * 1000

    jax.monitoring.register_event_duration_secs_listener(timing)
    startup = (
        time.time() - float(os.getenv("CFFG_WORKER_STARTED", time.time()))
    ) * 1000
    start = time.perf_counter()
    doc = Document.load((folder / "model.cffgz").read_bytes())
    ports = contract(doc.template)
    kind = spec["kind"]
    if kind == "generate":
        from .simulator import generate

        emit("phase", message="Requesting simulator source")
        context = {"DATA_ports": ports}
        if spec.get("include_structure"):
            context["factors"] = {
                name: {
                    "type": f.type,
                    "edges": f.edges,
                    "parameter_roles": f.param_blobs,
                }
                for name, f in doc.template.factors.items()
            }
            context["parameters"] = {
                name: list(a.shape) for name, a in doc.arrays.items()
            }
        result = generate(spec["prompt"], context)
        result.update(
            source_sha256=sha256(result["source"].encode()).hexdigest(), contract=ports
        )
        finish(result)
        return
    engine = Engine(compile_template(doc.template, arrays=doc.arrays))
    state = engine.init_state()
    if engine.arena.mu3_groups:
        state, _ = engine.prepare_mu3(state, jax.random.key(spec["seed"]))
    setup = (time.perf_counter() - start) * 1000
    tape_file = folder / "tape.json"
    if kind == "simulate":
        from .simulator import simulate

        emit("phase", message="Generating observation tape in isolated NumPy worker")
        generated, image = simulate(
            spec["source"], spec["seed"], spec["steps"], spec["parameters"], spec["id"]
        )
        obs = generated["observations"]
    elif tape_file.exists():
        obs = json.loads(tape_file.read_text())["observations"]
    elif spec.get("observations") is not None:
        obs = spec["observations"]
    else:
        obs = random_observations(ports, spec["seed"])
    validate_tape(ports, obs)
    state = engine.bind_data(state, obs)
    tape = {"observations": obs}
    if kind == "simulate":
        tape["ground_truth"] = generated.get("ground_truth")
    write(tape_file, tape)
    provenance = {
        "tape_sha256": sha256(json_bytes({"observations": obs})).hexdigest(),
        "process_startup_ms": startup,
        "setup_ms": setup,
        "device": str(jax.devices()[0]),
        "jax": jax.__version__,
        "engine_revision": os.getenv(
            "CFFG_STUDIO_ENGINE_REVISION", "development checkout"
        ),
        "dtype": str(state.messages.dtype),
        "schedule": engine.schedule.kind,
        "seed": spec["seed"],
        "cache": "fresh process, persistent compilation cache disabled",
    }
    if kind == "simulate":
        finish(
            {
                **provenance,
                "observations": obs,
                "ground_truth": tape["ground_truth"],
                "image": image,
                "source_sha256": sha256(spec["source"].encode()).hexdigest(),
            }
        )
        return
    emit("phase", message="Observations bound", **provenance)
    n = spec["iterations"]

    def evaluate(budget):
        tic = time.perf_counter()
        res = engine.run(state, budget, tol=spec["tolerance"], harden_last=0)
        jax.block_until_ready(res.state)
        return res, (time.perf_counter() - tic) * 1000

    if kind == "trace":
        emit(
            "phase",
            message="Replaying increasing iteration budgets from the same initial state",
        )
        next_step = 0
        for i in range(1, n + 1):
            while True:
                ctrl = json.loads((folder / "control.json").read_text())
                if ctrl["mode"] == "run" or ctrl["steps"] > next_step:
                    next_step = (
                        next_step + 1 if ctrl["mode"] == "pause" else ctrl["steps"]
                    )
                    break
                time.sleep(0.12)
            result, elapsed = evaluate(i)
            # A run can skip undefined free energy on early sweeps. Query the final state directly.
            try:
                energy = finite(engine.bfe(result.state))
            except (ValueError, FloatingPointError):
                energy = None
            emit(
                "sweep",
                iteration=i,
                completed=result.iterations,
                free_energy=energy,
                delta=finite(result.message_deltas[-1])
                if result.message_deltas
                else None,
                residual=finite(result.residual),
                elapsed_ms=elapsed,
                status=result.status.value,
            )
            if result.status.value == "converged":
                if i < n:
                    result, _ = evaluate(n)
                break
        cold = None
        warm = []
    else:
        emit("phase", message="Measuring first execution")
        result, cold = evaluate(n)
        warm = []
        if kind == "profile":
            emit("measurement", message="First execution complete", cold_ms=cold)
            for _ in range(spec["repetitions"]):
                result, duration = evaluate(n)
                warm.append(duration)
                emit("measurement", message="Warm execution", elapsed_ms=duration)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )
    detail = {
        **provenance,
        "status": result.status.value,
        "iterations": result.iterations,
        "residual": finite(result.residual),
        "inner_status": result.inner_status.value,
        "message_deltas": [finite(x) for x in result.message_deltas],
        "free_energies": [finite(x) for x in result.free_energies],
        "cold_ms": cold,
        "warm_ms": warm,
        "peak_rss_bytes": rss,
        "platform": platform.platform(),
        "compilation_ms": dict(durations),
        "device_memory": jax.devices()[0].memory_stats(),
        "failure": asdict(result.failure) if getattr(result, "failure", None) else None,
    }
    beliefs = {}
    for name, row in engine.marginals(result.state).items():
        beliefs[name] = [finite(x) for x in np.asarray(row).flat]
    for name, (mean, cov) in engine.gaussian_marginals(result.state).items():
        beliefs[name] = {
            "mean": [finite(x) for x in mean],
            "variance": [finite(x) for x in np.diag(cov)],
        }
    if kind == "profile":
        from cffg.resource_analysis import executable_reports, state_storage

        detail["executables"] = executable_reports(engine, state)
        detail["storage"] = state_storage(state)
    for name, row in engine.ef_marginals(result.state).items():
        beliefs[name] = {
            "family": row.family,
            "parameters": [finite(x) for x in row.params],
            "mean": finite(row.mean),
            "proper": bool(row.proper),
        }
    for name, row in engine.dirichlet_marginals(result.state).items():

        def numeric(a):
            return np.vectorize(finite, otypes=[object])(np.asarray(a)).tolist()

        beliefs[name] = {
            "family": "dirichlet",
            "alpha": numeric(row.alpha),
            "mean": numeric(row.mean),
            "proper": bool(row.proper),
        }
    detail["beliefs"] = beliefs
    finish(detail)


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    try:
        work(directory)
    except Exception as e:
        with (directory / "events.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "type": "error",
                        "time": time.time(),
                        "message": str(e) or type(e).__name__,
                    }
                )
                + "\n"
            )
        raise

"""Seal and run the primary Gameworld study in the order registered by its protocol."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from time import time_ns
import fcntl
import hashlib
import json
from importlib.metadata import version
import platform
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from reference_protocol import seed_pairs

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGES = ("development", "pre_evaluation", "evaluation")
MARKERS = dict(zip(STAGES, ("pilot_passed.json", "pre_evaluation_passed.json", "completed.json")))
REFERENCE_HASH = "ef5141f70e4ef6792a40e7ade7086109fc9e04b68afed19764294f9b53282caf"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        try:
            handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp, path)


@contextmanager
def exclusive(path, *, wait=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handle


def fingerprint():
    files = [REPO / f for f in ("uv.lock", "pyproject.toml")]
    files += list((REPO / "src/cffg").rglob("*.py"))
    files += [HERE / f for f in (
        "reference_protocol.json", "reference_cases.json", "reference_protocol.py",
        "reference_intermediates.py", "gameworld_replay.py", "gameworld_replay_check.py", "acted_smoke.py",
        "evaluation_campaign.py", "evaluation_episode.py", "evaluation_game.py",
        "evaluation_actor.py", "evaluation_checkpoint.py", "evaluation_reduction.py",
        "evaluation_report.py",
    )]
    files += [p for p in HERE.glob("evaluation_*.py") if not p.name.endswith("_check.py")]
    paths = {str(f.relative_to(REPO)) for f in files}
    return {name: digest(REPO / name) for name in sorted(paths)}


def runtime():
    return dict(python=platform.python_version(), system=platform.platform(), machine=platform.machine(),
                versions={name:version(name) for name in ("jax", "jaxlib", "numpy", "scipy", "equinox")})


def verify_source(study):
    if study["runtime"] != runtime():
        raise ValueError("Python runtime differs from the sealed candidate")
    found = fingerprint()
    if found != study["source_files"] or hashlib.sha256(canonical(found)).hexdigest() != study["source_sha256"]:
        raise ValueError("source fingerprint differs from the sealed candidate")
    if digest(Path(study["reference"]) / "reference-v1/protocol.json") != REFERENCE_HASH:
        raise ValueError("external Gameworld protocol changed")
    protocol = read(HERE / "reference_protocol.json")
    if (study["protocol_sha256"] != digest(HERE / "reference_protocol.json")
            or study["stages"] != {s: list(seed_pairs(protocol, s)) for s in STAGES}):
        raise ValueError("study differs from the registered configuration or seed inventory")


def seal(root, reference):
    if (root / "study.json").exists():
        verify_source(read(root / "study.json"))
        return
    protocol = read(HERE / "reference_protocol.json")
    ref_path = reference / "reference-v1/protocol.json"
    if digest(ref_path) != REFERENCE_HASH or read(ref_path)["protocol"] != protocol:
        raise ValueError("frozen reference protocol mismatch")
    sources = fingerprint()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *sources], cwd=REPO)
    if dirty.strip():
        raise ValueError("commit runtime and scoring sources before sealing")
    study = dict(version=1, stage_order=list(STAGES), reference=str(reference),
                 arm="untuned", dtype="float32", precision="compensated",
                 horizon=32, candidates=512, draws=3, reduction_cadence=500,
                 created=datetime.now(timezone.utc).isoformat(),
                 revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,text=True).strip(),
                 runtime=runtime(), module_fixtures=qualification_inputs(),
                 source_files=sources, source_sha256=hashlib.sha256(canonical(sources)).hexdigest(),
                 protocol_sha256=digest(HERE / "reference_protocol.json"), reference_sha256=digest(ref_path),
                 stages={s: list(seed_pairs(protocol, s)) for s in STAGES})
    publish(root / "study.json", study)


def run_name(case):
    return f"{case['game']}-{case['model_seed']}"


def audit_run(root, study, stage, case):
    path = root / stage / run_name(case)
    result = read(path / "result.json")
    if (result["status"] != "complete" or result["source_sha256"] != study["source_sha256"]
            or result["stage"] != stage or any(result[k] != v for k, v in case.items())):
        raise ValueError(f"incomplete or misidentified run at {path}")
    if digest(path / "steps.jsonl") != result["steps_sha256"]:
        raise ValueError(f"action trace digest mismatch at {path}")
    total = 0.
    count = 0
    with (path / "steps.jsonl").open() as stream:
        for tick, line in enumerate(stream):
            row = json.loads(line)
            if row["tick"] != tick or row["reduction_performed"] != ((tick+1) % 500 == 0):
                raise ValueError(f"action or reduction clock mismatch at {path}:{tick}")
            total += row["reward"]
            count += 1
    if count != case["steps"] or total != result["return_sum"]:
        raise ValueError(f"trace count/return mismatch at {path}")
    if stage == "pre_evaluation" and result["resume_matched_steps"] != case["steps"] - 500:
        raise ValueError(f"missing exact continuation comparison at {path}")
    return digest(path / "result.json")


def stage_receipt(root, study, stage):
    return dict(stage=stage, source_sha256=study["source_sha256"],
                runs={run_name(c): audit_run(root, study, stage, c) for c in study["stages"][stage]})


def qualification_inputs():
    files = sorted((REPO / "tests").glob("test_axiom_*.py"))
    return {str(path.relative_to(REPO)):digest(path) for path in files}


def qualify(root, study):
    """Exercise the frozen module fixtures, including the opt-in capacity cases."""
    inputs = qualification_inputs()
    if inputs != study["module_fixtures"]:
        raise ValueError("module fixtures differ from the freeze")
    path = root / "module_qualification.json"
    expected = dict(source_sha256=study["source_sha256"], fixtures=inputs, status="passed")
    if path.exists():
        if read(path) != expected:
            raise ValueError("module qualification differs from its frozen inputs")
        return
    cmd = [sys.executable, "-m", "pytest", "-q", "-m", "slow or not slow", *inputs]
    with (root / "module_qualification.log").open("ab") as log:
        subprocess.run(cmd, cwd=REPO, env=dict(os.environ, PYTHONPATH=str(REPO / "src")),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    publish(path, expected)


def verify_before(root, study, stage):
    if stage != "development":
        qualification = read(root / "module_qualification.json")
        if qualification != dict(source_sha256=study["source_sha256"], fixtures=study["module_fixtures"],
                                 status="passed"):
            raise ValueError("missing qualified module fixtures")
    for prior in STAGES[:STAGES.index(stage)]:
        if read(root / MARKERS[prior]) != stage_receipt(root, study, prior):
            raise ValueError(f"missing or invalid {prior} qualification")


def launch(root, study, stage, case):
    path = root / stage / run_name(case)
    path.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), JAX_PLATFORMS="cpu", JAX_ENABLE_X64="0",
               JAX_COMPILATION_CACHE_DIR=str(root / "jit-cache"))
    cmd = [sys.executable, "-u", str(HERE / "evaluation_episode.py"), str(root), stage,
           case["game"], str(case["model_seed"])]
    result = path / "result.json"
    for attempt in range(3):
        if result.is_file():
            saved = read(result)
            if saved.get("status") != "failed" or saved.get("infrastructure") is not True:
                audit_run(root, study, stage, case)
                return
            # Keep the failed receipt; its trace prefix and native rewards stay on disk.
            with exclusive(path / "worker.lock"):
                os.replace(result, path / f"infrastructure-{time_ns()}.json")
        with (path / "worker.log").open("ab") as log:
            # A per-seed lock in the worker outlives a killed campaign driver.
            proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode == 0:
            audit_run(root, study, stage, case)
            return
        if result.is_file() and read(result).get("infrastructure") is not True:
            raise RuntimeError(f"sealed model or consistency failure at {path}")
        if proc.returncode >= 0 and not result.is_file():
            raise RuntimeError(f"worker exited {proc.returncode} before writing {result}")
    raise RuntimeError(f"repeated infrastructure failure at {path}")


def run(root, workers, through="evaluation"):
    study = read(root / "study.json")
    verify_source(study)
    for stage in STAGES[:STAGES.index(through)+1]:
        if stage == "pre_evaluation":
            qualify(root, study)
        verify_before(root, study, stage)
        marker = root / MARKERS[stage]
        if marker.exists():
            if read(marker) != stage_receipt(root, study, stage):
                raise ValueError(f"stale {stage} receipt")
            continue
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda case: launch(root, study, stage, case), study["stages"][stage]))
        verify_source(study)
        publish(marker, stage_receipt(root, study, stage))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("seal", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path.home()/"agent-share/axiom-gameworld-reference")
    parser.add_argument("--through", choices=STAGES, default="evaluation",
                        help="Stop cleanly after publishing the selected stage receipt.")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    root = args.root.resolve()
    with exclusive(root / "campaign.lock"):
        if args.operation == "seal":
            seal(root, args.reference.resolve())
        else:
            run(root, args.workers, args.through)


if __name__ == "__main__":
    main()

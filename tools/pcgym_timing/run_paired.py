"""Run the frozen four-library comparison serially. Interrupted campaigns can resume."""
import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from time import time

from pilot import ARMS, HERE, ROOT, save


SEEDS = tuple(range(1001, 1031))
PHASES = ("short", "long", "forecast")


def read(path):
    return json.loads(path.read_text())


def signature(peer_root):
    files = [HERE / f"{arm}_adapter.py" for arm in ARMS]
    files += [HERE / name for name in ("pilot.py", "forecast.py", "qualification.py", "run_paired.py",
                                       "rxinfer_worker.jl", "gtsam_cstr.cpp")]
    files += [ROOT / name for name in ("uv.lock", "tools/cstr_reactor_pilot.py", "tools/cstr_runtime_fixture.py")]
    files += [peer_root / name for name in ("rxinfer/Manifest.toml", "rxinfer/Project.toml",
                                             "gtsam/build.lock", "gtsam/requirements.txt",
                                             "numpyro-arm.lock", "numpyro-extra.lock")]
    binary, = (peer_root / "gtsam/site").glob("_gtsam_cstr*.so")
    files.append(binary)
    lib_tree = subprocess.check_output(["git", "rev-parse", "HEAD:src/cffg"], cwd=ROOT, text=True).strip()
    return dict(library_tree=lib_tree, sha256={str(f): sha256(f.read_bytes()).hexdigest() for f in files})


def run_phase(folder, peer_root, arm, phase, seed=None):
    dest = folder / f"{arm}-{phase}"
    proc = dest / "process.json"
    result = dest / "result.json"
    if not proc.exists():
        cmd = [sys.executable, str(HERE / "pilot.py"), phase, "--output", str(folder),
               "--peer-root", str(peer_root), "--arm", arm]
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        subprocess.run(cmd, check=False)
    if not (result.exists() and proc.exists()):
        raise RuntimeError(f"Incomplete worker at {dest}. Inspect its worker.log before resuming.")
    out, process, manifest = read(result), read(proc), read(dest / "manifest.json")
    if (manifest["arm"], manifest["phase"], manifest["seed"]) != (arm, phase, seed):
        raise ValueError(f"Worker settings differ at {dest}")
    if not out.get("complete") or process["code"]:
        # A scored method failure stays in the paired data with its reason and denominator.
        if not (phase == "forecast" and out.get("status") == "failed" and out.get("n_failed", 0) > 0
                and len(out.get("origins", [])) == 12 and process["code"] == 1):
            raise RuntimeError(f"Worker failed at {dest}")
    return process["wall_s"]


def campaign(root, peer_root):
    protocol = dict(seeds=SEEDS, arms=ARMS, phases=PHASES, lengths=[59, 599], warm_deadline_s=1,
                    primary="599-decision summed return, CFFG minus each peer",
                    bootstrap="20000 paired seed resamples, percentile 95% pointwise intervals",
                    numerical_tie_abs_return=1e-8, truth_seed_offset=100000, draw_seed_offset=200000,
                    forecast=dict(origins=12, leads=[1, 5, 15, 60], particles=512),
                    order="rotate by seed index plus phase index; fresh process per arm and phase",
                    frozen=signature(peer_root))
    plan = root / "protocol.json"
    if plan.exists():
        if read(plan) != json.loads(json.dumps(protocol)):
            raise ValueError("Saved protocol or frozen sources differ; inspect before continuing.")
    else:
        save(plan, protocol)
    started_file = root / "started.json"
    if not started_file.exists():
        save(started_file, {"unix_time": time()})
    done = 0
    worker_s = 0.
    def progress(seed, phase, arm):
        save(root / "progress.json", dict(completed=done, total=len(SEEDS)*len(PHASES)*len(ARMS),
                                          latest_seed=seed, phase=phase, arm=arm, worker_s=worker_s,
                                          elapsed_s=time()-read(started_file)["unix_time"]))
    for arm in ARMS:
        run_phase(root / "qualification", peer_root, arm, "oracle")
    for i, seed in enumerate(SEEDS):
        folder = root / f"seed-{seed}"
        run_phase(folder, peer_root, "cffg", "truth", seed)
        for j, phase in enumerate(PHASES):
            offset = (i+j) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                worker_s += run_phase(folder, peer_root, arm, phase, seed)
                done += 1
                progress(seed, phase, arm)
    save(root / "finished.json", {"completed": done, "measured_worker_s": worker_s,
                                  "elapsed_s": time()-read(started_file)["unix_time"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peer-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    peers = args.peer_root.resolve()
    os.environ["PCGYM_RXINFER_PROJECT"] = str(peers / "rxinfer")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign(out, peers)

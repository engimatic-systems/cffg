"""Verify the measured workers used the qualified numerical stack and frozen study files."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from pilot import ARMS, HERE, save


def read(path):
    return json.loads(path.read_text())


def audit(root):
    plan = read(root / "protocol.json")
    finished = read(root / "finished.json")
    seeds = plan["seeds"]
    if finished["completed"] != len(seeds) * len(ARMS) * 3:
        raise ValueError("Incomplete campaign")
    frozen = {Path(p): digest for p, digest in plan["frozen"]["sha256"].items()}
    for path, digest in frozen.items():
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen component changed: {path}")
    sources = {p.name: digest for p, digest in frozen.items() if p.parent == HERE}
    qualified = {}
    workers = [(root/"qualification"/f"{arm}-oracle", arm, "oracle", None) for arm in ARMS]
    for seed in seeds:
        directory = root / f"seed-{seed}"
        workers.append((directory/"cffg-truth", "cffg", "truth", seed))
        workers.extend((directory/f"{arm}-{phase}", arm, phase, seed)
                       for phase in plan["phases"] for arm in ARMS)
    reference = read(workers[0][0]/"manifest.json")["runtime"]
    common = ("modules", "library_tree", "pcgym_source", "environment", "file_sha256",
              "hardware", "devices", "python_version", "reference_revision", "nested_checkout")
    if reference["library_tree"] != plan["frozen"]["library_tree"]:
        raise ValueError("CFFG library tree differs from the frozen plan")
    for p, digest in reference["file_sha256"].items():
        if sha256(Path(p).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Simulator or library dependency changed: {p}")
    for directory, arm, phase, seed in workers:
        meta = read(directory / "manifest.json")
        runtime = meta["runtime"]
        if (meta["arm"], meta["phase"], meta["seed"]) != (arm, phase, seed):
            raise ValueError(f"Misidentified worker at {directory}")
        if any(runtime[key] != reference[key] for key in common):
            raise ValueError(f"Runtime drift at {directory}")
        if any(meta["source_sha256"].get(name) != digest for name, digest in sources.items()):
            raise ValueError(f"Numerical source drift at {directory}")
        deps = meta["dependency_sha256"]
        binaries = meta.get("native_binary_sha256", {})
        if phase == "oracle":
            result = read(directory / "result.json")
            cases = result["cases"]
            if not result["complete"] or len(cases) != 3 or not all(c["passed"] for c in cases):
                raise ValueError(f"Unqualified arm {arm}")
            qualified[arm] = dict(cases=cases, deps=deps, binaries=binaries)
            expected = {"cffg": 0, "numpyro": 2, "gtsam": 2, "rxinfer": 2}
            if len(deps) != expected[arm] or (arm == "gtsam" and len(binaries) != 1):
                raise ValueError(f"Missing lock or native factor at {directory}")
            for suffix, digest in {**deps, **binaries}.items():
                if not any(p.as_posix().endswith("/" + suffix) and locked == digest for p, locked in frozen.items()):
                    raise ValueError(f"Unexpected dependency at {directory}: {suffix}")
        elif deps != qualified[arm]["deps"] or binaries != qualified[arm]["binaries"]:
            raise ValueError(f"Native dependency drift at {directory}")
    return dict(worker_manifests=len(workers), runtime={key: reference[key] for key in common},
                frozen_files=len(frozen), qualification=qualified, measured_workers=finished["completed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    root = parser.parse_args().output
    result = audit(root)
    save(root / "provenance.json", result)
    print(f"Verified {result['worker_manifests']} worker manifests against the qualified stack.")

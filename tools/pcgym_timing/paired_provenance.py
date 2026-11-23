"""Verify the measured workers used the qualified numerical stack and frozen study files."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from pilot import ARMS, ROOT, save


# This inventory belongs to the paired study's frozen run_paired.signature format.
HARNESS = Path("tools/pcgym_timing")
WORKER_SOURCES = (*(f"{arm}_adapter.py" for arm in ARMS), "pilot.py", "forecast.py", "qualification.py",
                  "run_paired.py", "rxinfer_worker.jl", "gtsam_cstr.cpp")
REPO_INPUTS = ("uv.lock", "tools/cstr_reactor_pilot.py", "tools/cstr_runtime_fixture.py")
REFERENCE = Path("tools/benchmark_cstr_chain.py")
EXPERIMENT_INPUTS = ("uv.lock", "common/envs.py", "common/plants.py", "common/physics.py",
                     "common/physics_agent.py", "cstr/experiment_physics.py",
                     "validation/v2_cstr_hostlin_baseline.py", "validation/v2_cstr_nonlinear_chain.py")
PEER_DEPS = {"cffg": (), "numpyro": ("numpyro-arm.lock", "numpyro-extra.lock"),
             "gtsam": ("gtsam/requirements.txt", "gtsam/build.lock"),
             "rxinfer": ("rxinfer/Project.toml", "rxinfer/Manifest.toml")}


def read(path):
    def unique_keys(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError(f"Duplicate JSON key in {path}")
        return result

    return json.loads(path.read_text(), object_pairs_hook=unique_keys)


def check_bytes(path, expected, role, revision):
    if not path.is_file():
        raise ValueError(f"Missing historical {role} file at {path} (recorded checkout {revision})")
    if sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Digest mismatch for historical {role} file at {path} (recorded checkout {revision})")


def frozen_inputs(frozen, reference, *, source_root, peer_root, experiment_root):
    """Resolve legacy absolute paths once, keeping checkout, peer and simulator identities distinct."""
    reference_path = Path(reference["reference_path"])
    if not reference_path.is_absolute() or reference_path.parts[-2:] != REFERENCE.parts:
        raise ValueError(f"Unexpected archived reference path: {reference_path}")
    recorded_repo = reference_path.parents[1]
    projects = [p for p in frozen if p.is_absolute() and p.parts[-2:] == ("rxinfer", "Project.toml")]
    if len(projects) != 1:
        raise ValueError("Expected exactly one frozen rxinfer/Project.toml to identify the peer directory")
    recorded_peers = projects[0].parents[1]
    binary_dir = recorded_peers / "gtsam/site"
    binaries = [p for p in frozen if p.parent == binary_dir and p.match("_gtsam_cstr*.so")]
    if len(binaries) != 1:
        raise ValueError("Expected exactly one frozen native GTSAM factor in gtsam/site")

    repo = Path(source_root) if source_root is not None else (recorded_repo if recorded_repo.is_dir() else ROOT)
    peers = Path(peer_root) if peer_root is not None else recorded_peers
    repo_files = [*(HARNESS / name for name in WORKER_SOURCES), *(Path(name) for name in REPO_INPUTS)]
    peer_files = [Path(name) for names in PEER_DEPS.values() for name in names]
    peer_files.append(binaries[0].relative_to(recorded_peers))
    expected = {recorded_repo / name for name in repo_files} | {recorded_peers / name for name in peer_files}
    if frozen.keys() != expected:
        missing = sorted(map(str, expected - frozen.keys()))
        unexpected = sorted(map(str, frozen.keys() - expected))
        raise ValueError(f"Frozen inventory mismatch: missing {missing}, unexpected {unexpected}")
    for old, local, files, role in ((recorded_repo, repo, repo_files, "checkout"),
                                    (recorded_peers, peers, peer_files, "peer")):
        for name in files:
            check_bytes(local / name, frozen[old / name], role, reference["head"])

    recorded_experiment = Path(reference["experiment"])
    if not recorded_experiment.is_absolute():
        raise ValueError(f"Expected an absolute archived experiment path: {recorded_experiment}")
    experiment = Path(experiment_root) if experiment_root is not None else recorded_experiment
    runtime = {Path(p): h for p, h in reference["file_sha256"].items()}
    if len(runtime) != len(reference["file_sha256"]):
        raise ValueError("Ambiguous runtime file identities")
    runtime_paths = {recorded_repo / name: repo / name for name in (*REPO_INPUTS, REFERENCE)}
    runtime_paths.update({recorded_experiment / name: experiment / name for name in EXPERIMENT_INPUTS})
    if len(runtime_paths) != len(REPO_INPUTS) + 1 + len(EXPERIMENT_INPUTS):
        raise ValueError("Archived checkout and experiment runtime file identities overlap")
    if missing := runtime_paths.keys() - runtime.keys():
        raise ValueError(f"Missing runtime file identities: {sorted(map(str, missing))}")
    for old, checksum in runtime.items():
        path = runtime_paths.get(old)
        if path is None:
            # load_runtime can additionally record the process launcher.
            if old.is_relative_to(recorded_repo):
                path = repo / old.relative_to(recorded_repo)
            elif old.is_relative_to(recorded_experiment):
                path = experiment / old.relative_to(recorded_experiment)
            else:
                path = old
        check_bytes(path, checksum, "runtime", reference["head"])

    sources = {name: frozen[recorded_repo / HARNESS / name] for name in WORKER_SOURCES}
    deps = {arm: {k: frozen[recorded_peers / k] for k in keys} for arm, keys in PEER_DEPS.items()}
    return sources, deps, {binaries[0].name: frozen[binaries[0]]}


def audit(root, *, source_root=None, peer_root=None, experiment_root=None):
    plan = read(root / "protocol.json")
    finished = read(root / "finished.json")
    seeds = plan["seeds"]
    if list(plan["phases"]) != ["short", "long", "forecast"] or finished["completed"] != len(seeds) * len(ARMS) * 3:
        raise ValueError("Incomplete campaign")
    frozen = {Path(p): digest for p, digest in plan["frozen"]["sha256"].items()}
    if len(frozen) != len(plan["frozen"]["sha256"]):
        raise ValueError("Ambiguous frozen file identities")
    qualified = {}
    workers = [(root/"qualification"/f"{arm}-oracle", arm, "oracle", None) for arm in ARMS]
    for seed in seeds:
        directory = root / f"seed-{seed}"
        workers.append((directory/"cffg-truth", "cffg", "truth", seed))
        workers.extend((directory/f"{arm}-{phase}", arm, phase, seed) for phase in plan["phases"] for arm in ARMS)
    reference = read(workers[0][0]/"manifest.json")["runtime"]
    common = ("modules", "library_tree", "pcgym_source", "environment", "file_sha256",
              "hardware", "devices", "python_version", "reference_revision", "nested_checkout")
    if reference["library_tree"] != plan["frozen"]["library_tree"]:
        raise ValueError("CFFG library tree differs from the frozen plan")
    sources, expected_deps, native = frozen_inputs(frozen, reference, source_root=source_root, peer_root=peer_root,
                                                   experiment_root=experiment_root)
    for directory, arm, phase, seed in workers:
        meta = read(directory / "manifest.json")
        runtime = meta["runtime"]
        if (meta["arm"], meta["phase"], meta["seed"]) != (arm, phase, seed):
            raise ValueError(f"Misidentified worker at {directory}")
        if any(runtime[key] != reference[key] for key in common):
            raise ValueError(f"Runtime drift at {directory}")
        worker_hashes = meta["source_sha256"]
        for name, digest in sources.items():
            if worker_hashes.get(name) != digest:
                raise ValueError(f"Missing or changed worker source {name} at {directory}")
        deps = meta["dependency_sha256"]
        binaries = meta.get("native_binary_sha256", {})
        if deps != expected_deps[arm] or binaries != (native if arm == "gtsam" else {}):
            raise ValueError(f"Native dependency mismatch at {directory}")
        if phase == "oracle":
            result = read(directory / "result.json")
            cases = result["cases"]
            if not result["complete"] or len(cases) != 3 or not all(c["passed"] for c in cases):
                raise ValueError(f"Unqualified arm {arm}")
            qualified[arm] = dict(cases=cases, deps=deps, binaries=binaries)
    return dict(worker_manifests=len(workers), runtime={key: reference[key] for key in common},
                frozen_files=len(frozen), runtime_files=len(reference["file_sha256"]),
                worker_source_files=len(sources),
                worker_sources={str(HARNESS / name): digest for name, digest in sources.items()},
                worker_source_checks=len(sources) * len(workers), qualification=qualified,
                measured_workers=finished["completed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Completed paired-study archive")
    parser.add_argument("--source-root", type=Path, help="Checkout or extracted tree containing the measured source bytes")
    parser.add_argument("--peer-root", type=Path, help="Directory containing the measured peer locks and native factor")
    parser.add_argument("--experiment-root", type=Path, help="External PCGym experiment directory used at measurement")
    parser.add_argument("--check-only", action="store_true", help="Verify all artifacts without writing provenance.json")
    args = parser.parse_args()
    result = audit(args.output, source_root=args.source_root, peer_root=args.peer_root,
                   experiment_root=args.experiment_root)
    if not args.check_only:
        save(args.output / "provenance.json", result)
    print(f"Verified {result['frozen_files']} frozen files, {result['runtime_files']} runtime files and "
          f"{result['worker_source_files']} sources in all {result['worker_manifests']} worker manifests "
          f"({result['worker_source_checks']} source hash comparisons).")

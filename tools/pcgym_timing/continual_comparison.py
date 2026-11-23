"""Compare four native planners receiving the same online estimate on a changing reactor."""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib
import importlib.metadata
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import mkdtemp
from time import perf_counter
from types import SimpleNamespace

from continual_run import (CHANGED, CONDITIONS, NOMINAL, STEPS, ReactorPlanner, digest, episode,
                           read, schedule, signatures)
from pilot import ARMS as LIBRARIES, DEADLINE, EXPERIMENT, HERE, ROOT, configure, parameters, save
from run_paired import signature

SEEDS = tuple(range(6601, 6611))
SETTINGS = {"frozen": None, "original": .012, "candidate": .003, "informed": None}
PRIOR = Path.home() / "agent-share/pcgym-one-percent"
PEERS = Path.home() / "agent-share/pcgym-four-library-timing"


def now():
    return datetime.now(timezone.utc).isoformat()


def sources(peers):
    pinned = signature(peers)
    files = dict(signatures())
    for name in ("one_percent_run.py", "continual_comparison.py", "run_paired.py"):
        files[str((HERE/name).relative_to(ROOT))] = digest(HERE/name)
    pinned["experiment_sha256"] = files
    return pinned


def freeze(out, peers, prior):
    seal = read(prior/"freeze.json")
    result = read(prior/"validation.json")
    study = seal["study"]
    if (not result["proceed"] or result["freeze"]["freeze_sha256"] != digest(prior/"freeze.json")
            or seal["selected"]["motion_sd"] != SETTINGS["candidate"]
            or study["original_motion_sd"] != SETTINGS["original"]
            or set(SEEDS).intersection(study["development_seeds"] + study["validation_seeds"])):
        raise ValueError("Prior learner selection or new seed split differs")
    plan = dict(ticket="cpcc-2dc3", libraries=list(LIBRARIES), seeds=list(SEEDS), conditions=list(CONDITIONS),
                arms=list(SETTINGS), motion_sd=SETTINGS, steps=STEPS, schedule=study["schedule"],
                scoring=study["scoring"], starting_method=study["starting_method"], decision_deadline_s=DEADLINE,
                primary="Mean paired candidate changed-period concentration RMSE difference in mol/L, CFFG minus peer.",
                intervals="Descriptive 95% pointwise paired whole-seed percentile bootstrap.",
                stationary_budget_pct=1, planner="20 sites: 4 history, current site, 15 future sites.",
                pairing="Same sensor noise draws, setpoints, dynamics and initial state for each seed. "
                        "Actions and measurement histories then depend on each planner.",
                methods="Same external log-UA EKF per learned condition, beginning at nominal UA. "
                        "Frozen retains nominal UA; informed receives this transition's true UA.",
                timing="Fresh process per library and seed; libraries rotate by seed and regimes by "
                       "seed and condition. One measured worker at a time. All episodes start fresh. "
                       "Warm decision includes estimator update, live parameter installation, window, "
                       "native solve, readout and action clipping. Hold last applied action on timeout or failure.",
                block_timing="Report process wall time only for a single measured launch completing all eight episodes. "
                             "Exclude a block with prior attempts or an unavailable process clock from timing totals.",
                prior_sha256={p: digest(prior/p) for p in ("freeze.json", "validation.json")},
                frozen=sources(peers))
    path = out/"protocol.json"
    if path.exists():
        previous = read(path)
        plan["source_revision"] = previous["source_revision"]
        if previous != plan:
            raise ValueError("Frozen protocol differs from current inputs")
    else:
        plan["source_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        save(path, plan)


def context(lib, peers):
    configure(SimpleNamespace(arm=lib, peer_root=peers))
    import cstr_reactor_pilot as reactor
    import cstr_runtime_fixture as fixture
    make_env, base, runtime = reactor.load_runtime(EXPERIMENT)
    return make_env, parameters(base, True), fixture.model()[1], fixture, reactor.window, runtime


def close(adapter):
    if hasattr(adapter, "close"):
        adapter.close()


def factory(lib):
    return ReactorPlanner if lib == "cffg" else importlib.import_module(lib + "_adapter").Adapter


def qualification(lib, data):
    import numpy as np
    from qualification import qualify
    from continual_plant import qualification as check_plant
    make_env, params, arrays, fixture, _, _ = data
    native = importlib.import_module(lib + "_adapter").Adapter
    affine = qualify(native, arrays)
    if not all(row["passed"] for row in affine):
        raise ValueError(f"{lib} failed affine oracle: {affine}")
    evidence = fixture.observations(history=True, goal=.19)
    planner = factory(lib)(arrays, fixture.CONFIGURATION)
    def action(adapter):
        row = adapter.solve(evidence)
        if row["status"] != "ok" or not np.isfinite(row["action"]):
            raise ValueError(f"Live-parameter probe failed: {row}")
        return float(row["action"])
    try:
        nominal = action(planner)
        planner.install(CHANGED)
        updated = action(planner)
        planner.install(NOMINAL)
        restored = action(planner)
    finally:
        close(planner)
    theta = arrays["theta"].copy()
    theta[2] = np.log(CHANGED)
    fresh = native({**arrays, "theta": theta}, fixture.CONFIGURATION)
    try:
        rebuilt = action(fresh)
    finally:
        close(fresh)
    if abs(updated-rebuilt) > 1e-8 or abs(nominal-restored) > 1e-8 or abs(nominal-updated) < 1e-4:
        raise ValueError(f"Parameter delivery mismatch: {nominal}, {updated}, {rebuilt}, {restored}")
    checks = dict(affine=affine, nominal=nominal, updated=updated, fresh=rebuilt, restored=restored)
    if lib == "cffg":
        checks["plant"] = check_plant(make_env, params, schedule("switching"))
    return checks


def valid_episode(stem, lib, seed, condition, arm, protocol_hash):
    summary_file = stem.with_suffix(".json")
    if not summary_file.exists():
        return False
    row = read(summary_file)
    if ((row.get("library"), row["seed"], row["condition"], row["arm"], row["steps"], row["motion_sd"])
            != (lib, seed, condition, arm, STEPS, SETTINGS[arm])
            or row.get("protocol_sha256") != protocol_hash
            or row["trace_sha256"] != digest(stem.with_suffix(".jsonl"))):
        raise ValueError(f"Saved episode differs from frozen measurement: {stem}")
    return True


def guard_open_campaign(out):
    if (out/"completed.json").exists():
        raise ValueError("Completed study is sealed; use continual_comparison_analysis.py --check to audit it")


def recover_inventory(out, lib, seed, ph, *, finished):
    """Validate published episodes before moving any recognizable interrupted files."""
    folder = out/lib/f"seed-{seed}"
    stems = {f"{c}-{a}" for c in CONDITIONS for a in SETTINGS}
    extras = []
    if folder.exists():
        for path in sorted(folder.iterdir()):
            if path.suffix not in (".jsonl", ".json", ".tmp"):
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Unexpected episode file: {path}")
            if path.stem in stems:
                if path.suffix == ".tmp":
                    extras.append(path)
            else:
                legacy = re.fullmatch(r"(.+)\.interrupted-[0-9]+(?:\.[0-9]+)?\.jsonl", path.name)
                if not legacy or legacy[1] not in stems:
                    raise ValueError(f"Unexpected episode file: {path}")
                extras.append(path)
    for condition in CONDITIONS:
        for arm in SETTINGS:
            stem = folder/f"{condition}-{arm}"
            if not valid_episode(stem, lib, seed, condition, arm, ph):
                if finished:
                    raise ValueError(f"Completed worker missing episode: {stem}")
                if stem.with_suffix(".jsonl").exists():
                    extras.append(stem.with_suffix(".jsonl"))
    if extras:
        archive = out/"attempts"/f"{lib}-{seed}"
        archive.mkdir(parents=True, exist_ok=True)
        dest = Path(mkdtemp(prefix="interrupted-", dir=archive))
        for path in extras:
            path.rename(dest/path.name)


def block(out, lib, seed, data, protocol_hash):
    from continual_estimator import warmup
    make_env, params, arrays, fixture, window, runtime = data
    attempts = out/"attempts"/f"{lib}-{seed}"
    staging = Path(mkdtemp(prefix="attempt-", dir=attempts))
    published = 0
    started = perf_counter()
    native_module = importlib.import_module(lib + "_adapter")
    planner = factory(lib)(arrays, fixture.CONFIGURATION)
    setup_s = perf_counter() - started
    try:
        started = perf_counter()
        warmup()
        for ua, history in ((51100., True), (NOMINAL, False)):
            planner.install(ua)
            proposal = planner.solve(fixture.observations(history=history, goal=.19))
            if proposal["status"] != "ok":
                raise ValueError(f"Initial warmup failed: {proposal}")
        first_warmup_s = perf_counter() - started
        for ci, condition in enumerate(CONDITIONS):
            arms = list(SETTINGS)
            offset = (SEEDS.index(seed) + ci) % len(arms)
            for arm in arms[offset:]+arms[:offset]:
                folder = out/lib
                stem = folder/f"seed-{seed}"/f"{condition}-{arm}"
                if valid_episode(stem, lib, seed, condition, arm, protocol_hash):
                    continue
                episode(make_env, params, arrays, fixture, window, seed, condition, arm, staging,
                        motion_sd=SETTINGS[arm], planner=planner,
                        provenance=dict(library=lib, protocol_sha256=protocol_hash))
                staged = staging/f"seed-{seed}"/stem.name
                if not valid_episode(staged, lib, seed, condition, arm, protocol_hash):
                    raise ValueError(f"Worker did not finish episode: {staged}")
                # The sidecar commits the trace. Recovery preserves an orphan if this pair is interrupted.
                stem.parent.mkdir(exist_ok=True)
                staged.with_suffix(".jsonl").rename(stem.with_suffix(".jsonl"))
                staged.with_suffix(".json").rename(stem.with_suffix(".json"))
                published += 1
        metadata = dict(getattr(planner, "metadata", {}))
        metadata["module"] = native_module.__file__
        if lib in ("numpyro","gtsam"):
            metadata["version"] = importlib.metadata.version(lib)
        return dict(library=lib, seed=seed, runtime=runtime, adapter_metadata=metadata,
                    adapter_setup_s=setup_s, first_warmup_s=first_warmup_s, episodes_run=published,
                    protocol_sha256=protocol_hash)
    finally:
        close(planner)


def worker(out, lib, seed, peers):
    guard_open_campaign(out)
    if seed is None:
        if any((out/f).exists() for f in ("qualified.json", "opened.json", f"qualification/{lib}.json")):
            raise ValueError("Qualification evidence already exists or is sealed")
    elif (out/"workers"/f"{lib}-{seed}.json").exists():
        raise ValueError("Worker record already exists")
    plan = read(out/"protocol.json")
    if plan["frozen"] != sources(peers):
        raise ValueError("Worker sources differ from frozen protocol")
    ph = digest(out/"protocol.json")
    if seed is not None:
        qualified = read(out/"qualified.json")
        opened = read(out/"opened.json")
        if (seed not in SEEDS or qualified["protocol_sha256"] != ph or opened["protocol_sha256"] != ph
                or opened["qualified_sha256"] != digest(out/"qualified.json")
                or qualified["evidence"][lib] != digest(out/"qualification"/f"{lib}.json")):
            raise ValueError("Missing qualification or unexpected seed")
        # A directly invoked worker can also fail before producing any episode.
        (out/"attempts"/f"{lib}-{seed}").mkdir(parents=True, exist_ok=True)
        recover_inventory(out, lib, seed, ph, finished=False)
    data = context(lib, peers)
    if data[-1]["library_tree"] != plan["frozen"]["library_tree"]:
        raise ValueError("CFFG runtime differs from frozen checkout")
    if seed is None:
        save(out/"qualification"/f"{lib}.json", dict(library=lib, protocol_sha256=ph, checks=qualification(lib, data)))
    else:
        record = block(out, lib, seed, data, ph)
        save(out/"workers"/f"{lib}-{seed}.json", record)


def run_child(out, lib, peers, lock_fd, seed=None):
    name = f"{lib}-{seed}" if seed is not None else f"qualify-{lib}"
    cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(out),
           "--library", lib, "--peer-root", str(peers), "--lock-fd", str(lock_fd)]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    start = perf_counter()
    with (out/"logs"/f"{name}.log").open("a") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, pass_fds=(lock_fd,))
    return perf_counter()-start


def measure_block(out, lib, seed, peers, ph, lock_fd):
    rec = out/"workers"/f"{lib}-{seed}.json"
    attempts = out/"attempts"/f"{lib}-{seed}"
    folder = out/lib/f"seed-{seed}"
    previous = (attempts.exists() or (out/"logs"/f"{lib}-{seed}.log").exists()
                or (folder.exists() and any(folder.iterdir())))
    elapsed = None
    if not rec.exists():
        # Persist launch intent before any worker code, including imports and setup.
        attempts.mkdir(parents=True, exist_ok=True)
        elapsed = run_child(out, lib, peers, lock_fd, seed)
    record = read(rec)
    if (record["library"], record["seed"], record["protocol_sha256"]) != (lib, seed, ph):
        raise ValueError(f"Worker identity mismatch: {rec}")
    recover_inventory(out, lib, seed, ph, finished=True)
    interrupted = any(attempts.glob("interrupted-*"))
    if "process_wall_s" not in record or (interrupted and record["process_wall_s"] is not None):
        # A worker may have committed all eight episodes before its parent recorded the clock.
        full = (elapsed is not None and not previous and not interrupted
                and record["episodes_run"] == len(CONDITIONS)*len(SETTINGS))
        record["process_wall_s"] = elapsed if full else None
        save(rec, record)
    return record


def campaign(out, peers, prior, lock_fd):
    guard_open_campaign(out)
    measured = ((out/"attempts").exists() or any((out/"workers").glob("*.json"))
                or any(folder for lib in LIBRARIES for folder in (out/lib).glob("seed-*"))
                or any(log for lib in LIBRARIES for log in (out/"logs").glob(f"{lib}-*.log")))
    if measured and not all((out/f).exists() for f in ("qualified.json", "opened.json")):
        raise ValueError("Missing qualification or opening seal for measured study")
    if not (out/"protocol.json").exists() and (any((out/"qualification").glob("*.json"))
            or (out/"qualified.json").exists() or (out/"opened.json").exists()):
        raise ValueError("Missing sealed protocol")
    freeze(out, peers, prior)
    for folder in ("qualification", "workers", "logs", *LIBRARIES):
        (out/folder).mkdir(exist_ok=True)
    ph = digest(out/"protocol.json")
    for lib in LIBRARIES:
        path = out/"qualification"/f"{lib}.json"
        if not path.exists():
            if (out/"qualified.json").exists() or (out/"opened.json").exists():
                raise ValueError("Missing sealed qualification evidence")
            print("Qualifying", lib, flush=True)
            run_child(out, lib, peers, lock_fd)
        if read(path)["protocol_sha256"] != ph:
            raise ValueError("Saved qualification is for a different protocol")
    qualified = dict(protocol_sha256=ph, evidence={lib:digest(out/"qualification"/f"{lib}.json") for lib in LIBRARIES})
    if (out/"qualified.json").exists() and read(out/"qualified.json") != qualified:
        raise ValueError("Qualification evidence changed")
    if not (out/"qualified.json").exists():
        if (out/"opened.json").exists():
            raise ValueError("Missing qualification seal for opened study")
        save(out/"qualified.json", qualified)
    opened = out/"opened.json"
    if not opened.exists():
        save(opened, dict(opened=now(), protocol_sha256=ph, qualified_sha256=digest(out/"qualified.json")))
    if read(opened)["qualified_sha256"] != digest(out/"qualified.json") or read(opened)["protocol_sha256"] != ph:
        raise ValueError("Opened study differs from current qualification")
    block_size = len(CONDITIONS)*len(SETTINGS)
    total = block_size*len(LIBRARIES)*len(SEEDS)
    blocks = {lib:[] for lib in LIBRARIES}
    for i, seed in enumerate(SEEDS):
        offset = i % len(LIBRARIES)
        for lib in LIBRARIES[offset:]+LIBRARIES[:offset]:
            record = measure_block(out, lib, seed, peers, ph, lock_fd)
            blocks[lib].append(record)
            count = block_size*sum(map(len, blocks.values()))
            save(out/"progress.json", dict(episodes=count, total=total, latest_library=lib, latest_seed=seed))
            print(seed, lib, count, "/", total, "episodes", flush=True)
    for lib, records in blocks.items():
        save(out/"workers"/f"{lib}.json", dict(library=lib, blocks=records))
    walls = [r["process_wall_s"] for records in blocks.values() for r in records]
    save(out/"completed.json", dict(protocol_sha256=ph, qualified_sha256=digest(out/"qualified.json"),
         episodes=total, decisions=STEPS*total, finished=now(), opened=read(opened)["opened"],
         worker_s=sum(walls) if None not in walls else None))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("run", "worker"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--peer-root", type=Path, default=PEERS)
    p.add_argument("--prior", type=Path, default=PRIOR)
    p.add_argument("--library", choices=LIBRARIES)
    p.add_argument("--seed", type=int)
    p.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    root, peer_root = args.output.resolve(), args.peer_root.resolve()
    os.environ.update(PCGYM_RXINFER_PROJECT=str(peer_root/"rxinfer"),
                      XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    root.mkdir(parents=True, exist_ok=True)
    # A surviving child keeps its parent's lock until it finishes publishing.
    guard_open_campaign(root)
    lock_path = root/"study.lock"
    with (lock_path.open("a") if args.lock_fd is None else os.fdopen(args.lock_fd, "a")) as lock:
        if not os.path.samestat(os.fstat(lock.fileno()), lock_path.stat()):
            raise ValueError("Worker lock does not match study")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == "worker":
            worker(root, args.library, args.seed, peer_root)
        else:
            campaign(root, peer_root, args.prior.resolve(), lock.fileno())

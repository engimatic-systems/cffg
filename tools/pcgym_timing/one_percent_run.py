"""Develop a slower reactor learner, seal its rule, then open fresh paired seeds."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

from continual_run import (CONDITIONS, SEEDS as DEVELOPMENT_SEEDS, NOMINAL, STEPS, TAPE_SEED,
                           digest, episode, read, schedule, signatures, write_row)
from pilot import EXPERIMENT, HERE, ROOT, configure, parameters, save


CANDIDATES = {"q0100": .010, "q0060": .006, "q0030": .003, "q0015": .0015}
ORIGINAL = .012
FRESH_SEEDS = tuple(range(6501, 6511))
FRESH_TAPE = 6500
ARMS = ("frozen", "original", "candidate", "informed")
ARCHIVE = Path.home() / "agent-share/pcgym-continual-learning"


def now():
    return datetime.now(timezone.utc).isoformat()


def revision(ref):
    return subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip()


def numerical_sources():
    return {**signatures(), str(Path(__file__).resolve().relative_to(ROOT)): digest(Path(__file__))}


def check_protocol(root):
    study = read(root / "protocol.json")
    if study["source_sha256"] != numerical_sources() or study["library_tree"] != revision("HEAD:src/cffg"):
        raise ValueError("Frozen numerical sources or library differ from the study protocol")
    return study


def init_protocol(root, archive):
    if (root / "protocol.json").exists():
        check_protocol(root)
        return
    previous = read(archive / "study.json")
    assert previous["library_tree"] == revision("HEAD:src/cffg")
    archive_hashes = {file: digest(archive/file) for file in ("study.json", "analysis.json", "qualified.json",
                                                               "completed.json", "tape-switching.jsonl")}
    if read(archive/"completed.json")["study_sha256"] != archive_hashes["study.json"]:
        raise ValueError("Pilot archive is incomplete")
    save(root / "protocol.json", dict(started=now(), revision=revision("HEAD"),
         library_tree=previous["library_tree"], source_sha256=numerical_sources(),
         archive=str(archive), archive_sha256=archive_hashes, starting_method=previous["method"],
         development_seeds=DEVELOPMENT_SEEDS, development_tape=TAPE_SEED, candidates=CANDIDATES,
         validation_seeds=FRESH_SEEDS, validation_tape=FRESH_TAPE, original_motion_sd=ORIGINAL,
         controls=ARMS, primary_stationary_limit_pct=1., sensitivity_limits_pct=[2., 3.],
         recovery_guard="median paired candidate minus original delay <=5 ticks at each change; all candidate recoveries reached",
         retention="candidate median paired tracking gain versus frozen >=90% of original median paired gain",
         pilot_gates="positive wins on >=80% of seeds and median tracking gain >=1% for candidate and informed; "
                    "settled UA MAE reduction >=50%; changed-tape sensor T RMSE < frozen; no held commands",
         ranking="fewest failed primary gates, least excess stationary penalty over 1%, largest motion SD",
         fallback="validate the first ranked development candidate even if none pass; never refit on fresh seeds",
         conditions=CONDITIONS, schedule=previous["ua"], scoring=previous["scoring"],
         decision_deadline_s=previous["warm_deadline_s"]))


def context():
    configure(SimpleNamespace(arm="cffg"))
    import cstr_reactor_pilot as reactor
    import cstr_runtime_fixture as fixture
    make_env, base, runtime = reactor.load_runtime(EXPERIMENT)
    return make_env, parameters(base, True), fixture.model()[1], fixture, reactor.window, runtime


def tape(out, q, *, archive=None, make_env=None, params=None):
    """Issue predictions before reading the successor, then update both independent learners once."""
    import numpy as np
    from continual_estimator import Learner, predict
    from continual_plant import make_plant
    from forecast import _commands
    out.mkdir(parents=True, exist_ok=True)
    file = out/"tape-switching.jsonl"
    source = archive/"tape-switching.jsonl" if archive is not None else None
    rows = [json.loads(line) for line in source.read_text().splitlines()] if source else None
    coefficients = schedule("switching")
    commands = _commands(STEPS)
    learners = {"original": Learner(), "candidate": Learner(motion_sd=q)}
    plant = None if rows is not None else make_plant(make_env, params, coefficients, FRESH_TAPE)
    tmp = file.with_suffix(".partial")
    try:
        obs = np.asarray(rows[0]["obs"]) if rows is not None else plant.obs.copy()
        with tmp.open("w") as stream:
            for tick, command in enumerate(commands):
                ua = dict(frozen=NOMINAL, informed=float(coefficients[tick]),
                          **{name: obj.ua for name, obj in learners.items()})
                predictions = {a: predict(obs, command, value) for a, value in ua.items()}
                if rows is not None:
                    r = rows[tick]
                    if (r["tick"] != tick or r["applied"] != command or r["true_ua"] != ua["informed"]
                            or not np.array_equal(obs, r["obs"])
                            or abs(r["estimates"]["learned"]-ua["original"]) > 1e-7
                            or not np.allclose(r["predictions"]["learned"], predictions["original"], atol=1e-12, rtol=0)):
                        raise ValueError("Original learner disagrees with archived development tape")
                    nxt, truth, reward, target = np.asarray(r["next_obs"]), r["truth"], r["reward"], r["target"]
                else:
                    nxt, truth, reward, done = plant.step(float(command))
                    target = params["SP"]["Ca"][tick+1]
                    if done != (tick == STEPS-1):
                        raise ValueError("Unexpected validation tape termination")
                updates = {name: obj.update(obs, command, nxt) for name, obj in learners.items()}
                write_row(stream, dict(tick=tick, obs=obs, next_obs=nxt, truth=truth, applied=float(command),
                          reward=reward, target=target, true_ua=coefficients[tick], estimates=ua,
                          predictions=predictions, updates=updates))
                obs = nxt
        tmp.replace(file)
    finally:
        if plant is not None:
            plant.close()


def completed(out, seed, condition, arm, q):
    stem = out/f"seed-{seed}"/f"{condition}-{arm}"
    if not stem.with_suffix(".json").exists():
        return False
    record = read(stem.with_suffix(".json"))
    if ((record["seed"], record["condition"], record["arm"], record["motion_sd"]) != (seed, condition, arm, q)
            or record["steps"] != STEPS or record["trace_sha256"] != digest(stem.with_suffix(".jsonl"))):
        raise ValueError(f"Completed episode integrity failure: {stem}")
    return True


def seal(root):
    study = check_protocol(root)
    archive = Path(study["archive"])
    for file, expected in study["archive_sha256"].items():
        if digest(archive/file) != expected:
            raise ValueError("Original development archive changed")
    if (root/"freeze.json").exists():
        return verify_seal(root)
    if (root/"validation").exists():
        raise ValueError("Fresh outputs existed before candidate freeze")
    from one_percent_analysis import development
    candidates = [development(root, archive, label, q) for label, q in CANDIDATES.items()]
    ordered = sorted(candidates, key=lambda x: (x["failed_gate_count"], max(0., x["stationary_penalty_pct"]-1), -x["q"]))
    save(root/"development.json", dict(candidates=candidates, ranking=[r["label"] for r in ordered]))
    winner = ordered[0]
    analysis_sources = {p: digest(HERE/p) for p in ("one_percent_analysis.py", "continual_analysis.py")}
    save(root/"freeze.json", dict(frozen=now(), revision=revision("HEAD"), study=study,
         protocol_sha256=digest(root/"protocol.json"), development_sha256=digest(root/"development.json"),
         analysis_sha256=analysis_sources, selected=dict(label=winner["label"], motion_sd=winner["q"]),
         development_passed=winner["proceed"], validation_started=False))
    print("Frozen", winner["label"], "development passed", winner["proceed"], flush=True)


def verify_seal(root):
    check_protocol(root)
    frozen = read(root/"freeze.json")
    if (frozen["protocol_sha256"] != digest(root/"protocol.json")
            or frozen["development_sha256"] != digest(root/"development.json")
            or any(digest(HERE/p) != h for p,h in frozen["analysis_sha256"].items())):
        raise ValueError("Selection, analysis or protocol changed after freeze")
    return frozen


def block(root, stage, seed, condition):
    study = check_protocol(root)
    validate = stage == "validation"
    sealed = verify_seal(root) if validate else None
    seeds = FRESH_SEEDS if validate else DEVELOPMENT_SEEDS
    if seed not in seeds or condition not in CONDITIONS:
        raise ValueError("Seed or condition outside stage")
    if not validate and (root/"freeze.json").exists():
        raise ValueError("Development is sealed")
    make_env, params, arrays, fixture, window, runtime = context()
    offset = (seeds.index(seed)+CONDITIONS.index(condition)) % 4
    if validate:
        folder = root/"validation"
        settings = {"frozen": None, "original": ORIGINAL, "candidate": sealed["selected"]["motion_sd"], "informed": None}
        jobs = [(folder, a, settings[a]) for a in ARMS]
    else:
        jobs = [(root/"dev"/label, "candidate", q) for label,q in CANDIDATES.items()]
    for folder, arm, q in jobs[offset:]+jobs[:offset]:
        if not completed(folder, seed, condition, arm, q):
            episode(make_env, params, arrays, fixture, window, seed, condition, arm, folder, motion_sd=q)
        print(stage, seed, condition, arm, q, flush=True)
    meta = root/stage/"workers"
    meta.mkdir(exist_ok=True, parents=True)
    if runtime["library_tree"] != study["library_tree"]:
        raise ValueError("Unexpected library in worker")
    save(meta/f"{seed}-{condition}.json", runtime)


def campaign(root, stage):
    study = check_protocol(root)
    is_validation = stage == "validation"
    if is_validation:
        sealed = verify_seal(root)
        folder = root/"validation"
        folder.mkdir(exist_ok=True)
        marker = folder/"opened.json"
        if not marker.exists():
            save(marker, dict(opened=now(), freeze_sha256=digest(root/"freeze.json")))
        if read(marker)["freeze_sha256"] != digest(root/"freeze.json"):
            raise ValueError("Validation belongs to a different freeze")
    else:
        if (root/"freeze.json").exists():
            raise ValueError("Development is sealed")
    clock = perf_counter()
    make_env, params, *_ = context()
    if is_validation:
        tape(folder, sealed["selected"]["motion_sd"], make_env=make_env, params=params)
    else:
        for label,q in CANDIDATES.items():
            tape(root/"dev"/label, q, archive=Path(study["archive"]))
    seeds = FRESH_SEEDS if is_validation else DEVELOPMENT_SEEDS
    for seed in seeds:
        for condition in CONDITIONS:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--root", str(root),
                            "--stage", stage, "--seed", str(seed), "--condition", condition], check=True)
    save(root/f"{stage}-completed.json", dict(episode_count=8*len(seeds), finished=now(),
                                              elapsed_s=perf_counter()-clock))


@contextmanager
def exclusive(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root/"study.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("development", "freeze", "validation", "worker"))
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--archive", default=ARCHIVE, type=Path)
    p.add_argument("--stage", choices=("development", "validation"))
    p.add_argument("--seed", type=int)
    p.add_argument("--condition", choices=CONDITIONS)
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    if args.phase == "worker":
        return block(root, args.stage, args.seed, args.condition)
    with exclusive(root):
        if args.phase == "freeze":
            return seal(root)
        if args.phase == "development":
            init_protocol(root, args.archive.expanduser().resolve())
        campaign(root, args.phase)


if __name__ == "__main__":
    main()

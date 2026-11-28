"""Compare original exact and public exact/compact I/O on the two retained Explode snapshots."""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from time import perf_counter

import jax
import numpy as np

from cffg.axiom_agent import AxiomAgent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pinned(checkout, paths):
    head = subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"], text=True).strip()
    hashes = {}
    for name in paths:
        actual = (checkout / name).read_bytes()
        if actual != subprocess.check_output(["git", "-C", checkout, "show", f"{head}:{name}"]):
            raise ValueError(f"worktree differs from measured source at {name}")
        hashes[name] = sha(actual)
    return {"git": head, "files": hashes}


def read_inputs(study, tape, tick):
    run = study / "development/Explode-88101"
    result_bytes = (run / "result.json").read_bytes()
    result = json.loads(result_bytes)
    marker = json.loads((study / "pilot_passed.json").read_bytes())
    if (sha((study / "study.json").read_bytes()) != "b42e006cfe7c88c2f1310b7ca2eb8fe953aeacb7000dd5f5f6819dcdee23041a"
            or marker["runs"]["Explode-88101"] != sha(result_bytes)
            or marker["source_sha256"] != result["source_sha256"]):
        raise ValueError("pilot evidence differs from the recorded study or completed run")
    with np.load(tape, allow_pickle=False) as archive:
        pixels = archive["pixels"][0]
    reset = json.loads((run / "initial.json").read_bytes())
    if sha(pixels.tobytes()) != reset["reset"]["rgb_sha256"]:
        raise ValueError("the archived reset pixels differ from the pilot")
    with (run / "steps.jsonl").open("rb") as stream:
        prefix = hashlib.sha256()
        for i in range(tick):
            line = stream.readline()
            if json.loads(line)["tick"] != i:
                raise ValueError("noncontiguous original action tape")
            prefix.update(line)
        next_row = stream.readline()
    metadata = {k: result[k] for k in ("stage", "game", "model_seed", "game_seed", "goal_steps", "source_sha256")}
    metadata.update(tick=tick, trace_sha256=prefix.hexdigest())
    path = run / "checkpoints" / f"{tick}.npz"
    if tick == 9500 and sha(path.read_bytes()) != "49212f0f0b8e2672e3ad5e2d4180635470ea7519067f3d23f9d334e88a51b1dc":
        raise ValueError("action-9500 snapshot differs from the original archive")
    if tick == result["steps"] and (sha(path.read_bytes()) != result["final_checkpoint_sha256"]
                                    or prefix.hexdigest() != result["steps_sha256"]):
        raise ValueError("final pilot receipt differs")
    next_action = json.loads(next_row)["action"] if next_row else None
    return pixels, metadata, path, next_action


def compare_state(actor, original, mode):
    state = actor.state_dict()
    if state.keys() != original.keys():
        raise AssertionError("restored state inventory differs")
    nonzero = 0
    for name, expected in original.items():
        actual = state[name]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise AssertionError(f"state layout differs at {name}")
        if mode == "compact" and ".posterior.carry." in name:
            nonzero += int(np.count_nonzero(expected))
            if np.any(actual):
                raise AssertionError(f"uncleared compact remainder at {name}")
        elif actual.tobytes() != expected.tobytes():
            raise AssertionError(f"state bytes differ at {name}")
    return nonzero


def worker(args, tick, work):
    # Import the original, unmodified actor/codec from an independently pinned checkout.
    sys.path.insert(0, str(args.baseline / "tools/axiom"))
    import evaluation_actor as old_actor
    import evaluation_checkpoint as old_codec

    pixels, meta, source, next_action = read_inputs(args.study, args.reset_tape, tick)
    constructors = {"legacy": old_actor.Actor, "exact": AxiomAgent, "compact": AxiomAgent}
    actors = {name: cls(pixels, "Explode", 3, 88101, audit_plan=True) for name, cls in constructors.items()}
    old_codec.restore(source, actors["legacy"], meta)
    original = actors["legacy"].state_dict()
    for name in ("exact", "compact"):
        actors[name].restore(source, meta)
        compare_state(actors[name], original, "exact")
    paths = {name: work / f"{tick}-{name}.npz" for name in actors}

    def save(name):
        if name == "legacy":
            old_codec.save(paths[name], actors[name], meta)
        else:
            actors[name].checkpoint(paths[name], meta, exact=name == "exact")

    def load(name):
        if name == "legacy":
            old_codec.restore(paths[name], actors[name], meta)
        else:
            actors[name].restore(paths[name], meta)
        jax.block_until_ready((actors[name].scene, actors[name].key, actors[name].proposal))

    for name in actors:
        save(name)
        load(name)
    if sha(paths["legacy"].read_bytes()) != sha(source.read_bytes()):
        raise AssertionError("legacy writer did not reproduce the original file")
    sizes = {name: path.stat().st_size for name, path in paths.items()}
    rows = []
    # Reinstall the same exact pilot state before every timed write. This also
    # restores the low words dropped by the previous compact round trip.
    for order in itertools.permutations(actors):
        row = {}
        for name in order:
            if name == "legacy":
                old_codec.restore(source, actors[name], meta)
            else:
                actors[name].restore(source, meta)
            t0 = perf_counter()
            save(name)
            t1 = perf_counter()
            load(name)
            t2 = perf_counter()
            row[name] = {"write_s": t1-t0, "restore_s": t2-t1}
        rows.append(row)
    modes = {}
    for name, actor in actors.items():
        omitted = compare_state(actor, original, name)
        timing = {}
        for field in ("restore_s", "write_s"):
            values = [row[name][field] for row in rows]
            ratios = [row[name][field]/row["legacy"][field] for row in rows]
            timing[field] = dict(median=float(np.median(values)), minimum=min(values), maximum=max(values),
                                 paired_median_change_pct=float(100*(np.median(ratios)-1)))
        modes[name] = dict(bytes=sizes[name], byte_change_pct=100*(sizes[name]/sizes["legacy"]-1),
                           omitted_nonzero_low_words=omitted, timing=timing)
        if next_action is not None:
            action = actor.choose()
            modes[name].update(next_action=action, plan_sha256=actor.last_decision["plan_sha256"])
            if name != "compact" and action != next_action:
                raise AssertionError("exact continuation disagrees with recorded next action")
    if next_action is not None and modes["legacy"]["plan_sha256"] != modes["exact"]["plan_sha256"]:
        raise AssertionError("exact plan arrays or next key differ")
    mask = original["scene[0].interaction.posterior.active"]
    return dict(tick=tick, active=int(mask.sum()), capacity=len(mask), recorded_next_action=next_action,
                original_sha256=sha(source.read_bytes()), trace_sha256=meta["trace_sha256"],
                reset_sha256=sha(pixels.tobytes()), modes=modes, blocks=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--reset-tape", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True, help="original checkout with evaluation tools")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tick", type=int, choices=(9500, 10000), help=argparse.SUPPRESS)
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        if args.tick is not None:
            result = worker(args, args.tick, work)
        else:
            checkout = Path(__file__).resolve().parents[2]
            old_paths = [f"tools/axiom/evaluation_{name}.py" for name in ("actor","checkpoint","planning","reduction")]
            new_paths = ["src/cffg/axiom_agent.py", "src/cffg/axiom_checkpoint.py", "tools/axiom/checkpoint_cost.py",
                         "src/cffg/experiments/axiom_cached_plan.py", "src/cffg/experiments/axiom_reduction_schedule.py"]
            result = dict(passed=True, platform=platform.platform(), python=sys.version, jax=jax.__version__,
                          numpy=np.__version__, device=str(jax.devices("cpu")[0]), clock="perf_counter",
                          scope="Six balanced warmed orders. Wall includes codec validation and disk I/O. "
                                "Fresh interpreter per snapshot. Setup and the exact pre-write reload are excluded. "
                                "Read-only next-action checks occur after timing. No simulator steps.",
                          baseline=pinned(args.baseline, old_paths), candidate=pinned(checkout, new_paths), cases=[])
            for tick in (9500, 10000):
                receipt = work / f"{tick}.json"
                subprocess.run([sys.executable, __file__, "--study", str(args.study), "--reset-tape", str(args.reset_tape),
                                "--baseline", str(args.baseline), "--tick", str(tick), "--output", str(receipt)],
                               check=True, env={**os.environ, "JAX_ENABLE_X64": "0", "JAX_PLATFORMS": "cpu"})
                result["cases"].append(json.loads(receipt.read_bytes()))
            if result["baseline"] != pinned(args.baseline, old_paths) or result["candidate"] != pinned(checkout, new_paths):
                raise ValueError("measured source changed during the comparison")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(f.name, args.output)


if __name__ == "__main__":
    main()

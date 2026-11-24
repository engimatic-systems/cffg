"""Run one sealed seed with native Gameworld inputs and an exactly restartable CFFG actor."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np

from evaluation_actor import Actor
from evaluation_campaign import (STAGES, canonical, digest, exclusive, publish, read,
                                 run_name, verify_before, verify_source)
from evaluation_checkpoint import save, restore
from evaluation_game import simulator


def actor_digest(actor):
    h = hashlib.sha256()
    for name, value in sorted(actor.state_dict().items()):
        a = np.ascontiguousarray(value)
        spec = canonical([name, a.dtype.str, a.shape])
        h.update(len(spec).to_bytes(8, "big"))
        h.update(spec)
        h.update(memoryview(a).cast("B") if a.size else b"")
    return h.hexdigest()


def existing_rows(path):
    rows, prefix = [], hashlib.sha256()
    if path.exists():
        with path.open("rb+") as stream:
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    # Only a killed write can leave an unterminated tail; keep its exact bytes.
                    offset = stream.tell() - len(line)
                    (path.parent / f"partial-line-{offset}.bin").write_bytes(line)
                    stream.truncate(offset)
                    break
                row = json.loads(line)
                if type(row["tick"]) is not int or row["tick"] != len(rows):
                    raise ValueError("noncontiguous existing action log")
                prefix.update(line)
                rows.append((row, prefix.hexdigest()))
    return rows, prefix


def compare_replay(actual, archived):
    keys = ["tick", "action", "game", "reset", "reward", "reduction_performed", "reduction",
            "frame_sha256", "active_interaction", "active_motion", "tracked", "state_sha256", "plan_sha256"]
    for k in keys:
        if actual.get(k) != archived.get(k):
            raise ArithmeticError(f"replay differs at action {actual['tick']}, field {k}")


def episode(root, study, stage, case):
    dest = root / stage / run_name(case)
    initial_file = dest / "initial.json"
    trace_file = dest / "steps.jsonl"
    checkpoints = dest / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    rows, prefix = existing_rows(trace_file)
    budget = case["steps"]
    identity = dict(source_sha256=study["source_sha256"], stage=stage, game=case["game"],
                    model_seed=case["model_seed"], game_seed=case["game_seed"], goal_steps=budget)

    def meta(t):
        return dict(identity, tick=t, trace_sha256=rows[t-1][1] if t else hashlib.sha256().hexdigest())

    def initialize(request, header):
        image, initial = request("reset")
        receipt = dict(header=header, reset=initial)
        if initial_file.exists():
            if read(initial_file) != receipt:
                raise ValueError("initial simulator identity or pixels differ on restart")
        else:
            publish(initial_file, receipt)
        actor = Actor(image, case["game"], header["actions"], case["model_seed"])
        actor.audit_plan = stage == "pre_evaluation"
        return actor

    def restore_at(actor, request, t):
        for archived, _ in rows[:t]:
            _, native = request({"tick": archived["tick"], "action": archived["action"]})
            if native != archived["game"]:
                raise ValueError(f"native replay differs at action {archived['tick']}")
            if archived["reset"] is not None:
                _, reset = request("reset")
                if reset != archived["reset"]:
                    raise ValueError(f"native reset differs at action {archived['tick']}")
        restore(checkpoints / f"{t}.npz", actor, meta(t))

    pending = {}
    def transition(actor, request, *, full_digest):
        tick = actor.tick
        start = perf_counter()
        action = actor.choose()
        decision_s = perf_counter() - start
        image, game = request({"action": action, "tick": tick})
        pending.update(tick=tick, action=action, game=game)
        if game["applied"] != action or game["tick"] != tick:
            raise ArithmeticError("native command receipt differs from planned action")
        reward = game["reward"]
        t_obs = perf_counter()
        observed = actor.observe(image, action, reward)
        reduced = observed["reduction_performed"]
        if (not observed["valid"] or actor.tick != tick + 1 or reduced != ((tick+1) % 500 == 0)):
            raise ArithmeticError("model observation or scheduled reduction failed")
        reduction = observed["reduction"]
        if reduced:
            if reduction["step"] != tick+1 or reduction["status"] != "ok":
                raise ArithmeticError("invalid reduction receipt")
            records = reduction.get("decisions", [])
            reduction = {k:v for k,v in reduction.items() if k != "decisions"}
            reduction["decisions_sha256"] = hashlib.sha256(canonical(records)).hexdigest()
        reset = None
        if game["terminated"] or game["truncated"]:
            frame, reset = request("reset")
            pending["reset"] = reset
            update = actor.observe(frame, 0, 0., reset=True)
            if (not update["valid"] or actor.tick != tick+1 or update["update"]
                    or update["identity_writes"] or update["interaction_writes"] or update["reduction_performed"]):
                raise ArithmeticError("a terminal reset lost the real action clock or wrote evidence")
        observation_s = perf_counter() - t_obs
        row = dict(tick=tick, action=action, reward=reward, game=game, reset=reset,
                   terminated=game["terminated"], truncated=game["truncated"], frame_sha256=game["rgb_sha256"],
                   observation_s=observation_s, decision_s=decision_s, decision=actor.last_decision,
                   valid=True, reduction_performed=reduced, reduction=reduction,
                   active_interaction=observed["active_interaction"], active_motion=observed["active_motion"],
                   tracked=observed["tracked"])
        if full_digest:
            row["plan_sha256"] = actor.last_decision["plan_sha256"]
            row["state_sha256"] = actor_digest(actor)
        pending.clear()
        return row

    try:
        start = perf_counter()
        with simulator(Path(study["reference"]), case["game"], case["game_seed"]) as (request, header):
            actor = initialize(request, header)
            initialization_s = perf_counter() - start
            available = sorted(int(p.stem) for p in checkpoints.glob("*.npz"))
            if available:
                if available[-1] > len(rows):
                    raise ValueError("checkpoint is ahead of the verified action log")
                restore_at(actor, request, available[-1])
            with trace_file.open("ab") as trace:
                while actor.tick < budget:
                    t = actor.tick
                    row = transition(actor, request, full_digest=stage=="pre_evaluation" and t>=499)
                    if t < len(rows):
                        compare_replay(row, rows[t][0])
                    else:
                        line = canonical(row) + b"\n"
                        trace.write(line)
                        trace.flush()
                        prefix.update(line)
                        rows.append((row, prefix.hexdigest()))
                    if actor.tick % 500 == 0:
                        os.fsync(trace.fileno())
                        checkpoint = checkpoints / f"{actor.tick}.npz"
                        save(checkpoint, actor, meta(actor.tick))
                        # Keep the recovery point and its predecessor until a later save. Pre-eval
                        # also needs the 500 checkpoint to run the uninterrupted comparison.
                        previous = checkpoints / f"{actor.tick-1000}.npz"
                        if previous.exists() and stage != "pre_evaluation":
                            previous.unlink()
                    if actor.tick <= 3 or actor.tick % 10 == 0:
                        print(f"step {actor.tick}/{budget}, reward {row['reward']}, "
                              f"decision {row['decision_s']:.3f}s, interaction rules {row['active_interaction']}",
                              flush=True)
            end_digest = actor_digest(actor)
        matched = 0
        if stage == "pre_evaluation":
            with simulator(Path(study["reference"]), case["game"], case["game_seed"]) as (request, header):
                resumed = initialize(request, header)
                restore_at(resumed, request, 500)
                for archived, _ in rows[500:]:
                    replay = transition(resumed, request, full_digest=True)
                    compare_replay(replay, archived)
                    matched += 1
                if actor_digest(resumed) != end_digest:
                    raise ArithmeticError("final state differs after the exact half-run restart")
        verify_source(study)
        result = dict(status="complete", initialization_s=initialization_s,
                      resumed_from=available[-1] if available else 0, resume_matched_steps=matched, final_state_sha256=end_digest,
                      process_s=perf_counter()-start, final_checkpoint_sha256=digest(checkpoints / f"{budget}.npz"))
    except Exception as exc:
        result = dict(status="failed", failure_type=type(exc).__name__, reason=str(exc),
                      applied_before_failure=dict(pending), traceback=traceback.format_exc())
        raise
    finally:
        if "result" in locals():
            result.update(**identity, steps=len(rows), return_sum=sum(r["reward"] for r,_ in rows),
                          steps_sha256=digest(trace_file) if trace_file.exists() else hashlib.sha256().hexdigest())
            publish(dest / "result.json", result)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("stage", choices=STAGES)
    cli.add_argument("game")
    cli.add_argument("model_seed", type=int)
    args = cli.parse_args()
    root = args.root.resolve()
    study = read(root / "study.json")
    verify_source(study)
    verify_before(root, study, args.stage)
    case, = [case for case in study["stages"][args.stage]
             if case["game"] == args.game and case["model_seed"] == args.model_seed]
    dest = root / args.stage / run_name(case)
    with exclusive(dest / "worker.lock"):
        if (dest / "result.json").exists():
            raise FileExistsError("a sealed result already exists for this seed")
        episode(root, study, args.stage, case)


if __name__ == "__main__":
    main()

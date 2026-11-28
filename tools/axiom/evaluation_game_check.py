"""Qualify the process adapter against the ten pinned native development tapes."""

import argparse
import json
from pathlib import Path
from unittest import TestCase

import numpy as np

import evaluation_game as adapter
import gameworld_replay as gw
from reference_protocol import EVIDENCE


def expected_events(tape):
    for tick, action in enumerate(tape["action"]):
        clock = {"tick": tick, "episode": int(tape["episode"][tick])}
        if tick == 0 or tape["episode"][tick] != tape["episode"][tick - 1]:
            pre = tape["pixels"][tape["pre_frame"][tick]]
            yield {"event": "reset", **clock, "rgb_sha256": gw.digest(pre.tobytes())}, pre
        post = tape["pixels"][tape["post_frame"][tick]]
        yield {"event": "step", **clock, "rgb_sha256": gw.digest(post.tobytes()),
               "applied": int(action), "reward": float(tape["reward"][tick]),
               "terminated": bool(tape["terminated"][tick]),
               "truncated": bool(tape["truncated"][tick])}, post


def check_row(request, expected, frame):
    command = ("reset" if expected["event"] == "reset"
               else {"tick": expected["tick"], "action": expected["applied"]})
    image, receipt = request(command)
    if receipt != expected or image.flags.writeable:
        raise AssertionError(f"receipt or mutability differs: {receipt}, expected {expected}")
    np.testing.assert_array_equal(image, frame)
    return receipt


def game_check(root, manifest, run):
    game, seed = run["game"], run["game_seed"]
    tape = gw.load_tape(root / "tapes", run)
    events = list(expected_events(tape))
    labels = run["actions"]
    np.testing.assert_array_equal(np.unique(tape["action"]), np.arange(len(labels)))
    receipts = []
    with adapter.simulator(root, game, seed) as (request, header):
        assert header == {"runtime": manifest["runtime"], "game": game, "game_seed": seed,
                          "actions": labels, "condition": "base", "perturb_step": None}
        for expected, frame in events:
            if expected["event"] == "reset":
                with TestCase().assertRaisesRegex(ValueError, "invalid command"):
                    request({"tick": expected["tick"], "action": 0})
            elif expected["tick"] == 0:
                for bad in ("reset", {"tick": 1, "action": 0}, {"tick": 0, "action": len(labels)},
                            {"tick": 0, "action": False}, {"tick": False, "action": 0}):
                    with TestCase().assertRaisesRegex(ValueError, "invalid command"):
                        request(bad)
            receipts.append(check_row(request, expected, frame))

    # Stop immediately after a terminal transition when the tape includes one.
    # The resumed process must then require a reset before accepting the next action.
    boundaries = [i for i, row in enumerate(receipts) if row["event"] == "reset" and i > 0]
    split = boundaries[0] if boundaries else len(events) // 2
    with adapter.simulator(root, game, seed) as (request, restarted):
        assert restarted == header
        frame, row = adapter.replay_receipts(request, iter(receipts[:split]))
        assert row == receipts[split - 1]
        np.testing.assert_array_equal(frame, events[split - 1][1])
        for expected, frame in events[split:]:
            check_row(request, expected, frame)
    return {"game": game, "game_seed": seed, "actions": labels, "ticks": len(tape["action"]),
            "initial_and_later_resets": 1 + len(boundaries),
            "restarted_after_receipts": split, "resumed_receipts": len(events) - split,
            "reward_sum": float(tape["reward"].sum()), "tape_sha256": run["npz_sha256"],
            "receipts_sha256": gw.digest(json.dumps(receipts, sort_keys=True,
                                                     separators=(",", ":")).encode())}


def perturbation_settings_check(root, manifest):
    # Both qualified tape variants switch after eight actions. Their first frames also
    # test the evaluation adapter's configuration before its scheduled action 5000.
    settings = {}
    for run in manifest["runs"]:
        if run["game"] != "Explode" or run["condition"] == "base":
            continue
        condition = run["condition"]
        tape = gw.load_tape(root / "tapes", run)
        with adapter.simulator(root, "Explode", run["game_seed"], condition=condition) as (request, header):
            assert header["condition"] == condition and header["perturb_step"] == 5000
            for row, frame in expected_events(tape):
                if row["tick"] == run["perturb_step"]:
                    break
                check_row(request, row, frame)
            settings[condition] = {"matched_prefix_steps": run["perturb_step"],
                                   "configured_switch": header["perturb_step"]}
    assert set(settings) == {"color", "shape"}
    return settings


def damaged_receipts_check():
    reset = {"event": "reset", "episode": 0, "tick": 0, "rgb_sha256": "a" * 64}
    step = {"event": "step", "episode": 0, "tick": 0, "applied": 1, "reward": 0.0,
            "terminated": False, "truncated": False, "rgb_sha256": "b" * 64}
    for target, change in ((0, {"rgb_sha256": "e" * 64}), (1, {"rgb_sha256": "e" * 64}),
                           (1, {"reward": 1.0}), (1, {"reward": -0.0}), (1, {"applied": 0}),
                           (1, {"terminated": True}), (1, {"truncated": True}),
                           (1, {"episode": 9}), (1, {"tick": 3}), (1, {"episode": False})):
        expected = [reset.copy(), step.copy()]
        expected[target].update(change)
        actual = iter((reset, step))
        with TestCase().assertRaisesRegex(ValueError, "diverged"):
            adapter.replay_receipts(lambda _: (None, next(actual)), expected)
    with TestCase().assertRaisesRegex(ValueError, "initial reset"):
        adapter.replay_receipts(None, [])


def qualify(root):
    manifest_path = root / "tapes/manifest.json"
    manifest_sha = gw.digest(manifest_path.read_bytes())
    assert manifest_sha == EVIDENCE["tapes/manifest.json"]
    manifest = gw.load_manifest(root / "tapes")
    damaged_receipts_check()
    rng_before = np.random.get_state()
    results = [game_check(root, manifest, run) for run in manifest["runs"] if run["condition"] == "base"]
    variants = perturbation_settings_check(root, manifest)
    np.testing.assert_equal(np.random.get_state(), rng_before)
    assert len(results) == len(gw.ACTIONS) and any(row["initial_and_later_resets"] > 1 for row in results)
    return {"passed": True, "stage": "archived development tapes", "games": results,
            "perturbation_settings": variants,
            "manifest_sha256": manifest_sha, "runtime": manifest["runtime"],
            "adapter_sha256": gw.digest(Path(adapter.__file__).read_bytes()),
            "checker_sha256": gw.digest(Path(__file__).read_bytes())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path.home() / "agent-share/axiom-gameworld-reference")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = qualify(args.reference)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        gw.json_write(args.output, result)
    print(f"Passed {len(result['games'])} games, {sum(r['ticks'] for r in result['games'])} "
          f"archived steps, {sum(r['initial_and_later_resets'] for r in result['games'])} resets "
          "and exact continuation after replay on every game.")

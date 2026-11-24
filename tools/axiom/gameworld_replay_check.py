"""Independent development checks for the MIT Gameworld reference archive."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
from unittest import TestCase

import numpy as np

import gameworld_replay as gw


def boundary_check():
    from gymnasium.spaces import Discrete

    class Game:
        action_space = Discrete(2)

        def __init__(self):
            self.buffer = np.zeros(gw.SHAPE, dtype=np.uint8)
            self.tick = 0
            np.random.randint(255)  # Constructor and explicit resets share the game stream.

        def draw(self):
            self.buffer.fill(np.random.randint(255))
            return self.buffer

        def reset(self):
            return self.draw(), {"secret": "reset"}

        def step(self, action):
            self.tick += 1
            return self.draw(), (-1)**self.tick, self.tick == 1, self.tick == 2, {"secret": self.tick}

        def close(self):
            pass

    received = []

    def policy(frame, reward):
        assert frame.shape == gw.SHAPE and frame.dtype == np.uint8 and not frame.flags.writeable
        received.append((int(frame[0, 0, 0]), reward))
        frame.flags.writeable = True
        frame.fill(255)  # The simulator and archive must retain their own frames.
        np.random.randint(255, size=10)  # Must not alter the simulator's stream.
        return 0

    caller = np.random.RandomState(43)
    initial_global = np.random.get_state()
    try:
        np.random.set_state(caller.get_state())
        no_policy = gw.trajectory(Game, lambda _x, _r: 0, 17, 3, 2)
        assert np.random.randint(10**6) == caller.randint(10**6)
        tape = gw.trajectory(Game, policy, 17, 3, 2)
        caller.randint(255, size=30)
        assert np.random.randint(10**6) == caller.randint(10**6)
    finally:
        np.random.set_state(initial_global)
    for key in no_policy:
        np.testing.assert_array_equal(tape[key], no_policy[key])
    oracle = np.random.RandomState(17)
    expected_frames = oracle.randint(255, size=7)[1:]
    np.testing.assert_array_equal(tape["pixels"][:, 0, 0, 0], expected_frames)
    assert received == list(zip(expected_frames[[0, 2, 4]], [None, -1., 1.]))
    assert tape["episode"].tolist() == [0, 1, 2]
    assert tape["post_frame"].tolist() == [1, 3, 5]
    design = {"steps": 3, "actions": ["stay", "move"]}
    gw.validate(tape, design)
    broken = dict(tape, pre_frame=np.array([0, 1, 4], dtype="<i4"))
    with TestCase().assertRaisesRegex(ValueError, "pre_frame"):
        gw.validate(broken, design)


def rolling_oracle(rewards, width):
    return [math.fsum(rewards[end-width:end]) / width for end in range(width, len(rewards)+1)]


def scoring_check():
    # The first reward leaves the window at step 1001. A per-episode restart gives a different result.
    rewards = [7.] + [0.] * 998 + [-2., 3.]
    out = gw.reward_scores(rewards, 1000)
    assert out["return"] == 8 and out["rolling_mean"] == [.005, .001]
    assert out["rolling_mean"] == rolling_oracle(rewards, 1000)
    assert gw.reward_scores(rewards[:3], 1000)["rolling_mean"] == []
    with TestCase().assertRaises(ValueError):
        gw.reward_scores([np.nan], 1000)


def golden_check():
    # Independent development observation from the pinned native Explode reset and first step.
    run = {"game": "Explode", "condition": "base", "game_seed": 12345, "perturb_step": None,
           "steps": 1, "actions": list(gw.ACTIONS["Explode"])}
    tape = gw.play(run, lambda _x, _r: 0)
    hashes = [hashlib.sha256(image).hexdigest() for image in tape["pixels"]]
    assert hashes == ["44550578bf825e29cf2b87796d0375692ec90a727ae442f38f724985161bea76",
                      "71963585f0734609f6f314dd99ec623aa8b3b7565a3ae861a91f51fa38ff11e1"]
    assert tape["reward"].tolist() == [0.] and not tape["terminated"][0] and not tape["truncated"][0]
    return hashes


def archive_check(archive, replay_path, score_path):
    manifest_path = archive / "manifest.json"
    manifest = gw.load_manifest(archive)
    replay = json.loads(replay_path.read_text())
    scores = json.loads(score_path.read_text())
    expected_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert replay["manifest_sha256"] == scores["manifest_sha256"] == expected_hash
    current_source = hashlib.sha256(Path(gw.__file__).read_bytes()).hexdigest()
    assert replay["replayer_sha256"] == scores["scorer_sha256"] == manifest["recorder_sha256"] == current_source
    frames = ticks = terminal_count = 0
    reward_differences = []
    test = TestCase()
    for run, replayed, scored in zip(manifest["runs"], replay["runs"], scores["runs"], strict=True):
        assert run["file"] == replayed["file"] == scored["file"] and replayed["exact"] is True
        with np.load(archive / run["file"], allow_pickle=False) as stored:
            rewards = stored["reward"].tolist()
            images = stored["pixels"]
            assert replayed["frame_sha256"] == [hashlib.sha256(im).hexdigest() for im in images]
            for label, width in (("smoke", 8), ("paper", 1000)):
                metric = scored[label]
                assert metric["decisions"] == len(rewards) and metric["window"] == width
                assert metric["return"] == sum(rewards)
                assert metric["rolling_mean"] == rolling_oracle(rewards, width)
                assert metric["first_full_window_ending_step"] == (width if len(rewards) >= width else None)
            ends = [i for i,(t,u) in enumerate(zip(stored["terminated"], stored["truncated"]), 1) if t or u]
            assert scored["terminal_steps"] == ends
            assert scored["resets"] == 1 + sum(i < len(rewards) for i in ends)
            assert len(images) == len(rewards) + scored["resets"]
            frames += len(images)
            ticks += len(rewards)
            terminal_count += len(ends)
        if run["condition"] != "base":
            assert replayed["aligned_through_switch"]
            assert replayed["first_different_post_frame"] == run["perturb_step"] + 1
            if replayed["first_different_reward"] is not None:
                reward_differences.append((run["file"], replayed["first_different_reward"]))
    assert reward_differences == [("Explode/shape.npz", 46)]
    assert terminal_count and any(r["smoke"]["return"] > 0 for r in scores["runs"])
    assert any(r["smoke"]["return"] < 0 for r in scores["runs"])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first = manifest["runs"][0]
        path = root / first["file"]
        path.parent.mkdir()
        path.write_bytes((archive / first["file"]).read_bytes()[:-1])
        with test.assertRaisesRegex(ValueError, "archive digest"):
            gw.load_tape(root, first)
        wrong = dict(first, array_sha256={})
        with test.assertRaisesRegex(ValueError, "array digest"):
            gw.load_tape(archive, wrong)
        manifest["runs"][1] = first
        (root / "manifest.json").write_text(json.dumps(manifest))
        with test.assertRaisesRegex(ValueError, "design mismatch"):
            gw.load_manifest(root)
    return {"traces": len(replay["runs"]), "decisions": ticks, "native_RGB_frames": frames,
            "ended_episodes": terminal_count, "manifest_sha256": expected_hash,
            "post_switch_reward_differences": reward_differences}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    args = parser.parse_args()
    gw.source_identity(args.checkout)
    boundary_check()
    scoring_check()
    out = archive_check(args.archive, args.replay, args.score)
    out.update({"passed": True, "explode_golden_frame_sha256": golden_check(),
                "checker_sha256": gw.digest(Path(__file__).read_bytes())})
    print(json.dumps(out, indent=2))

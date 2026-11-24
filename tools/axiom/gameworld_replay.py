"""Record and replay small MIT Gameworld development tapes. Uses a separate Gameworld environment."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import subprocess
import tarfile
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np


COMMIT = "85493c477004ecfd3299b0b053884f23d59a5ec5"
# Order is part of the development seed assignment. Labels follow the step implementations.
ACTIONS = {
    "Aviate": ("stay", "flap"), "Bounce": ("stay", "up", "down"),
    "Cross": ("stay", "up", "down"), "Drive": ("stay", "left", "right"),
    "Explode": ("stay", "left", "right"), "Fruits": ("left", "stay", "right"),
    "Gold": ("left", "right", "up", "down", "stay"),
    "Hunt": ("left", "right", "up", "down", "stay"),
    "Impact": ("stay", "left", "right"), "Jump": ("stay", "jump"),
}
CONDITIONS = ("base", "color", "shape")
STEPS = 48
SWITCH = 8
SHAPE = (210, 160, 3)
DTYPES = {"action": "<i2", "reward": "<f8", "terminated": "?", "truncated": "?",
          "episode": "<i4", "pre_frame": "<i4", "post_frame": "<i4"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def designs():
    for index, (game, actions) in enumerate(ACTIONS.items()):
        for condition in CONDITIONS:
            yield {"game": game, "condition": condition, "actions": list(actions),
                   "game_seed": 41010 + index, "action_seed": 51010 + index,
                   "steps": STEPS, "perturb_step": SWITCH if condition != "base" else None,
                   "file": f"{game}/{condition}.npz"}


def source_identity(checkout):
    """Verify the imported MIT distribution byte-for-byte against the pinned Git objects."""
    import gameworld.envs

    if tuple(gameworld.envs.GAME_NAMES) != tuple(ACTIONS):
        raise ValueError("unexpected Gameworld registry")
    if not (3, 10) <= tuple(map(int, platform.python_version_tuple()[:2])) < (3, 12):
        raise ValueError("Gameworld v0.1.0 requires Python 3.10 or 3.11")
    data = subprocess.check_output(["git", "-C", str(checkout), "archive", COMMIT,
                                    "gameworld", "LICENSE", "pyproject.toml"])
    installed = Path(gameworld.envs.__file__).resolve().parents[1]
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive:
            if not member.isfile():
                continue
            content = archive.extractfile(member).read()
            name = member.name
            files[name] = digest(content)
            if name.startswith("gameworld/") and (installed.parent / name).read_bytes() != content:
                raise ValueError(f"installed Gameworld source differs at {name}")
    expected = {p for p in files if p.startswith("gameworld/") and p.endswith(".py")}
    actual = {p.relative_to(installed.parent).as_posix() for p in installed.rglob("*.py")}
    if actual != expected:
        raise ValueError("installed Gameworld Python inventory differs")
    return {"gameworld_commit": COMMIT, "license": "MIT", "source_sha256": files,
            "python": platform.python_version(), "packages": {p: version(p) for p in
            ("gameworld", "gymnasium", "numpy", "pillow", "imageio")}}


class GameRandom:
    """Isolate the legacy process-global NumPy stream used by the serial game simulator."""
    def __init__(self, seed):
        self.state = np.random.RandomState(seed).get_state()

    def call(self, function, *args, **kwargs):
        caller = np.random.get_state()
        np.random.set_state(self.state)
        try:
            return function(*args, **kwargs)
        finally:
            self.state = np.random.get_state()
            np.random.set_state(caller)


def pixels(value):
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.shape != SHAPE:
        raise ValueError(f"unexpected pixels {array.shape} {array.dtype}")
    return array.copy(order="C")


def trajectory(make, choose, seed, steps, n_actions):
    """Call choose(pre-action RGB, previous real reward). Only tick zero receives None."""
    rng = GameRandom(seed)
    env = rng.call(make)
    frames = []
    rows = {key: [] for key in DTYPES}
    previous_reward = None
    episode = -1
    reset = True
    try:
        if env.action_space.n != n_actions or env.action_space.start != 0:
            raise ValueError("action inventory differs")
        for _ in range(steps):
            if reset:
                image, _info = rng.call(env.reset)
                frames.append(pixels(image))
                episode += 1
            pre = len(frames) - 1
            # The policy owns this copy. Neither simulator storage nor scorer metadata is exposed.
            observed = frames[pre].copy()
            observed.flags.writeable = False
            action = choose(observed, previous_reward)
            if not env.action_space.contains(action):
                raise ValueError(f"invalid policy action {action!r}")
            image, reward, terminated, truncated, _info = rng.call(env.step, int(action))
            frames.append(pixels(image))
            previous_reward = float(reward)
            reset = terminated or truncated
            for key, value in (("action", action), ("reward", reward), ("terminated", terminated),
                               ("truncated", truncated), ("episode", episode), ("pre_frame", pre),
                               ("post_frame", pre + 1)):
                rows[key].append(value)
    finally:
        rng.call(env.close)
    return {"pixels": np.stack(frames), **{k: np.asarray(v, dtype=DTYPES[k]) for k, v in rows.items()}}


def random_policy(seed, n_actions):
    generator = np.random.Generator(np.random.PCG64(seed))
    return lambda _pixels, _reward: int(generator.integers(n_actions))


def play(design, choose):
    import gymnasium as gym

    def make():
        return gym.make("Gameworld-" + design["game"] + "-v0",
                        perturb=None if design["condition"] == "base" else design["condition"],
                        perturb_step=design["perturb_step"])
    return trajectory(make, choose, design["game_seed"], design["steps"], len(design["actions"]))


def validate(tape, design):
    n = design["steps"]
    if set(tape) != {"pixels", *DTYPES}:
        raise ValueError("tape inventory differs")
    for key, dtype in DTYPES.items():
        if tape[key].shape != (n,) or tape[key].dtype != np.dtype(dtype):
            raise ValueError(f"invalid {key} shape or dtype")
    if not np.isfinite(tape["reward"]).all():
        raise ValueError("nonfinite reward")
    if np.any((tape["action"] < 0) | (tape["action"] >= len(design["actions"]))):
        raise ValueError("invalid applied action")
    previous_ended = np.r_[False, (tape["terminated"] | tape["truncated"])[:-1]]
    episode = previous_ended.cumsum()
    pre = np.arange(n) + episode
    frame_shape = (int(pre[-1]) + 2, *SHAPE)
    if tape["pixels"].shape != frame_shape or tape["pixels"].dtype != np.uint8:
        raise ValueError("invalid RGB frame sequence")
    for key, expected in (("episode", episode), ("pre_frame", pre), ("post_frame", pre + 1)):
        if not np.array_equal(tape[key], expected):
            raise ValueError(f"invalid {key} boundary")


def capture(archive, checkout):
    if archive.exists():
        raise FileExistsError(f"choose a fresh archive path: {archive}")
    identity = source_identity(checkout)
    manifest = {"format": 1, "purpose": "development", "runtime": identity,
                "recorder_sha256": digest(Path(__file__).read_bytes()),
                "observation": "Native uint8 RGB [height=210,width=160,channel=3], no resizing.",
                "clock": "One row per action. pre_frame and post_frame index pixels. After a terminal "
                         "or truncated step, the next decision consumes a fresh reset frame and the "
                         "actual last reward. At the first reset previous_reward is None.",
                "rng": "MT19937 seeded before constructor, advances across explicit resets. Separate "
                       "PCG64 uniform integer policy advances once per decision, across episodes.",
                "perturbation": {"reference_switch_after_steps": 5000, "development_switch_after_steps": SWITCH,
                                 "timing": "Each wrapper renders step N then switches. The next render "
                                           "(including a reset) can show the change."}, "runs": []}
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=archive.parent) as temporary:
        stage = Path(temporary) / "archive"
        stage.mkdir()
        for design in designs():
            tape = play(design, random_policy(design["action_seed"], len(design["actions"])))
            validate(tape, design)
            path = stage / design["file"]
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, **tape)
            manifest["runs"].append({**design, "npz_sha256": digest(path.read_bytes()),
                                     "array_sha256": {k: digest(v.tobytes()) for k, v in tape.items()}})
        json_write(stage / "manifest.json", manifest)
        stage.rename(archive)
    return manifest


def load_manifest(archive):
    manifest = json.loads((archive / "manifest.json").read_text())
    if manifest["format"] != 1 or manifest["purpose"] != "development":
        raise ValueError("unexpected tape protocol")
    if manifest["runtime"]["gameworld_commit"] != COMMIT or manifest["runtime"]["license"] != "MIT":
        raise ValueError("unexpected native game source")
    expected = list(designs())
    if len(manifest["runs"]) != len(expected):
        raise ValueError("incomplete run inventory")
    for run, design in zip(manifest["runs"], expected, strict=True):
        if any(run.get(k) != v for k, v in design.items()):
            raise ValueError(f"design mismatch at {design['file']}")
    return manifest


def load_tape(archive, run):
    blob = (archive / run["file"]).read_bytes()
    if digest(blob) != run["npz_sha256"]:
        raise ValueError(f"archive digest mismatch: {run['file']}")
    with np.load(io.BytesIO(blob), allow_pickle=False) as stored:
        tape = {k: stored[k] for k in stored.files}
    validate(tape, run)
    if {k: digest(v.tobytes()) for k, v in tape.items()} != run["array_sha256"]:
        raise ValueError(f"array digest mismatch: {run['file']}")
    return tape


def perturbation_observation(base, altered, switch):
    """Use one-based action numbers to compare paired runs, including reset frames."""
    def first_difference(left, right):
        unequal = (left != right).reshape(len(left), -1).any(axis=1)
        indices = np.flatnonzero(unequal)
        return int(indices[0]) + 1 if indices.size else None

    out = {f"first_different_{role}": first_difference(
            base["pixels"][base[role]], altered["pixels"][altered[role]])
           for role in ("pre_frame", "post_frame")}
    out["first_different_reward"] = first_difference(base["reward"], altered["reward"])
    out["first_different_episode_end"] = first_difference(
        np.column_stack((base["terminated"], base["truncated"])),
        np.column_stack((altered["terminated"], altered["truncated"])))
    out["aligned_through_switch"] = all(t is None or t > switch for t in out.values())
    return out


def replay(archive, checkout):
    manifest = load_manifest(archive)
    if manifest["runtime"] != source_identity(checkout):
        raise ValueError("replay runtime differs from capture")
    reports = []
    for run in manifest["runs"]:
        tape = load_tape(archive, run)
        actions = tape["action"]
        expected = np.random.Generator(np.random.PCG64(run["action_seed"])).integers(
            len(run["actions"]), size=run["steps"])
        if not np.array_equal(actions, expected):
            raise ValueError(f"action seed mismatch: {run['file']}")
        sequence = iter(actions.tolist())
        actual = play(run, lambda _image, _previous_reward: next(sequence))
        for field in tape:
            if not np.array_equal(tape[field], actual[field]):
                raise ValueError(f"replay mismatch: {run['file']} {field}")
        row = {"file": run["file"], "exact": True,
               "frame_sha256": [digest(frame.tobytes()) for frame in tape["pixels"]]}
        if run["condition"] == "base":
            base = tape
        else:
            if not np.array_equal(actions, base["action"]):
                raise ValueError("unpaired perturbation actions")
            row.update(perturbation_observation(base, tape, run["perturb_step"]))
        reports.append(row)
    return {"manifest_sha256": digest((archive / "manifest.json").read_bytes()),
            "replayer_sha256": digest(Path(__file__).read_bytes()), "runs": reports}


def reward_scores(rewards, window):
    """Total raw reward and full trailing-window reward per interaction, across episode resets."""
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1 or not np.isfinite(rewards).all() or window <= 0:
        raise ValueError("expected finite 1D rewards and a positive window")
    prefix = np.r_[0., np.cumsum(rewards, dtype=np.float64)]
    return {"decisions": len(rewards), "return": math.fsum(rewards), "window": window,
            "first_full_window_ending_step": window if len(rewards) >= window else None,
            "rolling_mean": ((prefix[window:] - prefix[:-window]) / window).tolist()}


def score(archive):
    manifest = load_manifest(archive)
    records = []
    for run in manifest["runs"]:
        tape = load_tape(archive, run)
        records.append({"file": run["file"], "resets": int(tape["episode"][-1]) + 1,
                        "terminal_steps": (np.flatnonzero(tape["terminated"] | tape["truncated"]) + 1).tolist(),
                        "smoke": reward_scores(tape["reward"], 8), "paper": reward_scores(tape["reward"], 1000)})
    return {"manifest_sha256": digest((archive / "manifest.json").read_bytes()),
            "scorer_sha256": digest(Path(__file__).read_bytes()), "runs": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("capture", "replay", "score"))
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--checkout", type=Path, help="MIT Gameworld Git checkout (needed for capture and replay)")
    args = parser.parse_args()
    if args.operation != "score" and args.checkout is None:
        parser.error("--checkout is required to verify native source identity")
    if args.operation == "capture":
        capture(args.archive, args.checkout)
    else:
        result = score(args.archive) if args.operation == "score" else replay(args.archive, args.checkout)
        print(json.dumps(result, indent=2, allow_nan=False))

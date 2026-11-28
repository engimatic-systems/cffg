"""Run one native MIT Gameworld environment in the pinned reference overlay."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from gameworld_replay import ACTIONS, CONDITIONS, SHAPE, GameRandom, digest, pixels, source_identity
from reference_protocol import EVIDENCE


def _send(stream, metadata, frame=None):
    stream.write(json.dumps(metadata, allow_nan=False).encode("utf-8") + b"\n")
    if frame is not None:
        stream.write(frame)
    stream.flush()


def _worker(reference, game, seed, condition, commands, output):
    import gymnasium as gym

    for name in ("tapes/manifest.json", "overlay/uv.lock"):
        if digest((reference / name).read_bytes()) != EVIDENCE[name]:
            raise ValueError(f"qualified reference file changed: {name}")
    manifest = json.loads((reference / "tapes/manifest.json").read_bytes())
    runtime = source_identity(reference / "gameworld")
    if runtime != manifest["runtime"]:
        raise ValueError("native Gameworld runtime differs from the qualified reference")

    switch = None if condition == "base" else manifest["perturbation"]["reference_switch_after_steps"]
    rng = GameRandom(seed)
    env = rng.call(gym.make, f"Gameworld-{game}-v0",
                   perturb=None if condition == "base" else condition, perturb_step=switch)
    tick, episode, needs_reset = 0, -1, True
    try:
        if env.action_space.n != len(ACTIONS[game]) or env.action_space.start != 0:
            raise ValueError(f"native action space differs for {game}")
        _send(output, {"game": game, "game_seed": seed, "condition": condition,
                       "perturb_step": switch, "actions": ACTIONS[game], "runtime": runtime})
        for line in commands:
            command = json.loads(line)
            if command == "reset" and needs_reset:
                frame, _ = rng.call(env.reset)
                episode += 1
                needs_reset = False
                receipt = {"event": "reset", "episode": episode, "tick": tick}
            elif (not needs_reset and isinstance(command, dict) and set(command) == {"tick", "action"}
                  and type(command["tick"]) is int and command["tick"] == tick
                  and type(command["action"]) is int and env.action_space.contains(command["action"])):
                action = command["action"]
                frame, reward, terminated, truncated, _ = rng.call(env.step, action)
                receipt = {"event": "step", "episode": episode, "tick": tick, "applied": action,
                           "reward": float(reward), "terminated": bool(terminated),
                           "truncated": bool(truncated)}
                needs_reset = terminated or truncated
                tick += 1
            else:
                _send(output, {"error": f"invalid command at tick {tick}, needs_reset={bool(needs_reset)}"})
                continue
            raw = pixels(frame).tobytes()
            _send(output, {**receipt, "rgb_sha256": digest(raw)}, raw)
    finally:
        rng.call(env.close)


@contextmanager
def simulator(reference: Path, game: str, seed: int, *, condition="base"):
    """Yield (request, header). Reset once initially and once after each native episode end.

    request("reset") or request({"action": a, "tick": t}) returns read-only RGB and a receipt.
    Tick zero is the first action. Resets advance the episode number and preserve the tick.
    The private worker keeps the game's MT19937 stream across construction, steps and resets.
    """
    if game not in ACTIONS or condition not in CONDITIONS:
        raise ValueError(f"unknown Gameworld game or condition: {game!r}, {condition!r}")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("game seed must be a 32-bit unsigned integer")
    root = Path(reference).resolve()
    cmd = ["uv", "run", "--frozen", "--no-sync", "python", "-sB", str(Path(__file__).resolve()),
           str(root), game, str(seed), condition]
    env = os.environ.copy()
    for variable in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        env.pop(variable, None)
    with subprocess.Popen(cmd, cwd=root / "overlay", env=env, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE) as worker:
        def receive():
            line = worker.stdout.readline()
            if not line:
                raise RuntimeError(f"Gameworld worker stopped before sending a receipt (status {worker.poll()})")
            return json.loads(line)

        def request(command):
            _send(worker.stdin, command)
            row = receive()
            if "error" in row:
                raise ValueError(row["error"])
            raw = worker.stdout.read(math.prod(SHAPE))
            if len(raw) != math.prod(SHAPE) or digest(raw) != row["rgb_sha256"]:
                raise ValueError("Gameworld RGB transfer failed")
            return np.frombuffer(raw, dtype=np.uint8).reshape(SHAPE), row

        try:
            yield request, receive()
        finally:
            try:
                worker.stdin.close()
                code = worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
                raise
        if code:
            raise RuntimeError(f"Gameworld worker exited {code}")


def replay_receipts(request, receipts):
    """Replay a recorded prefix from a fresh matching simulator and return its last (RGB, receipt).

    The caller binds the header and the complete ordered receipt list to the checkpoint.
    Every field is verified, including the terminal post-frame and each following reset frame.
    """
    last = None
    for index, expected in enumerate(receipts):
        if not isinstance(expected, dict) or expected.get("event") not in ("reset", "step"):
            raise ValueError(f"invalid Gameworld receipt at index {index}")
        if expected["event"] == "reset":
            command = "reset"
        elif "applied" in expected and "tick" in expected:
            command = {"action": expected["applied"], "tick": expected["tick"]}
        else:
            raise ValueError(f"missing applied action or tick at receipt {index}")
        last = request(command)
        actual = last[1]
        # JSON types and signed-zero rewards survive the wire format and the comparison.
        if (json.dumps(actual, sort_keys=True, allow_nan=False)
                != json.dumps(expected, sort_keys=True, allow_nan=False)):
            raise ValueError(f"Gameworld replay diverged at receipt {index} (tick {expected.get('tick')})")
    if last is None:
        raise ValueError("Gameworld replay needs at least the initial reset receipt")
    return last


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("game", choices=ACTIONS)
    cli.add_argument("seed", type=int)
    cli.add_argument("condition", choices=CONDITIONS)
    args = cli.parse_args()
    _worker(args.reference, args.game, args.seed, args.condition, sys.stdin.buffer, sys.stdout.buffer)

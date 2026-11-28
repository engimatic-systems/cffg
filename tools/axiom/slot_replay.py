"""Replay the spatial layer on recorded Gameworld observations in both source precisions."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cffg.experiments.axiom_slots import (
    FrameStatus,
    frame_tokens,
    initial_model,
    initialize_frame,
    observe_frame,
)


def run(tapes, game, steps):
    manifest = json.loads((tapes / "manifest.json").read_text())
    record = next(r for r in manifest["runs"] if r["game"] == game and r["condition"] == "base")
    data = tapes / record["file"]
    if hashlib.sha256(data.read_bytes()).hexdigest() != record["npz_sha256"]:
        raise ValueError("recording digest mismatch")
    with np.load(data, allow_pickle=False) as archive:
        if not 1 <= steps <= len(archive["pre_frame"]):
            raise ValueError("trace length exceeds the recording")
        frames = archive["pixels"][archive["pre_frame"][:steps]]
    base = initial_model(jax.random.key(17), dtype=jnp.float64)
    initialize, observe = jax.jit(initialize_frame), jax.jit(observe_frame)
    replay = {}
    assignments = {}
    for dtype in (jnp.float64, jnp.float32):
        name = np.dtype(dtype).name
        model = jax.tree.map(lambda x: x.astype(dtype), base)
        records, hard = [], []
        for i, image in enumerate(frames):
            tokens = frame_tokens(jnp.asarray(image), dtype)
            if i == 0:
                model, out = initialize(model, tokens)
                status, growth = "INITIAL", []
            else:
                result = observe(model, tokens, out.positions)
                model, out = result.model, result.readout
                status = FrameStatus(int(result.status)).name
                growth = np.asarray(result.growth[: int(result.births)]).tolist()
            if not bool(out.valid):
                raise ArithmeticError(f"{game} {name} frame {i} failed")
            weights = np.asarray(out.assignment)
            winners = weights.argmax(axis=1).astype(np.uint8)
            hard.append(winners)
            records.append(
                dict(
                    frame=i,
                    status=status,
                    growth=growth,
                    winner_sha256=hashlib.sha256(winners.tobytes()).hexdigest(),
                    best_score_min=float(out.best_log_likelihood.min()),
                    max_weight_sum_error=float(abs(weights.sum(axis=1) - 1).max()),
                    occupancy=np.asarray(out.occupancy).tolist(),
                    positions=np.asarray(out.positions.mean).tolist(),
                    counts=np.asarray(model.counts).tolist(),
                )
            )
        replay[name] = records
        assignments[name] = hard
    return dict(
        game=game,
        recording_sha256=record["npz_sha256"],
        initial_appearance_seed=17,
        scope="32 spatial slots at full frame resolution. Previous positions are reused as predictions. No identity, motion, reward or acting.",
        frames=replay,
        pixel_disagreements=[
            int(np.count_nonzero(x != y))
            for x, y in zip(assignments["float32"], assignments["float64"], strict=True)
        ],
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("tapes", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--steps", type=int, default=6)
    cli.add_argument("--games", nargs="+", default=["Explode", "Bounce"])
    args = cli.parse_args()
    if not jax.config.x64_enabled:
        raise SystemExit("enable JAX_ENABLE_X64 to compare both precisions")
    results = [run(args.tapes, game, args.steps) for game in args.games]
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")

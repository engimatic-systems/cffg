"""Measure compact-reload drift in an RGB patch learner on recorded development frames."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from cffg.mixed_learning import MixedBank, MixedBatch, score, update


def digest(data):
    return hashlib.sha256(data).hexdigest()


def patches(frames):
    """Use sixteen RGB patch means and their observed spatial quadrants."""
    columns = np.array_split(frames, 4, axis=2)
    tiles = [row for col in columns for row in np.array_split(col, 4, axis=1)]
    return np.stack([tile.mean(axis=(1, 2)) / 255 for tile in tiles], axis=1)


def prior(dtype):
    colors = [
        [0.05, 0.05, 0.05],
        [0.75, 0.15, 0.15],
        [0.15, 0.75, 0.15],
        [0.15, 0.15, 0.75],
        [0.85, 0.85, 0.85],
        [0.5, 0.5, 0.5],
    ]
    k = len(colors)

    def f(values):
        return jnp.asarray(values, dtype=dtype)

    return MixedBank(
        f(colors),
        f([1] * k),
        f(np.tile(np.eye(3) * 0.3, (k, 1, 1))),
        f([5] * k),
        f([1] * k),
        (f(np.ones((k, 4))),),
        jnp.ones(k, bool),
    )


@jax.jit
def tick(bank, mean):
    n, d = mean.shape
    quadrant = jnp.array([(x // 2) * 2 + y // 2 for x in range(4) for y in range(4)])
    batch = MixedBatch(mean, jnp.zeros((n, d, d), mean.dtype), jnp.ones(n, bool), (quadrant,))
    _, weights, scored = score(bank, batch, include_mixing=True)
    bank, updated = update(bank, batch, weights)
    return bank, weights, scored & updated


def snapshot_roundtrip(bank, exact):
    encoded = io.BytesIO()
    np.savez(encoded, **bank.state_dict(exact=exact))
    encoded.seek(0)
    with np.load(encoded, allow_pickle=False) as snapshot:
        return MixedBank.from_state_dict(snapshot)


def run(bank, patch_tape, boundaries=(), exact=False):
    weights, valid = [], []
    for t, patch in enumerate(patch_tape):
        if t in boundaries:
            bank = snapshot_roundtrip(bank, exact)
        bank, w, ok = tick(bank, patch)
        weights.append(w)
        valid.append(ok)
    if not np.asarray(valid).all():
        raise ArithmeticError("invalid score or update on a development patch")
    return jax.device_get(bank), np.asarray(weights)


def metrics(run_a, reference):
    bank, weights = run_a
    other, baseline = reference
    final_high_delta = max(
        float(np.max(abs(x.astype(float) - y.astype(float))))
        for x, y in zip(jax.tree.leaves(bank)[:5], jax.tree.leaves(other)[:5])
    )
    exact = all(
        np.array_equal(x, y)
        for x, y in zip(jax.tree.leaves(bank), jax.tree.leaves(other), strict=True)
    )
    return {
        "max_assignment_difference": float(np.max(abs(weights.astype(float) - baseline))),
        "changed_patch_labels": int(np.count_nonzero(weights.argmax(2) != baseline.argmax(2))),
        "max_primary_gaussian_or_mixture_parameter_difference": final_high_delta,
        "all_posterior_arrays_identical": exact,
        "all_assignment_arrays_identical": np.array_equal(weights, baseline),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tapes", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    manifest = (args.tapes / "manifest.json").read_bytes()
    runs = json.loads(manifest)["runs"]
    games = {
        "Aviate",
        "Bounce",
        "Cross",
        "Drive",
        "Explode",
        "Fruits",
        "Gold",
        "Hunt",
        "Impact",
        "Jump",
    }
    if {(r["game"], r["condition"]) for r in runs} != {
        (g, c) for g in games for c in ("base", "color", "shape")
    } or len(runs) != 30:
        raise ValueError("incomplete or duplicated development tapes")
    results = []
    for record in runs:
        data = (args.tapes / record["file"]).read_bytes()
        if digest(data) != record["npz_sha256"]:
            raise ValueError(f"changed tape {record['file']}")
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            samples = patches(archive["pixels"][archive["pre_frame"]])
            episode = archive["episode"]
        if samples.shape != (48, 16, 3):
            raise ValueError("unexpected development dimensions")
        tape = jnp.asarray(samples, dtype=jnp.float32)
        boundaries = set(np.flatnonzero(episode[1:] != episode[:-1]) + 1)
        regular = set(range(16, len(episode), 16))
        b = prior(jnp.float32)
        reference = run(b, tape)
        alternatives = {
            "compact_episode": run(b, tape, boundaries),
            "compact_every_16": run(b, tape, regular),
            "exact_every_16": run(b, tape, regular, True),
        }
        rows = {key: metrics(value, reference) for key, value in alternatives.items()}
        if not (
            rows["exact_every_16"]["all_assignment_arrays_identical"]
            and rows["exact_every_16"]["all_posterior_arrays_identical"]
        ):
            raise ArithmeticError("exact restart changed the trajectory")
        results.append(
            {
                "file": record["file"],
                "episode_reloads": len(boundaries),
                "assignments": len(tape) * 16,
                "comparisons": rows,
            }
        )
    out = {
        "schema": 1,
        "manifest_sha256": digest(manifest),
        "worker_sha256": digest(Path(__file__).read_bytes()),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "jax": jax.__version__,
        "device": str(jax.devices()[0]),
        "runs": results,
        "scope": "Six-component RGB patch learner on prior-to-action frames, recorded actions. No AXIOM object identity, planning or policy reward.",
    }
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

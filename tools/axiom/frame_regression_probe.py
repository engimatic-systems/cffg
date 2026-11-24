"""Qualify a full 210x160 frame of weighted spatial regressions on a recorded RGB observation."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import digamma

from cffg.frame_regression import Frame, Regression, fit_frame, score_pixels


def run(archive):
    manifest = json.loads((archive / "manifest.json").read_text())
    record = next(
        r for r in manifest["runs"] if r["game"] == "Explode" and r["condition"] == "base"
    )
    data = archive / record["file"]
    assert hashlib.sha256(data.read_bytes()).hexdigest() == record["npz_sha256"]
    with np.load(data, allow_pickle=False) as z:
        rgb = z["pixels"][z["pre_frame"][0]].astype(float)
    assert rgb.shape == (210, 160, 3)
    y, x = np.indices(rgb.shape[:2])
    pixels = np.column_stack(
        ((x.ravel() - 80) / 80, (y.ravel() - 105) / 105, (rgb.reshape(-1, 3) - 128) / 128)
    )
    k, d, o = 32, 3, 5
    centers = np.column_stack((np.linspace(-0.9, 0.9, k), np.sin(np.arange(k)), np.ones(k)))
    cov = np.tile(np.diag([0.025, 0.03, 0.0]), (k, 1, 1))
    beta = np.tile(np.vstack((np.eye(2, 3), np.zeros((3, 3)))), (k, 1, 1))
    beta[:, 2:, -1] = np.random.default_rng(4131).uniform(-0.9, 0.9, (k, 3))
    prior = Regression(
        *map(
            jnp.float32,
            (beta, np.tile(np.eye(d), (k, 1, 1)), np.ones((k, o)) * 3, np.ones((k, o)) * 0.15),
        )
    )
    frame = Frame(*map(jnp.float32, (pixels, centers, cov)), jnp.ones(k, bool))

    @jax.jit
    def fit(p, f):
        score, valid = score_pixels(p, f)
        w = jax.nn.softmax(score, axis=-1)
        posterior, updated = fit_frame(p, f, w)
        return posterior, score, w, valid & updated

    def measured():
        start = perf_counter()
        result = jax.block_until_ready(fit(prior, frame))
        return result, perf_counter() - start

    _, cold = measured()
    times = []
    for _ in range(5):
        (post, scores, w, ok), elapsed = measured()
        times.append(elapsed)
    if not ok:
        raise ArithmeticError("full frame update failed")
    m, V, a, b = (np.asarray(v, dtype=float) for v in prior)
    X = np.asarray(frame.design, dtype=float)
    C = np.asarray(frame.covariance, dtype=float)
    Y = np.asarray(frame.pixels, dtype=float)
    weight = np.asarray(w, dtype=float)
    # Independent full Gaussian moment equations, evaluated in float64 on normalized coordinates.
    delta = {key: 0.0 for key in ("ell", "mean", "precision", "shape", "rate")}
    for i in range(k):
        h = weight[:, i]
        n = h.sum()
        xx = C[i] + np.outer(X[i], X[i])
        err = Y - X[i] @ m[i].T
        expected = 0.5 * np.sum(
            digamma(a[i])
            - np.log(2 * np.pi * b[i])
            - (a[i] / b[i]) * (err**2 + np.diag(m[i] @ C[i] @ m[i].T))
            - np.trace(np.linalg.solve(V[i], xx)),
            axis=-1,
        )
        pr = V[i] + n * xx
        mean = np.linalg.solve(pr, V[i] @ m[i].T + np.outer(X[i], h @ Y)).T
        rates = b[i] + 0.5 * ((h @ Y**2) + np.diag(m[i] @ V[i] @ m[i].T - mean @ pr @ mean.T))
        for key, v, ref in zip(
            delta,
            (scores[:, i], *(u[i] for u in post)),
            (expected, mean, pr, a[i] + n / 2, rates),
            strict=True,
        ):
            delta[key] = max(delta[key], float(np.max(abs(np.asarray(v) - ref) / (1 + abs(ref)))))
    if max(delta.values()) > 3e-4:
        raise AssertionError(delta)
    return dict(
        pixels=len(Y),
        slots=k,
        frame_sha256=record["array_sha256"]["pixels"],
        device=str(jax.devices()[0]),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        jax=jax.__version__,
        synchronized_first_s=cold,
        synchronized_warm_s=times,
        max_scaled_error=delta,
        scope="Frame-local score, assignments and appearance fit under supplied slot beliefs, CPU float32.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tapes", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.tapes), indent=2, allow_nan=False) + "\n")

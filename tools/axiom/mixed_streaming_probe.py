"""Check invariance of a mixed NIW posterior to batching and translation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from cffg import mixed_learning as mixed


def trial(origin, dtype, n=1024):
    def array(values):
        return jnp.array(values, dtype=dtype)

    bank = mixed.MixedBank(
        array([[origin], [0.]]), array([4096., 2.]), array([[[3.]], [[3.]]]),
        array([4098., 4.]), array([4096., 2.]), (), jnp.array([True, True]),
    )
    one = mixed.MixedBatch(array([[origin + 1.]]), array([[[.125]]]), jnp.array([True]), ())
    weights = array([[1., 0.]])
    batch = jax.tree.map(lambda x: jnp.repeat(x, n, axis=0), one)
    full, full_ok = jax.jit(mixed.update)(bank, batch, jnp.repeat(weights, n, axis=0))

    def stream_step(carry, _):
        current, ok = carry
        posterior, valid = mixed.update(current, one, weights)
        return (posterior, ok & valid), None

    @jax.jit
    def run_stream(prior):
        (posterior, ok), _ = jax.lax.scan(stream_step, (prior, jnp.asarray(True)), None, length=n)
        return posterior, ok

    sequential, stream_ok = run_stream(bank)
    expected_mean = origin + n / (4096 + n)
    expected_scale = 3 + n * .125 + 4096 * n / (4096 + n)
    stream_scale = float(sequential.psi[0, 0, 0])
    relative_error = stream_scale / expected_scale - 1
    return {
        "dtype": np.dtype(dtype).name, "origin": origin, "records": n,
        "prior_kappa": 4096., "observation_offset": 1., "observation_variance": .125,
        "batch_valid": bool(full_ok), "all_stream_updates_valid": bool(stream_ok),
        "expected_mean": expected_mean, "batch_mean": float(full.mu[0, 0]),
        "stream_mean": float(sequential.mu[0, 0]), "expected_scale": expected_scale,
        "batch_scale": float(full.psi[0, 0, 0]), "stream_scale": stream_scale,
        "stream_relative_scale_error": relative_error,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with jax.enable_x64(True):
        rows = [trial(origin, dtype) for dtype in (jnp.float32, jnp.float64)
                for origin in (0., 1., 10_000.)]
    code = Path(mixed.__file__).read_bytes()
    result = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "update_sha256": hashlib.sha256(code).hexdigest(),
        "device": str(jax.devices()[0]), "jax": jax.__version__, "results": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

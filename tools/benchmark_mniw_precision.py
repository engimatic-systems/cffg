"""Measure host setup, one centered update, and archive size for two MNIW forms."""

import argparse
import io
import json
from pathlib import Path
import platform
from time import perf_counter_ns

import numpy as np

from cffg import MNIW, SquareRootMNIW


def storage(state, precision):
    fields = {key: getattr(state, key) for key in ("m", precision, "psi", "nu")}
    stream = io.BytesIO()
    np.savez(stream, **fields)
    return {
        "payload_bytes": sum(np.asarray(x).nbytes for x in fields.values()),
        "npz_bytes": stream.tell(),
    }


def benchmark(d, repetitions):
    m, eye, psi = np.zeros((1, d + 1)), np.eye(d + 1), np.eye(1)
    dense, root = MNIW(m, eye, psi, 3.0), SquareRootMNIW(m, eye, psi, 3.0)
    x = np.full((1, d + 1), 4096.0)
    x[0, 0] = 0.0
    cov, weights = np.eye(d + 1)[None], np.ones(1)
    calls = {
        "dense_setup": lambda: MNIW(m, eye, psi, 3.0),
        "root_setup": lambda: SquareRootMNIW(m, eye, psi, 3.0),
        "dense_to_root": lambda: SquareRootMNIW.from_dense(dense),
        "dense_update": lambda: dense.update_centered(x, cov, weights),
        "root_update": lambda: root.update_centered(x, cov, weights),
    }
    for call in calls.values():
        call()
    samples = {key: [] for key in calls}
    for i in range(repetitions):
        names = list(calls)
        first = i % len(names)
        for key in names[first:] + names[:first]:
            start = perf_counter_ns()
            calls[key]()
            samples[key].append((perf_counter_ns() - start) / 1000)
    # These complete payloads use the registered family's array names.
    return {
        "predictors": d,
        "measurements": {
            key: {
                "median_us": float(np.median(values)),
                "p95_us": float(np.percentile(values, 95)),
                "samples_us": values,
            }
            for key, values in samples.items()
        },
        "dense": storage(dense.update_centered(x, cov, weights), "lam"),
        "root": storage(root.update_centered(x, cov, weights), "lam_root"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    record = {
        "platform": platform.platform(),
        "numpy": np.__version__,
        "working_dtype": "float64",
        "predictor_mean": 4096,
        "n_observations_per_update": 1,
        "cases": [benchmark(d, args.repetitions) for d in (2, 20)],
    }
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

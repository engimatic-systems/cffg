"""Compare learning speed and accumulated count error for two public bank types."""

import argparse
import json
import platform
import statistics
import subprocess
import time
from fractions import Fraction
from importlib.metadata import version

import jax
import jax.numpy as jnp
import numpy as np

from cffg.mixed_learning import score, update
from online_precision_probe import counts_fixture, run_stream
from online_precision_timing import STEPS, sequence


@jax.jit
def score_updates(bank, batches):
    def step(state, batch):
        previous, valid = state
        _, weights, scored = score(previous, batch)
        posterior, updated = update(previous, batch, weights)
        return (posterior, valid & scored & updated), weights

    return jax.lax.scan(step, (bank, jnp.array(True)), batches)


@jax.jit
def fixed_updates(bank, batches, weights):
    def step(state, inputs):
        previous, valid = state
        posterior, updated = update(previous, *inputs)
        return (posterior, valid & updated), None

    return jax.lax.scan(step, (bank, jnp.array(True)), (batches, weights))


def synchronized(function, *args):
    result = jax.block_until_ready(function(*args))
    if not bool(result[0][1]):
        raise ValueError("public mixed-learning call reported invalid parameters")
    return result


def timings(dtype, n, k):
    bank, batches = jax.device_put(sequence(n, k, dtype=dtype))
    training = jax.tree.map(lambda x: x[:-STEPS], batches)
    measured = jax.tree.map(lambda x: x[-STEPS:], batches)
    starts = {}
    for mode, prior in (("compensated", bank), ("rounded", bank.as_rounded())):
        (starts[mode], _), _ = synchronized(score_updates, prior, training)
    _, weights = synchronized(score_updates, starts["compensated"], measured)
    results = {}
    for name, function, args in (
        ("update", fixed_updates, (measured, weights)),
        ("score_and_update", score_updates, (measured,)),
    ):
        samples = {mode: [] for mode in starts}
        for block in range(-3, 12):
            for mode in list(starts)[::1 if block % 2 == 0 else -1]:
                start = time.perf_counter_ns()
                synchronized(function, starts[mode], *args)
                elapsed_ms = (time.perf_counter_ns() - start) / (1e6 * STEPS)
                if block >= 0:
                    samples[mode].append(elapsed_ms)
        ratios = [r / c for r, c in zip(samples["rounded"], samples["compensated"], strict=True)]
        results[name] = {
            "ms_per_update": samples,
            "median_ms": {mode: statistics.median(values) for mode, values in samples.items()},
            "paired_ratio": statistics.median(ratios),
            "paired_ratio_quartiles": np.quantile(ratios, [0.25, 0.75]).tolist(),
        }
    return results


def count_errors(dtype):
    bank, batch, weights = counts_fixture(dtype)
    results = {}
    for name, initial in (("ordinary", 16), ("large", 2 ** (np.finfo(dtype).nmant - 1))):
        prior = bank._replace(alpha=jnp.full_like(bank.alpha, initial))
        expected = Fraction(initial) + sum(Fraction(float(x)) for x in weights[:, 0])
        results[name] = {"initial": initial, "expected_increment": float(expected - initial)}
        for mode, start in (("compensated", prior), ("rounded", prior.as_rounded())):
            posterior = run_stream(start, batch, weights)
            actual = Fraction(float(posterior.alpha[0]))
            results[name][mode] = {
                "visible_increment": float(actual - initial), "count_error": float(actual - expected)
            }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dtype", choices=("float32", "float64"))
    parser.add_argument("--records", type=int, default=1)
    parser.add_argument("--components", type=int, default=32)
    args = parser.parse_args()
    if args.records < 1 or args.components < 1:
        parser.error("records and components must be positive")
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    dtype = np.dtype(args.dtype).type
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    results = {
        "cpu": cpu, "platform": platform.platform(), "device": str(jax.devices()[0]),
        "versions": {key: version(key) for key in ("jax", "jaxlib", "numpy")},
        "dtype": args.dtype, "records": args.records, "components": args.components,
        "training_ticks": 32, "ticks_per_block": STEPS, "warmup_blocks": 3, "measured_blocks": 12,
        "count_errors": count_errors(dtype), "timings": timings(dtype, args.records, args.components),
    }
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

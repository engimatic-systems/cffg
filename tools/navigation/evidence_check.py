"""Qualify the navigation evidence scorer on synthetic data.

From the checkout run:
uv run --frozen --no-sync python tools/navigation/evidence_check.py
"""

from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import time
from unittest import TestCase

import jax
import numpy as np

if __package__:
    from . import evidence
else:
    import evidence


def accuracy(dtype):
    # An asymmetric table detects transposition and any extra evidence for the source.
    training = [(0, 0)] * 6 + [(0, 2)] * 3 + [(1, 1)] * 9
    counts = evidence.fit_counts(training, n_rows=3, n_outcomes=3)
    events = [(r, y) for r in range(3) for y in range(3)] + [(0, 1), (0, 1)]
    expected = np.array([[7, 1, 4], [1, 10, 1], [1, 1, 1]], dtype=float)
    expected /= expected.sum(axis=1, keepdims=True)
    result = evidence.score(counts, events, dtype=dtype)
    oracle = -np.log([expected[event] for event in events])
    np.testing.assert_allclose(result["energies"], oracle, rtol=8*np.finfo(dtype).eps, atol=0)
    np.testing.assert_array_equal(counts, [[6, 0, 3], [0, 9, 0], [0, 0, 0]])

    # One ambiguous cue and two registered source-place rows.
    visits = [(0, 0), (1, 1)] * 5
    simple = evidence.score([[5, 5]], [(0, y) for _, y in visits], dtype=dtype)
    split = evidence.score([[5, 0], [0, 5]], visits, dtype=dtype)
    assert evidence.select(simple, split)["selected"] == "split"
    assert evidence.select(split, simple)["selected"] == "appearance"
    assert evidence.select(simple, simple)["reason"] == "precision_tie"
    # Distinct source IDs with identical frozen probabilities carry no extra prior cost.
    equal = evidence.score([[5, 5], [5, 5]], visits, dtype=dtype)
    assert evidence.select(simple, equal)["selected"] == "appearance"
    # The tolerance also protects near ties on either side of the boundary.
    within = dict(simple, total_bfe=simple["total_bfe"] - simple["total_error_bound"])
    assert evidence.select(simple, within)["reason"] == "precision_tie"
    repeat = evidence.score([[0, 0]], [(0, 0)] * 2, dtype=dtype)
    np.testing.assert_array_equal(repeat["energies"], [simple["energies"][0]] * 2)

    # Score a rare event as well as its nearly certain complement.
    extreme = evidence.score([[np.iinfo(np.int64).max, 0]], [(0, 0), (0, 1)], dtype=dtype)
    assert extreme["energies"][1] > 40 and all(map(math.isfinite, extreme["energies"]))
    assert evidence.score([[0]], [(0, 0)], dtype=dtype)["total_bfe"] == 0
    return dict(events=len(events), max_abs_error=result["max_abs_error"],
                large_count_error=extreme["max_abs_error"], selection=evidence.select(simple, split))


def reject_invalid(dtype):
    check = TestCase()
    for invalid in ([], [[-1]], [[np.nan]], [[.5]], [[np.inf]], [[True]]):
        with check.assertRaises(ValueError):
            evidence.score(invalid, [(0, 0)], dtype=dtype)
    for invalid in ([(-1, 0)], [(0, 2)], [(True, 0)], [(0, 0.25)]):
        with check.assertRaises(ValueError):
            evidence.fit_counts(invalid, n_rows=2, n_outcomes=2)
        with check.assertRaises(ValueError):
            evidence.score([[0, 0], [0, 0]], invalid, dtype=dtype)
    scored = evidence.score([[1, 1]], [(0, 0)], dtype=dtype)
    for changes in (dict(outcomes=[1]), dict(status="failed"), dict(total_bfe=math.nan)):
        with check.assertRaises(ValueError):
            evidence.select(scored, dict(scored, **changes))
    empty = evidence.score([[1]], [], dtype=dtype)
    with check.assertRaises(ValueError):
        evidence.select(empty, empty)
    with jax.enable_x64(dtype is np.float64):
        tree = evidence._tree(2, 2, np.dtype(dtype))
        state = evidence._condition(tree, np.array([[1, .25], [0, .75]], dtype=dtype))
        with check.assertRaisesRegex(ArithmeticError, "infeasible"):
            evidence._event_energy(tree, state, 0, 1)
        assert evidence._event_energy(tree, state, 0, 0) == 0


def scale(n_rows, dtype):
    k = 8
    counts = np.arange(n_rows*k, dtype=np.int64).reshape(n_rows, k) % 11
    events = [(i*17 % n_rows, i*3 % k) for i in range(32)]
    timings = []
    for _ in range(2):
        start = time.perf_counter()
        result = evidence.score(counts, events, dtype=dtype)
        # Conversion of every native energy to a Python float synchronizes each solve.
        timings.append(time.perf_counter() - start)
    return dict(rows=n_rows, outcomes=k, cells=n_rows*k, events=len(events),
                first_score_s=timings[0], repeat_score_s=timings[1],
                max_abs_error=result["max_abs_error"])


def main():
    checks = {}
    for dtype in (np.float32, np.float64):
        checks[dtype.__name__] = accuracy(dtype)
        reject_invalid(dtype)
        checks[dtype.__name__]["scale"] = [scale(rows, dtype) for rows in (32, 320)]
    sources = [Path(__file__), Path(evidence.__file__)]
    return dict(status="passed", backend=jax.default_backend(), machine=platform.machine(), jax=version("jax"),
                source_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in sources},
                timing="One first call including engine construction and lazy JIT, one repeat call. "
                       "Each score includes 32 native bind/run/readout operations and oracle checks.",
                checks=checks)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, allow_nan=False))

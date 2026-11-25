"""Check exact MAP decisions on an existing registered cue-and-commit trace.

The archived actions remain the measured decisions. This post-hoc comparison
breaks exact posterior ties to the lowest arm index, using integer arithmetic.
"""

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reduce_odds(left, right):
    scale = math.gcd(left, right)
    return left // scale, right // scale


def predict(odds, switch):
    left, right = odds
    move = switch.numerator
    stay = switch.denominator - move
    return reduce_odds(stay*left + move*right, move*left + stay*right)


def observe(odds, accuracy, cue):
    hit = accuracy.numerator
    miss = accuracy.denominator - hit
    a, b = (hit, miss) if cue == 0 else (miss, hit)
    return reduce_odds(odds[0]*a, odds[1]*b)


def score(correct, total):
    return dict(correct=correct, trials=total, rate=correct/total)


def audit(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    config = data["protocol"]
    expected = Path(__file__).with_name("drift_protocol.json").read_bytes()
    require(config == json.loads(expected), "trace differs from the registered configuration")
    require([c["seed"] for c in data["cells"]] == config["seeds"], "incomplete registered seed set")
    require(config["first_parent_prior"] == [.5, .5], "expected the registered uniform starting odds")
    flip = Fraction(str(config["parent_flip_probability"]))
    accuracy = Fraction(str(config["cue_accuracy"]))
    k = config["child_ticks_per_period"]
    # The drift protocol assigns the history-using flat control this per-tick rate.
    flat_flip = flip / k
    models = {c["artifact_hash"] for c in data["cells"]}
    require(len(models) == 1, "registered cells have different model artifacts")
    ties = {"hierarchy_arm": 0, "memory_flat_arm": 0}
    changes, cells = [], []
    native_correct = dict.fromkeys(("hierarchy_arm", "memory_flat_arm", "flat_arm"), 0)
    exact_correct = native_correct.copy()
    max_error = 0.
    total = 0
    for cell in data["cells"]:
        rows = cell["trials"]
        require(len(rows) == k*config["periods"] and cell["passed"], "incomplete or failed cell")
        seed = cell["seed"]
        rng = np.random.Generator(np.random.PCG64(seed))
        context = int(rng.integers(2))
        parent = flat = (1, 1)
        native = dict.fromkeys(native_correct, 0)
        exact = native.copy()
        for i, row in enumerate(rows):
            period, tick = divmod(i, k)
            if i and tick == 0:
                if rng.random() < config["parent_flip_probability"]:
                    context = 1-context
                parent = predict(parent, flip)
            cue = context if rng.random() < config["cue_accuracy"] else 1-context
            require((period,tick,context,cue,"converged") == tuple(row[key] for key in
                    ("period","tick","true_arm","cue","status")), f"archived tape/status mismatch at {seed}/{i}")
            parent = observe(parent, accuracy, cue)
            flat = observe(predict(flat, flat_flip), accuracy, cue)
            for field, qfield, odds in (("hierarchy_arm", "parent", parent),
                                       ("memory_flat_arm", "flat_with_memory", flat)):
                q = row[qfield]
                require(len(q)==2 and all(math.isfinite(v) for v in q), "nonfinite or missing arm posterior")
                require(row[field] == int(q[1] > q[0]), f"recorded native MAP differs at {seed}/{i}")
                left, right = odds
                max_error = max(max_error, abs(q[0] - left/(left+right)), abs(q[1] - right/(left+right)))
                tie = left == right
                ties[field] += tie
                choice = int(right > left)
                exact[field] += choice == context
                native[field] += row[field] == context
                if row[field] != choice:
                    changes.append(dict(seed=seed, period=period, tick=tick, policy=field,
                                        exact_tie=tie, true_arm=context, native_arm=row[field], exact_arm=choice,
                                        native_posterior=q))
            require(row["flat_arm"] == cue, f"independent flat action differs at {seed}/{i}")
            exact["flat_arm"] += cue == context
            native["flat_arm"] += cue == context
        for field in native:
            require(cell["metrics"][field] == native[field]/len(rows), f"native metric differs for {seed}/{field}")
            native_correct[field] += native[field]
            exact_correct[field] += exact[field]
        cells.append(dict(seed=seed, trials=len(rows), native_correct=native, exact_correct=exact))
        total += len(rows)
    for field in native_correct:
        archived_mean = float(np.mean([c["native_correct"][field]/c["trials"] for c in cells]))
        require(archived_mean == data["scores"][field], f"archived aggregate differs for {field}")
    scores = {field: dict(native=score(native_correct[field],total), exact=score(exact_correct[field],total))
              for field in native_correct}
    native_advantage = Fraction(native_correct["hierarchy_arm"] - native_correct["flat_arm"], total)
    exact_advantage = Fraction(exact_correct["hierarchy_arm"] - exact_correct["flat_arm"], total)
    # MAP ties above use integer equality only. The configuration separately
    # registers this float64 tolerance for the posterior accuracy check.
    passed = max_error <= 1024*np.finfo(float).eps and all(c["exact_tie"] for c in changes)
    return dict(passed=bool(passed), analysis="post-hoc exact integer MAP replay", native_revision=data["revision"],
                analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                hierarchy_artifact_hash=models.pop(), trace_sha256=hashlib.sha256(raw).hexdigest(),
                protocol_sha256=hashlib.sha256(expected).hexdigest(),
                trials=total, scores=scores, ties=ties, changes=changes, cells=cells, max_abs_posterior_error=max_error,
                native_advantage_over_independent=float(native_advantage), exact_advantage_over_independent=float(exact_advantage))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="registered drift.json from tools/hierarchy/drift.py")
    parser.add_argument("--output", type=Path, help="write a new audit JSON file (default: print)")
    args = parser.parse_args()
    result = audit(args.trace)
    encoded = json.dumps(result, allow_nan=False, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        with args.output.open("x") as out:
            out.write(encoded)
        print(f"Audited {result['trials']} trials; {len(result['changes'])} MAP choices differ.")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Coverage reliability study, incremental output per length.

Dev seeds 5000-5499 at L=1280 (with failure detail), 5000-5299 at longer
lengths. Writes/updates the output JSON after every length.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/Users/ravd/projects/experiments/cffg-nascar-validation-cff-oyie")
sys.path.insert(0, str(ROOT / "src"))
from cffg_nascar_validation import nascar
from cffg_nascar_validation.source_design import gate_coverage, validate_gate_coverage

OUT = Path(sys.argv[1])
PLAN = ((1280, 500), (2560, 300), (5120, 300))

def clopper_lower(k, n, alpha=0.05):
    from scipy.stats import beta as beta_dist
    if k == 0:
        return 0.0
    return float(beta_dist.ppf(alpha, k, n - k + 1))

state = {"results": {}, "fail_detail_1280": {}, "margins_1280": []}
if OUT.exists():
    state = json.loads(OUT.read_text())

for L, n_seeds in PLAN:
    if str(L) in state["results"]:
        print(f"L={L} already done, skipping", flush=True)
        continue
    steps = int(np.ceil(L / 0.8))
    passed, failed = 0, 0
    fail_detail = Counter()
    margins = []
    for seed in range(5000, 5000 + n_seeds):
        params, ds = nascar.source_heterogeneous_problem(n_steps=steps, seed=seed)
        cov = gate_coverage(params, ds, train_steps=L)
        try:
            validate_gate_coverage(cov)
            passed += 1
        except ValueError as e:
            failed += 1
            if L == 1280:
                for part in str(e).split(": ", 1)[1].split("; "):
                    fail_detail[part] += 1
        if L == 1280:
            margins.append({
                "seed": seed,
                "min_successes": cov["minimum_successes"],
                "min_failures": cov["minimum_failures"],
                "min_exposures": cov["minimum_exposure"],
            })
    n = passed + failed
    state["results"][str(L)] = {
        "n": n, "passed": passed, "failed": failed,
        "pass_rate": passed / n,
        "clopper_lower_95": clopper_lower(passed, n),
    }
    if L == 1280:
        state["fail_detail_1280"] = dict(fail_detail)
        state["margins_1280"] = margins
    OUT.write_text(json.dumps(state, indent=1))
    r = state["results"][str(L)]
    print(f"L={L}: {passed}/{n} pass ({r['pass_rate']:.3f}), 95% LB={r['clopper_lower_95']:.4f}", flush=True)
    if L == 1280:
        print("failure modes:", dict(fail_detail), flush=True)

print("done")

"""Regime-label agreement across and within chains for all 10 adapter arms."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ART = Path("/Users/ravd/projects/experiments/cffg-nascar-validation-cff-oyie/artifacts/rslds-confirmatory")
OUT = Path(sys.argv[1])
K = 4
STEP = 40  # subsample draws

ARMS = []
for fam, seeds in (("canonical_homogeneous", (2001, 2002, 2003)),
                   ("source_heterogeneous", (2101, 2103))):
    for seed in seeds:
        base = ART / fam / f"seed-{seed}" / "replicate-0"
        for att in sorted(base.glob("attempt-*"))[::-1]:
            if (att / "shared_initialization/paper_adapter/chain-0.npz").exists():
                for tier in ("shared_initialization", "native_end_to_end"):
                    ARMS.append((fam, seed, tier, att / tier / "paper_adapter"))
                break

PERMS = list(itertools.permutations(range(K)))

def best_perm_agreement(a, b):
    """a, b: [T] int labels. Return (identity_agreement, best_agreement, best_perm)."""
    conf = np.zeros((K, K))
    np.add.at(conf, (a, b), 1.0)
    ident = float(np.trace(conf) / len(a))
    best, bp = -1.0, None
    for p in PERMS:
        val = sum(conf[i, p[i]] for i in range(K)) / len(a)
        if val > best:
            best, bp = float(val), p
    return ident, best, bp

results = []
for fam, seed, tier, path in ARMS:
    regs, avg_seq = [], []
    for c in range(4):
        with np.load(path / f"chain-{c}.npz") as d:
            r = d["regimes"][::STEP]  # [n_draws_sub, T]
            regs.append(r)
    rec = {"family": fam, "seed": seed, "tier": tier, "pairs": [], "within": []}
    # within-chain movement: first vs last retained draw
    for c in range(4):
        ident, best, bp = best_perm_agreement(regs[c][0], regs[c][-1])
        # average per-draw label-flip fraction vs previous subsampled draw
        flips = np.mean(regs[c][1:] != regs[c][:-1])
        rec["within"].append({"chain": c, "first_last_identity": round(ident, 4),
                              "first_last_best": round(best, 4),
                              "mean_flip_frac_subsampled": round(float(flips), 5)})
    # across chains: use final draw of each chain
    for a in range(4):
        for b in range(a + 1, 4):
            ident, best, bp = best_perm_agreement(regs[a][-1], regs[b][-1])
            rec["pairs"].append({"chains": [a, b], "identity": round(ident, 4),
                                 "best": round(best, 4), "best_perm": list(bp)})
    results.append(rec)
    idents = [p["identity"] for p in rec["pairs"]]
    bests = [p["best"] for p in rec["pairs"]]
    nonid = sum(1 for p in rec["pairs"] if p["best_perm"] != [0, 1, 2, 3] and p["best"] > p["identity"] + 0.05)
    print(f"{fam[:9]} {seed} {tier[:6]}: cross-chain identity {min(idents):.3f}-{max(idents):.3f} "
          f"best-perm {min(bests):.3f}-{max(bests):.3f} nontrivial_perms={nonid}", flush=True)
    for w in rec["within"]:
        print(f"   chain {w['chain']}: first-vs-last identity={w['first_last_identity']} "
              f"best={w['first_last_best']} flip_frac={w['mean_flip_frac_subsampled']}", flush=True)

OUT.write_text(json.dumps(results, indent=1))
print("wrote", OUT)

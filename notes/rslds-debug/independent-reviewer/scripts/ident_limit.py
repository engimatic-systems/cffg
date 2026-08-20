"""Posterior distribution of the registered functional-recovery metric,
computed from the retained 2101 adapter chains (diagnosis of the recorded
failure; no repair tuning). Per-draw alignment and labels.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path("/Users/ravd/projects/experiments/cffg-nascar-validation-cff-oyie")
sys.path.insert(0, str(ROOT / "src"))
from cffg_nascar_validation import nascar
from cffg_nascar_validation.functional_recovery import aligned_transition_functional_recovery

ART = ROOT / "artifacts/rslds-confirmatory"
OUT = Path(sys.argv[1])
K, TRAIN = 4, 1280
SUB = 20  # 80 draws per chain

r = json.load(open(ART / "results.json"))
grid = None
for c in r["coordinate_results"]:
    if c["family"] == "source_heterogeneous":
        fam_rec = r["protocol"]["dataset_families"]["source_heterogeneous"]
        grid = np.asarray(fam_rec["balanced_source_state_grid"]["states"], dtype=np.float64)
        break
print("grid shape:", grid.shape)

truth, ds = nascar.source_heterogeneous_problem(n_steps=1600, seed=2101)

def arm_path(tier):
    base = ART / "source_heterogeneous/seed-2101/replicate-0"
    for a in sorted(base.glob("attempt-*"))[::-1]:
        p = a / tier / "paper_adapter"
        if (p / "chain-0.npz").exists():
            return p

def draw_metrics(x_states, regs, W, Roff, dyn):
    """Per-draw alignment, labels, and registered functional metrics."""
    design = np.concatenate((x_states, np.ones((TRAIN, 1))), axis=1)
    affine, *_ = np.linalg.lstsq(design, ds.states[:TRAIN], rcond=None)
    linear = affine[:-1].T
    offset = affine[-1]
    if np.linalg.matrix_rank(linear) != 2:
        return None
    onehot = np.zeros((TRAIN, K))
    onehot[np.arange(TRAIN), regs] = 1.0
    confusion = np.zeros((K, K))
    for truth_regime in range(K):
        confusion[:, truth_regime] = np.sum(onehot[ds.regimes[:TRAIN] == truth_regime], axis=0)
    fit_rows, truth_cols = linear_sum_assignment(-confusion)
    fit_to_truth = np.full(K, -1, dtype=np.int32)
    fit_to_truth[fit_rows] = truth_cols
    params = {"recurrence_weights": W, "recurrence_offsets": Roff, "dynamics_matrix": dyn}
    try:
        return aligned_transition_functional_recovery(
            "rslds", params, truth, grid,
            affine_linear=linear, affine_offset=offset,
            fit_to_truth_labels=fit_to_truth)
    except ValueError:
        return None

records = []
for tier, chains in (("shared_initialization", [0, 1, 2, 3]), ("native_end_to_end", [1, 3])):
    P = arm_path(tier)
    for c in chains:
        with np.load(P / f"chain-{c}.npz") as d:
            X = d["gaussian_states"][::SUB]
            Z = d["regimes"][::SUB]
            W = d["recurrence_weights"][::SUB]
            Roff = d["recurrence_offsets"][::SUB]
            dyn = d["dynamics_matrix"][::SUB]
            llm = float(np.mean(d["log_likelihood"])) / TRAIN
        per_src, worst, contrast = [], [], []
        for i in range(X.shape[0]):
            m = draw_metrics(X[i], Z[i], W[i], Roff[i], dyn[i])
            if m is None:
                continue
            per_src.append(m["per_source_transition_probability_rmse"])
            worst.append(m["worst_source_transition_probability_rmse"])
            contrast.append(m["worst_source_contrast_rmse"])
        per_src = np.asarray(per_src)
        worst = np.asarray(worst)
        contrast = np.asarray(contrast)
        rec = {
            "tier": tier, "chain": c, "loglik_per_step": llm, "n_draws": int(len(worst)),
            "worst_src_rmse_mean": float(worst.mean()), "worst_src_rmse_sd": float(worst.std()),
            "worst_src_rmse_q05": float(np.percentile(worst, 5)),
            "worst_src_rmse_min": float(worst.min()),
            "frac_draws_below_cap": float(np.mean(worst <= 0.15)),
            "worst_contrast_mean": float(contrast.mean()),
            "worst_contrast_q05": float(np.percentile(contrast, 5)),
            "frac_contrast_below_cap": float(np.mean(contrast <= 0.15)),
            "per_source_mean": [float(v) for v in per_src.mean(axis=0)],
            "per_source_sd": [float(v) for v in per_src.std(axis=0)],
        }
        records.append(rec)
        print(f"{tier[:6]} chain {c} (ll={llm:.3f}): worst-src rmse mean={rec['worst_src_rmse_mean']:.3f} "
              f"sd={rec['worst_src_rmse_sd']:.3f} min={rec['worst_src_rmse_min']:.3f} "
              f"P(<=0.15)={rec['frac_draws_below_cap']:.2f} | contrast P(<=0.15)={rec['frac_contrast_below_cap']:.2f}\n"
              f"   per-source mean={np.round(rec['per_source_mean'],3)} sd={np.round(rec['per_source_sd'],3)}", flush=True)

OUT.write_text(json.dumps(records, indent=1))
print("wrote", OUT)

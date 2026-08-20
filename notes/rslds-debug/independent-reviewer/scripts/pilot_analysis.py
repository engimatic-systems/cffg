"""Analyze the dev-seed prior pilot: registered diagnostics + mode structure."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/ravd/projects/experiments/cffg-nascar-validation-cff-oyie")
sys.path.insert(0, str(ROOT / "src"))
from cffg_nascar_validation.diagnostics import split_rhat, effective_sample_size

BASE = Path(sys.argv[1])
OUT = Path(sys.argv[2])
K = 4

def ess_fft(x):
    m, n = x.shape
    half = n // 2
    split = np.concatenate((x[:, :half], x[:, -half:]), axis=0)
    m2, n2 = split.shape
    centered = split - split.mean(axis=1, keepdims=True)
    within = split.var(axis=1, ddof=1).mean()
    between = n2 * split.mean(axis=1).var(ddof=1)
    var = (n2 - 1) / n2 * within + between / n2
    if within == 0 and between == 0:
        return float(m2 * n2)
    if not np.isfinite(var) or var <= 0:
        return 0.0
    nfft = 1 << (2 * n2 - 1).bit_length()
    f = np.fft.rfft(centered, nfft, axis=1)
    acov = np.fft.irfft(f * np.conj(f), nfft, axis=1)[:, :n2].real
    acov /= (n2 - np.arange(n2))[None, :]
    rho = 1.0 - (within - acov.mean(axis=0)) / var
    rho[0] = 1.0
    paired = []
    for lag in range(1, n2 - 1, 2):
        v = rho[lag] + rho[lag + 1]
        if v < 0:
            break
        paired.append(v)
    tau = max(1.0, -1.0 + 2.0 * (1.0 + sum(paired)))
    total = m2 * n2
    return float(min(total, total / tau))

results = {}
for arm in ("frozen_prior_1e-3", "moderate_prior_1.0"):
    path = BASE / arm
    chains, info = [], []
    for c in range(4):
        st = json.loads((path / f"chain-{c}.json").read_text())
        with np.load(path / f"chain-{c}.npz") as d:
            X = d["diagnostic_draws"]
            ll = d["log_likelihood"] / 1280.0
            W = d["recurrence_weights"][-400:]
            regs = d["regimes"][-400::40]
        norms = np.linalg.norm(W, axis=-1)
        occ = np.bincount(regs.ravel(), minlength=K) / regs.size
        chains.append(X)
        info.append({
            "chain": c, "status": st["status"], "wall_seconds": round(st["wall_seconds"], 1),
            "ll_per_step_mean": float(ll.mean()),
            "ll_segments": [float(ll[i*320:(i+1)*320].mean()) for i in range(5)],
            "mean_gate_norm": float(norms.mean()), "max_gate_norm": float(norms.max()),
            "occupancy": [float(x) for x in occ],
            "collapsed": bool((occ < 0.005).any()),
        })
    Xs = np.stack(chains)
    rhat = split_rhat(Xs)
    ess = np.array([ess_fft(Xs[:, :, v]) for v in range(Xs.shape[2])])
    results[arm] = {
        "max_rhat": float(rhat.max()), "min_ess": float(ess.min()),
        "n_fail_rhat": int((rhat > 1.05).sum()), "n_fail_ess": int((ess < 50).sum()),
        "loglik_rhat": float(rhat[0]), "loglik_ess": float(ess[0]),
        "worst_var": int(np.argmax(rhat)),
        "passes_frozen_gate": bool(rhat.max() <= 1.05 and ess.min() >= 50),
        "chains": info,
    }
    r = results[arm]
    print(f"== {arm}: max_rhat={r['max_rhat']:.3f} min_ess={r['min_ess']:.1f} "
          f"fails rhat={r['n_fail_rhat']}/43 ess={r['n_fail_ess']}/43 "
          f"loglik_rhat={r['loglik_rhat']:.3f} PASS={r['passes_frozen_gate']}")
    for i in info:
        print(f"   chain {i['chain']}: ll={i['ll_per_step_mean']:.3f} occ={np.round(i['occupancy'],3)} "
              f"gate_norm mean={i['mean_gate_norm']:.2f} max={i['max_gate_norm']:.1f} collapsed={i['collapsed']}")

OUT.write_text(json.dumps(results, indent=1))
print("wrote", OUT)

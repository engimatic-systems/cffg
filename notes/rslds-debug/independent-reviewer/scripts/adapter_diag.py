"""Per-variable mixing diagnostics for the 10 retained cff-oyie adapter arms.

Read-only over the frozen artifact. Writes summary JSON to the scratchpad.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/ravd/projects/experiments/cffg-nascar-validation-cff-oyie")
ART = ROOT / "artifacts/rslds-confirmatory"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("adapter_diag_out.json")

sys.path.insert(0, str(ROOT / "src"))
from cffg_nascar_validation.diagnostics import split_rhat  # registered code

# variable-family index map for K=4, D=2, p=10 (43 vars)
FAMS = (
    ("loglik_per_step", 0, 1),
    ("sorted_recurrence_weight_norms", 1, 13),
    ("sorted_recurrence_offsets", 13, 25),
    ("sorted_dynamics_cov_eigvals", 25, 33),
    ("sorted_emission_cov_eigvals", 33, 43),
)

def fam_of(i):
    for name, lo, hi in FAMS:
        if lo <= i < hi:
            return name
    raise ValueError(i)

ARMS = []
for fam, seeds in (("canonical_homogeneous", (2001, 2002, 2003)),
                   ("source_heterogeneous", (2101, 2103))):
    for seed in seeds:
        base = ART / fam / f"seed-{seed}" / "replicate-0"
        # find the attempt directory holding the completed run
        attempts = sorted(p for p in base.glob("attempt-*") if p.is_dir())
        for att in attempts[::-1]:
            if (att / "shared_initialization" / "paper_adapter" / "chain-0.npz").exists():
                for tier in ("shared_initialization", "native_end_to_end"):
                    ARMS.append((fam, seed, tier, att / tier / "paper_adapter"))
                break

def ess_fft(x):
    """ESS per Geyer initial monotone-ish (initial positive pairs), split chains.

    x: [chains, draws]. Matches the registered estimator's structure
    (split chains, pooled variance) but uses FFT autocovariance.
    """
    m, n = x.shape
    half = n // 2
    split = np.concatenate((x[:, :half], x[:, -half:]), axis=0)
    m2, n2 = split.shape
    means = split.mean(axis=1, keepdims=True)
    centered = split - means
    within = split.var(axis=1, ddof=1).mean()
    between = n2 * split.mean(axis=1).var(ddof=1)
    var = (n2 - 1) / n2 * within + between / n2
    if within == 0 and between == 0:
        return float(m2 * n2)
    if not np.isfinite(var) or var <= 0:
        return 0.0
    # FFT autocovariance per split chain, averaged; normalization 1/(n-lag)
    nfft = 1 << (2 * n2 - 1).bit_length()
    f = np.fft.rfft(centered, nfft, axis=1)
    acov = np.fft.irfft(f * np.conj(f), nfft, axis=1)[:, :n2].real
    acov /= (n2 - np.arange(n2))[None, :]
    rho = 1.0 - (within - acov.mean(axis=0)) / var
    rho[0] = 1.0
    paired = []
    for lag in range(1, n2 - 1, 2):
        val = rho[lag] + rho[lag + 1]
        if val < 0:
            break
        paired.append(val)
    tau = max(1.0, -1.0 + 2.0 * (1.0 + sum(paired)))
    total = m2 * n2
    return float(min(total, total / tau))


results = []
for fam, seed, tier, path in ARMS:
    chains = []
    for c in range(4):
        with np.load(path / f"chain-{c}.npz") as d:
            chains.append(d["diagnostic_draws"])
    X = np.stack(chains)  # [4, 1600, 43]
    rhat = split_rhat(X)
    ess = np.array([ess_fft(X[:, :, v]) for v in range(X.shape[2])])
    # within-chain-only view: remove chain means, recompute
    Xc = X - X.mean(axis=1, keepdims=True)
    rhat_centered = split_rhat(Xc + X.mean())  # constant shift, keeps scale
    ess_centered = np.array([ess_fft(Xc[:, :, v]) for v in range(X.shape[2])])
    per_var = []
    for v in range(X.shape[2]):
        per_var.append({
            "var": v,
            "family": fam_of(v),
            "rhat": float(rhat[v]),
            "ess": float(ess[v]),
            "rhat_within": float(rhat_centered[v]),
            "ess_within": float(ess_centered[v]),
            "chain_means": [float(mu) for mu in X[:, :, v].mean(axis=1)],
            "chain_stds": [float(s) for s in X[:, :, v].std(axis=1)],
        })
    results.append({
        "family": fam, "seed": seed, "tier": tier,
        "max_rhat": float(rhat.max()), "min_ess": float(ess.min()),
        "max_rhat_within": float(rhat_centered.max()),
        "min_ess_within": float(ess_centered.min()),
        "worst_var": int(np.argmax(rhat)),
        "n_fail_rhat": int((rhat > 1.05).sum()),
        "n_fail_ess": int((ess < 50).sum()),
        "fail_rhat_vars": [int(i) for i in np.flatnonzero(rhat > 1.05)],
        "per_var": per_var,
    })
    r = results[-1]
    print(f"{fam} {seed} {tier}: max_rhat={r['max_rhat']:.3f} min_ess={r['min_ess']:.2f} "
          f"| centered: max_rhat={r['max_rhat_within']:.3f} min_ess={r['min_ess_within']:.2f} "
          f"| fails rhat={r['n_fail_rhat']} ess={r['n_fail_ess']}", flush=True)

OUT.write_text(json.dumps(results, indent=1))
print("wrote", OUT)

"""Label-canonicalized, identified functional diagnostics from retained draws.

For each arm: monitor loglik/step plus the full K x K transition-probability
matrix evaluated at the draw's own latent state at ~40 fixed time indices,
with each chain's labels canonicalized by best permutation of its regime
sequence against chain 0. Frame-invariant, label-invariant, identified.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ART = Path("/Users/ravd/projects/experiments/cffg-nascar-validation-cff-oyie/artifacts/rslds-confirmatory")
ROOT = Path("/Users/ravd/projects/experiments/cffg-nascar-validation-cff-oyie")
sys.path.insert(0, str(ROOT / "src"))
from cffg_nascar_validation.diagnostics import split_rhat

K = 4
SUB = 4        # draw subsample -> 400 draws/chain
N_TIMES = 40
PERMS = list(itertools.permutations(range(K)))
OUT = Path(sys.argv[1])

def arm_paths():
    for fam, seeds in (("canonical_homogeneous", (2001, 2002, 2003)),
                       ("source_heterogeneous", (2101, 2103))):
        for seed in seeds:
            base = ART / fam / f"seed-{seed}" / "replicate-0"
            for att in sorted(base.glob("attempt-*"))[::-1]:
                if (att / "shared_initialization/paper_adapter/chain-0.npz").exists():
                    for tier in ("shared_initialization", "native_end_to_end"):
                        yield fam, seed, tier, att / tier / "paper_adapter"
                    break

def best_perm(a, b):
    conf = np.zeros((K, K))
    np.add.at(conf, (a, b), 1.0)
    best, bp = -1.0, None
    for p in PERMS:
        v = sum(conf[i, p[i]] for i in range(K))
        if v > best:
            best, bp = v, p
    return bp

def decision_list_probs(R, r, x):
    """R:[K,K-1,D], r:[K,K-1], x:[N,D] -> probs [N,K,K] (source, dest)."""
    logits = np.einsum("skd,nd->nsk", R, x) + r[None]      # [N,K,K-1]
    g = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))  # gate probs
    probs = np.empty((x.shape[0], K, K))
    stay = np.ones((x.shape[0], K))
    for k in range(K - 1):
        probs[:, :, k] = stay * g[:, :, k]
        stay = stay * (1.0 - g[:, :, k])
    probs[:, :, K - 1] = stay
    return probs

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

results = []
for fam, seed, tier, path in arm_paths():
    T = 10000 if fam == "canonical_homogeneous" else 1280
    times = np.linspace(0, T - 2, N_TIMES).astype(int)
    chains_func, ref_seq = [], None
    for c in range(4):
        with np.load(path / f"chain-{c}.npz") as d:
            ll = d["log_likelihood"][::SUB] / T
            W = d["recurrence_weights"][::SUB]
            r_off = d["recurrence_offsets"][::SUB]
            X = d["gaussian_states"][::SUB][:, times, :]  # [n,Nt,D]
            seq = d["regimes"][-1]
        if c == 0:
            ref_seq, perm = seq, tuple(range(K))
        else:
            perm = best_perm(seq, ref_seq)  # chain label i -> ref label perm[i]
        n = W.shape[0]
        feats = np.empty((n, 1 + N_TIMES * K * K))
        for i in range(n):
            probs = decision_list_probs(W[i], r_off[i], X[i])  # [Nt,K,K]
            p = np.asarray(perm)
            canon = np.empty_like(probs)
            canon[:, p, :] = probs  # permute sources
            canon2 = np.empty_like(canon)
            canon2[:, :, p] = canon  # permute dests
            feats[i] = np.concatenate(([ll[i]], canon2.ravel()))
        chains_func.append(feats)
    Xf = np.stack(chains_func)  # [4, n, V]
    rhat = split_rhat(Xf)
    # ESS only for the 200 worst-rhat vars to keep compute down
    worst = np.argsort(rhat)[::-1][:200]
    ess = np.array([ess_fft(Xf[:, :, v]) for v in worst])
    rec = {
        "family": fam, "seed": seed, "tier": tier,
        "n_vars": int(Xf.shape[2]),
        "max_rhat": float(rhat.max()),
        "frac_rhat_fail": float((rhat > 1.05).mean()),
        "min_ess_of_worst200": float(ess.min()),
        "loglik_rhat": float(rhat[0]),
    }
    results.append(rec)
    print(f"{fam[:9]} {seed} {tier[:6]}: canon-func max_rhat={rec['max_rhat']:.3f} "
          f"frac_fail={rec['frac_rhat_fail']:.3f} min_ess={rec['min_ess_of_worst200']:.1f} "
          f"loglik_rhat={rec['loglik_rhat']:.3f}", flush=True)

OUT.write_text(json.dumps(results, indent=1))
print("wrote", OUT)

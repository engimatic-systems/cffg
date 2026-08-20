"""Mixing diagnostics restricted to realized-source transition probabilities.

Monitor pi(z_{t+1}=k | x_t, z_t) only at the chain's own sampled (x_t, z_t),
labels canonicalized: the K destination probabilities of the exercised row.
Also split failures of the previous run by realized vs counterfactual source.
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
SUB = 4
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

def decision_list_probs_row(R, r, x, src):
    """R:[K,K-1,D], r:[K,K-1], x:[N,D], src:[N] -> probs [N,K] for the realized source row."""
    Rs = R[src]                      # [N,K-1,D]
    rs = r[src]                      # [N,K-1]
    logits = np.einsum("nkd,nd->nk", Rs, x) + rs
    g = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
    N = x.shape[0]
    probs = np.empty((N, K))
    stay = np.ones(N)
    for k in range(K - 1):
        probs[:, k] = stay * g[:, k]
        stay = stay * (1.0 - g[:, k])
    probs[:, K - 1] = stay
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
            X = d["gaussian_states"][::SUB][:, times, :]
            Z = d["regimes"][::SUB][:, times]
            seq = d["regimes"][-1]
        if c == 0:
            ref_seq, perm = seq, tuple(range(K))
        else:
            perm = best_perm(seq, ref_seq)
        p = np.asarray(perm)
        n = W.shape[0]
        feats = np.empty((n, 1 + N_TIMES * K))
        for i in range(n):
            probs = decision_list_probs_row(W[i], r_off[i], X[i], Z[i])  # [Nt,K]
            canon = np.empty_like(probs)
            canon[:, p] = probs      # destination labels -> reference frame
            feats[i] = np.concatenate(([ll[i]], canon.ravel()))
        chains_func.append(feats)
    Xf = np.stack(chains_func)
    rhat = split_rhat(Xf)
    worst = np.argsort(rhat)[::-1][:80]
    ess = np.array([ess_fft(Xf[:, :, v]) for v in worst])
    rec = {
        "family": fam, "seed": seed, "tier": tier, "n_vars": int(Xf.shape[2]),
        "max_rhat": float(rhat.max()),
        "frac_rhat_fail": float((rhat > 1.05).mean()),
        "min_ess_of_worst80": float(ess.min()),
        "loglik_rhat": float(rhat[0]),
    }
    results.append(rec)
    print(f"{fam[:9]} {seed} {tier[:6]}: realized-row max_rhat={rec['max_rhat']:.3f} "
          f"frac_fail={rec['frac_rhat_fail']:.3f} min_ess={rec['min_ess_of_worst80']:.1f} "
          f"loglik_rhat={rec['loglik_rhat']:.3f}", flush=True)

OUT.write_text(json.dumps(results, indent=1))
print("wrote", OUT)

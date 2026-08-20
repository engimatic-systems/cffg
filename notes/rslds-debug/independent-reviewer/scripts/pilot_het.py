"""Dev-seed pilot: heterogeneous adapter mixing under frozen vs moderate recurrence prior.

Dev dataset seed 4201 (never used in calibration or finals). Native-style chains
(each chain runs the adapter's own observation-only initialization). Frozen chain
budget: warmup 400, draws 1600, thin 1, 4 chains. Chains run in parallel.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/ravd/projects/experiments/cffg-nascar-validation-cff-oyie")
sys.path.insert(0, str(ROOT / "src"))
from cffg_nascar_validation import nascar
from cffg_nascar_validation.rslds_mapping import reference_prior_arrays
from cffg.models import vague_rslds_fit_priors

OUTBASE = Path(sys.argv[1])
DATA_SEED = 4201
TRAIN_STEPS = 1280
STEPS = 1600
CHAIN_SEEDS = [420110, 420111, 420112, 420113]
REFERENCE_PROJECT = ROOT / "reference" / "legacy-runtime"

params, dataset = nascar.source_heterogeneous_problem(n_steps=STEPS, seed=DATA_SEED)
train = np.asarray(dataset.observations[:TRAIN_STEPS], dtype=np.float64)

procs = []
for arm, strength in (("frozen_prior_1e-3", 1e-3), ("moderate_prior_1.0", 1.0)):
    outdir = OUTBASE / arm
    outdir.mkdir(parents=True, exist_ok=True)
    priors = vague_rslds_fit_priors(4, 2, 10, recurrence_strength=strength)
    arrays = reference_prior_arrays(priors)
    obs_path = outdir / "observations.npy"
    priors_path = outdir / "priors.npz"
    np.save(obs_path, train, allow_pickle=False)
    np.savez(priors_path, **arrays)
    for chain, seed in enumerate(CHAIN_SEEDS):
        cmd = [
            "uv", "run", "--project", str(REFERENCE_PROJECT),
            "cffg-rslds-comparison-chain",
            "--observations", str(obs_path),
            "--priors", str(priors_path),
            "--seed", str(seed),
            "--warmup", "400", "--draws", "1600", "--thin", "1",
            "--output", str(outdir / f"chain-{chain}.npz"),
            "--status", str(outdir / f"chain-{chain}.json"),
        ]
        log = open(outdir / f"chain-{chain}.log", "w")
        env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   VECLIB_MAXIMUM_THREADS="2")
        procs.append((arm, chain, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
        print("launched", arm, "chain", chain, flush=True)

for arm, chain, p, log in procs:
    rc = p.wait()
    log.close()
    print(arm, "chain", chain, "exit", rc, flush=True)
print("all done")

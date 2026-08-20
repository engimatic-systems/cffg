"""Instrumented reproduction of heterogeneous outer-loop stall on a dev seed.

Frozen protocol config but max_outer_iterations=48 (first 16 iterations equal
the frozen budget trajectory because the fit is deterministic given the key).
Per-family residual attribution via monkeypatched _fixed_point_residual.
"""
import json
import sys
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

ROOT = Path("/Users/ravd/projects/experiments/cffg-nascar-validation-cff-oyie")
sys.path.insert(0, str(ROOT / "src"))
from cffg_nascar_validation import nascar
import cffg.models.rslds_fit as rf
from cffg.models import PCAARHMMInitializer, vague_rslds_fit_priors

OUT = Path(sys.argv[1])
DATA_SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 4201
MAX_OUTER = int(sys.argv[3]) if len(sys.argv) > 3 else 48

params, ds = nascar.source_heterogeneous_problem(n_steps=1600, seed=DATA_SEED)
train = np.asarray(ds.observations[:1280], dtype=np.float64)

per_iter = []
_orig = rf._fixed_point_residual
def instrumented(old, new):
    detail = {}
    for name in old:
        b, a = np.asarray(old[name]), np.asarray(new[name])
        scale = max(1.0, float(np.max(np.abs(b), initial=0.0)), float(np.max(np.abs(a), initial=0.0)))
        detail[name] = float(np.max(np.abs(a - b), initial=0.0)) / scale
    per_iter.append(detail)
    return _orig(old, new)
rf._fixed_point_residual = instrumented

result = rf.fit_rslds(
    train,
    regimes=4,
    latent_dimension=2,
    initializer=PCAARHMMInitializer(),
    priors=vague_rslds_fit_priors(4, 2, 10),
    max_outer_iterations=MAX_OUTER,
    max_inner_iterations=4000,
    inner_tolerance=1e-4,
    outer_tolerance=0.1,
    damp=0.9,
    max_inner_retries=3,
    key=jax.random.key(97001),
)

print("status:", result.status, "iterations:", result.iterations, "residual:", result.residual)
rows = []
for i, (res, fe) in enumerate(zip(result.residuals, result.free_energies)):
    detail = per_iter[i] if i < len(per_iter) else {}
    top = sorted(detail.items(), key=lambda kv: -kv[1])[:3]
    rows.append({"iter": i + 1, "residual": res, "free_energy": fe,
                 "top_families": [[k, round(v, 5)] for k, v in top]})
    print(f"it {i+1:2d}: resid={res:.4f} FE={fe:.1f} top={[(k.split('/')[-1], round(v,4)) for k,v in top]}", flush=True)

OUT.write_text(json.dumps({
    "data_seed": DATA_SEED, "max_outer": MAX_OUTER,
    "status": str(result.status), "iterations": result.iterations,
    "residual": result.residual, "rows": rows,
    "residual_detail": per_iter,
}, indent=1))
print("wrote", OUT)

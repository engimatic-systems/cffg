"""Decide pilot readiness and seal sources and fresh seeds before evaluation."""
from __future__ import annotations

import argparse
import importlib.metadata
import platform
from pathlib import Path
import subprocess

from study import (CONDITIONS, DEFAULT_ROOT, DEVELOPMENT, ENDURANCE, HELDOUT, JULIA_PROJECT,
                   LIBRARIES, PRACTICAL, REPO, configure, read, save, sha, sources)
from campaign import identity, runtime


def decide(root):
    import numpy as np

    if (root/"freeze.json").exists() or (root/"heldout").exists():
        raise ValueError("Evaluation already frozen or opened")
    for stage in ("development","endurance"):
        seal=read(root/(stage+"-completed.json"))
        if not seal or seal["runtime_sha256"]!=identity(runtime()):
            raise ValueError(f"Missing or stale {stage} measurement inventory")
        for name,h in seal["runs"].items():
            if sha(root/name)!=h: raise ValueError(f"Changed {stage} archive {name}")
    qual=root/"qual"
    with_scope=read(qual/"plant-qualification.json")
    nonlinear=read(qual/"nonlinear.json")
    native={lib:read(qual/f"{lib}.json") for lib in LIBRARIES}
    if not (with_scope and nonlinear and all(native.values())): raise ValueError("Missing qualification evidence")
    checks={}
    def gate(key,ok,detail):
        checks[key]=dict(passed=bool(ok),detail=detail)
    gate("plant",with_scope["passed"],"Independent integration, hidden outlets, flow excitation, reachable targets and sensor benefit.")
    gate("affine",all(r["passed"] for r in native.values()),"Both native filters and raw two-flow planners compared with independent dense references.")
    gate("nonlinear",nonlinear["passed"],"Independent particle reference over short nonlinear windows, with repeated budgets and numerical error checks.")
    results={}
    agreement=[]
    workloads={}
    all_causal=True
    for lib in LIBRARIES:
        gains=[]; adaptive=[]; worst=0.; total_s=0
        for seed in DEVELOPMENT:
            dir_=root/"development"/f"seed-{seed}"
            costs={}
            for arm in ("frozen","informed","adaptive"):
                with np.load(dir_/"changing"/f"tape-{arm}-{lib}.npz") as f:
                    mask=f["phase"]!=0
                    residual=f["m"][mask,:10]-f["x"][1:][mask]
                    costs[arm]=float(np.sqrt(np.mean(residual[:,[0,2,3,4,5,6,7,9]]**2)))
                if arm=="adaptive": adaptive.append(costs[arm])
                for cond in CONDITIONS:
                    for mode in ("control","tape"):
                        path=dir_/cond/f"{mode}-{arm}-{lib}.npz"
                        total_s+=read(path.with_suffix(".json"))["episode_s"]
                        if arm=="adaptive" and mode=="control":
                            with np.load(path) as f:
                                worst=max(worst,float(np.mean(f["held"]|f["failed"])))
            gains.append(100*(1-costs["informed"]/costs["frozen"]))
            # The two regimes have identical past data and all forecasts until their first change.
            with (np.load(dir_/"stationary"/f"tape-adaptive-{lib}.npz") as st,
                  np.load(dir_/"changing"/f"tape-adaptive-{lib}.npz") as ch):
                t=int(np.flatnonzero(ch["phase"])[0])
                all_causal &= all(np.array_equal(st[k][:t],ch[k][:t]) for k in ("obs","u","m","P"))
        long=root/"endurance"/f"seed-{ENDURANCE}"/"changing"/f"control-adaptive-{lib}.npz"
        with np.load(long) as f:
            worst=max(worst,float(np.mean(f["held"]|f["failed"])))
            finite=bool(np.isfinite(f["m"]).all() and len(f["m"])==6000)
        gate("information_"+lib,np.mean(gains)>=5 and np.count_nonzero(np.array(gains)>0)>=2,
             f"Informed hidden-state RMSE reduction vs frozen by seed (%) {np.round(gains,3).tolist()}.")
        gate("reliability_"+lib,worst<=.01 and finite,
             f"Worst adaptive control hold/failure fraction {worst:.6g} across short and 6000-tick development episodes.")
        agreement.append(np.array(adaptive))
        results[lib]=dict(headroom_percent=gains,adaptive_rmse=adaptive,worst_unreliable_fraction=worst)
        workloads[lib]=total_s/len(DEVELOPMENT)*10*len(HELDOUT)
    gate("causal_history",all_causal,"Identical fixed-tape adaptive posteriors before the first hidden plant change.")
    gate("endurance",all(read(root/"workers"/"endurance"/f"{ENDURANCE}-{lib}.json") for lib in LIBRARIES),
         "Both libraries completed the 6000-transition endurance case.")
    status="go" if all(v["passed"] for v in checks.values()) else "no_go"
    dev_deltas=agreement[0]-agreement[1]
    pilot=dict(status=status,gates=checks,development=results,
               planned_episode_wall_seconds=workloads,
               projected_serial_episode_hours=sum(workloads.values())/3600,
               development_primary_differences=dev_deltas.tolist(),
               development_projection_95pct_halfwidth=float(1.96*np.std(dev_deltas,ddof=1)/np.sqrt(len(HELDOUT))),
               practical_margin=PRACTICAL,source_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip())
    save(root/"pilot.json",pilot)
    if status=="go":
        versions={p:importlib.metadata.version(p) for p in ("jax","jaxlib","numpy","scipy","casadi","pcgym")}
        save(root/"freeze.json",dict(seeds=list(HELDOUT),steps=6000,protocol_sha256=sha(root/"protocol.json"),
             pilot_sha256=sha(root/"pilot.json"),runtime_sha256=identity(runtime()),source_sha256=sources(),
             practical_margin=PRACTICAL,selected="matched native affine Gaussian",
             bootstrap_unit="paired seed",estimated_serial_episode_hours=pilot["projected_serial_episode_hours"],
             head=pilot["source_revision"],python=platform.python_version(),platform=platform.platform(),versions=versions,
             julia_manifest_sha256=sha(JULIA_PROJECT/"Manifest.toml"),
             qualification_sha256={p.name:sha(p) for p in sorted(qual.glob("*.json"))},
             development_sha256=sha(root/"development-completed.json"),endurance_sha256=sha(root/"endurance-completed.json")))
    return pilot


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=DEFAULT_ROOT); args=p.parse_args()
    configure()
    result=decide(args.output.resolve())
    print(result["status"],result["projected_serial_episode_hours"],result["gates"])

"""Run serial native workers and retain independently verifiable extraction episodes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import time

from study import (ARMS, CONDITIONS, DEFAULT_ROOT, DEVELOPMENT, ENDURANCE, EXPERIMENT, HELDOUT,
                   HERE, HORIZON, JULIA_PROJECT, LIBRARIES, MODES, REPO, configure, inventory,
                   protocol, read, save, sha, sources)


def identity(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",",":")).encode()).hexdigest()


def runtime():
    names=("campaign.py","episode.py","study.py","plant.py","dynamics.py","cffg_native.py",
           "rxinfer_native.py","rxinfer_native.jl")
    files=[HERE / name for name in names] + list(HERE.glob("*.c")) + list(HERE.glob("*.cpp"))
    files.extend(REPO.joinpath("src/cffg").rglob("*.py"))
    source={str(p.relative_to(REPO)):sha(p) for p in files}
    ext=EXPERIMENT/".venv/lib/python3.12/site-packages"
    external={str(p):sha(p) for p in (REPO/"uv.lock", JULIA_PROJECT/"Project.toml",
              JULIA_PROJECT/"Manifest.toml",ext/"pcgym/model_classes.py",EXPERIMENT/"uv.lock")}
    return dict(source=source,external=external)


def setup(root):
    path=root/"protocol.json"
    if not path.exists():
        save(path,protocol())
    if identity(read(path))!=identity(protocol()):
        raise ValueError("Persisted protocol differs from current study")


def expected_seeds(root,stage):
    if stage=="development": return DEVELOPMENT
    if stage=="endurance": return (ENDURANCE,)
    freeze=read(root/"freeze.json")
    if not freeze or freeze["protocol_sha256"]!=sha(root/"protocol.json") or freeze["pilot_sha256"]!=sha(root/"pilot.json"):
        raise ValueError("Fresh seeds require a frozen green pilot")
    if freeze["runtime_sha256"]!=identity(runtime()):
        raise ValueError("Measurement source differs from frozen revision")
    if any(sha(REPO/p)!=h for p,h in freeze["source_sha256"].items()):
        raise ValueError("Frozen source file changed")
    return tuple(freeze["seeds"])


def keys(stage):
    if stage=="endurance":
        return [("changing","control","adaptive")]
    return [(cond,mode,arm) for cond in CONDITIONS for mode in MODES for arm in ARMS]


def valid(path, signature, stage, seed, cond, mode, arm, lib):
    side=read(path.with_suffix(".json"))
    if not side: return False
    if not path.is_file() or sha(path)!=side["sha256"]:
        raise ValueError(f"Broken episode seal {path}")
    if [side[k] for k in ("stage","seed","condition","mode","arm","library","runtime_sha256")]!=[
            stage,seed,cond,mode,arm,lib,signature]:
        raise ValueError(f"Episode identity mismatch {path}")
    return True


def worker(root,stage,seed,lib):
    import importlib
    import numpy as np
    import plant
    import episode
    if seed not in expected_seeds(root,stage): raise ValueError("Seed is outside selected stage")
    paths=[(cond,mode,arm,root/stage/f"seed-{seed}"/cond/f"{mode}-{arm}-{lib}.npz")
           for cond,mode,arm in keys(stage)]
    signature=identity(runtime())
    if all(valid(path,signature,stage,seed,c,m,a,lib) for c,m,a,path in paths): return
    block=root/"workers"/stage/f"{seed}-{lib}.json"
    epoch=time.time()
    cls=importlib.import_module(lib+"_native").Backend
    start=perf_counter(); backend=cls(); setup_s=perf_counter()-start
    try:
        simulator=plant.Plant()
        Q=np.eye(11)*1e-5
        nominal=np.r_[plant.nominal_profile(),np.log(5.)]
        start=perf_counter()
        wm,wp=backend.update(nominal,episode.PRIOR,np.eye(11),np.zeros(11),Q,episode.H,episode.R,
                             episode.H@nominal)
        episode.plan(backend,wm,wp,episode.NOMINAL_U,np.full(HORIZON,.33),None,Q)
        warmup_s=perf_counter()-start
        startup_s=time.time()-float(os.environ.get("EXTRACTION_LAUNCH",epoch))
        n=600 if stage=="development" else 6000
        for c,m,a,path in paths:
            if valid(path,signature,stage,seed,c,m,a,lib): continue
            tape=plant.design(seed,n,c)
            begin=perf_counter()
            arrays=episode.simulate(backend,simulator,tape,a,m)
            wall=perf_counter()-begin
            meta=dict(stage=stage,seed=seed,steps=n,condition=c,mode=m,arm=a,library=lib,
                      setup_s=setup_s,warmup_s=warmup_s,cold_s=startup_s,peak_rss_mib=float(arrays["rss_samples"][:,1].max()),
                      episode_s=wall,runtime_sha256=signature,protocol_sha256=sha(root/"protocol.json"))
            path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_suffix(".partial")
            with tmp.open("wb") as f:
                np.savez_compressed(f,**arrays,seed=np.asarray(seed),meta=np.asarray(json.dumps(meta)))
            os.replace(tmp,path)
            save(path.with_suffix(".json"),dict(meta,sha256=sha(path)))
            print(f"{stage} {seed} {c} {m} {a} {lib} {wall:.2f}s holds={arrays['held'].sum()} failures={arrays['failed'].sum()}",flush=True)
        save(block,dict(library=lib,seed=seed,stage=stage,runtime_sha256=signature,
                        setup_s=setup_s,warmup_s=warmup_s,process_to_first_synthetic_plan_s=startup_s,
                        finished=time.time()))
    finally:
        if hasattr(backend,"close"): backend.close()


def run(root,stage,lock_fd):
    seeds=expected_seeds(root,stage)
    manifest=runtime()
    signature=identity(manifest)
    save(root/"sources"/(signature+".json"),manifest)
    for i,seed in enumerate(seeds):
        for lib in LIBRARIES[i%2:]+LIBRARIES[:i%2]:
            if all(valid(root/stage/f"seed-{seed}"/c/f"{m}-{a}-{lib}.npz",signature,stage,seed,c,m,a,lib)
                   for c,m,a in keys(stage)):
                receipt=root/"workers"/stage/f"{seed}-{lib}.json"
                if not receipt.exists():
                    save(receipt,dict(library=lib,seed=seed,stage=stage,runtime_sha256=signature,
                                      recovered_from_sealed_episodes=True,setup_s=None,warmup_s=None,
                                      process_to_first_synthetic_plan_s=None))
                continue
            print("Starting",stage,seed,lib,flush=True)
            start=perf_counter()
            subprocess.run([sys.executable,__file__,"worker","--output",str(root),"--stage",stage,
                            "--seed",str(seed),"--library",lib,"--lock-fd",str(lock_fd)],check=True,pass_fds=(lock_fd,),
                           env={**os.environ,"EXTRACTION_LAUNCH":str(time.time())})
            print("Block elapsed",round(perf_counter()-start,2),flush=True)
    paths=inventory(root,stage,seeds) if stage!="endurance" else [
        root/stage/f"seed-{ENDURANCE}"/"changing"/f"control-adaptive-{lib}.npz" for lib in LIBRARIES]
    digest={str(p.relative_to(root)):sha(p) for p in paths}
    seal=dict(runtime_sha256=signature,seeds=seeds,steps=600 if stage=="development" else 6000,
              protocol_sha256=sha(root/"protocol.json"),runs=digest)
    if stage=="heldout":
        seal["freeze_sha256"]=sha(root/"freeze.json")
    dest=root/("completed.json" if stage=="heldout" else stage+"-completed.json")
    if dest.exists() and read(dest)!=json.loads(json.dumps(seal)):
        raise ValueError("Existing seal differs from current inventory")
    save(dest,seal)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","run","worker"))
    p.add_argument("--output",type=Path,default=DEFAULT_ROOT)
    p.add_argument("--stage",choices=("development","endurance","heldout"))
    p.add_argument("--seed",type=int)
    p.add_argument("--library",choices=LIBRARIES)
    p.add_argument("--lock-fd",type=int,help=argparse.SUPPRESS)
    args=p.parse_args(); root=args.output.resolve()
    configure()
    root.mkdir(parents=True,exist_ok=True)
    lock_path=root/"campaign.lock"
    with (lock_path.open("a") if args.lock_fd is None else os.fdopen(args.lock_fd,"a")) as lock:
        if not os.path.samestat(os.fstat(lock.fileno()),lock_path.stat()):
            raise ValueError("Inherited lock belongs to a different study")
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=="worker":
            if not read(root/"protocol.json"): raise ValueError("Missing protocol")
            worker(root,args.stage,args.seed,args.library)
        else:
            setup(root)
            if args.action=="run": run(root,args.stage,lock.fileno())


if __name__=="__main__":
    main()

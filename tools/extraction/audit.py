"""Reconstruct measurements from frozen tapes and spot-check both native implementations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from study import DEFAULT_ROOT, configure, read, save, sha


def audit(root,stage):
    import numpy as np
    import plant
    from episode import H,HIDDEN,PRIOR,R
    from dynamics import linearize
    from study import DEADLINE, MOTION_SD
    from cffg_native_check import dense_update

    seal=read(root/("completed.json" if stage=="heldout" else "development-completed.json"))
    if not seal: raise ValueError("The scoring stage is still open")
    simulator=plant.Plant()
    check_max=dict(state=0.,sensor=0.,mean=0.,cov=0.,pre_observation=0.)
    ncheck=total=0
    primary={}; full_means={}; held_total=0
    def agree(key,a,b,tol):
        err=float(np.max(np.abs(a-b)))
        if not np.isfinite(err) or err>tol:
            raise ValueError(f"Independent {key} check differs by {err}")
        check_max[key]=max(check_max[key],err)
    for rel, digest in seal["runs"].items():
        path=root/rel
        if sha(path)!=digest or read(path.with_suffix(".json"))["sha256"]!=digest:
            raise ValueError(f"Trace hash mismatch {path}")
        with np.load(path) as src:
            meta=json.loads(str(src["meta"]))
            lib=meta["library"]; arm=meta["arm"]; seed=meta["seed"]; cond=meta["condition"]
            n=meta["steps"]; mode=meta["mode"]
            assert path.name==f"{mode}-{arm}-{lib}.npz" and path.parent.name==cond and path.parent.parent.name==f"seed-{seed}"
            tape=plant.design(seed,n,cond)
            x=src["x"]; obs=src["obs"]; u=src["u"]; m=src["m"]; P=src["P"]
            held=src["held"]; late=src["late"]; failed=src["filter_failed"]
            held_total+=int(held.sum()); total+=n
            assert np.array_equal(x[0],tape["initial"]) and np.array_equal(src["y0"],tape["y0"])
            for k in ("kla","phase","event","target"):
                assert np.array_equal(src[k],tape[k])
            agree("sensor",obs,x[1:][:,plant.OUTLETS]+tape["noise"],1e-12)
            if mode=="tape":
                assert np.array_equal(u,tape["flows"])
            else:
                last=np.vstack((plant.NOMINAL_FLOW,u[:-1]))
                assert np.array_equal(held,late|src["planner_failed"]|np.r_[False,failed[:-1]])
                assert np.array_equal(late,src["latency"]>DEADLINE)
                assert np.array_equal(u,np.where(held[:,None],last,np.clip(src["raw_u"],plant.FLOW_LOW,plant.FLOW_HIGH)))
            Q=np.diag([.001**2]*10+[MOTION_SD**2 if arm=="adaptive" else 1e-6])
            indices=[0,1,n//2,n-1]
            switches=np.flatnonzero(np.diff(tape["phase"]))+1
            if len(switches): indices += [switches[0]-1,switches[0],switches[-1]]
            pred=src["pred"]
            for t in set(indices):
                state=simulator.step(x[t],u[t],tape["kla"][t])+tape["process"][t]
                agree("state",state,x[t+1],1e-10)
                if failed[t]: continue
                if t:
                    mu,cov=m[t-1],P[t-1]
                else:
                    prior=PRIOR.copy()
                    if arm!="adaptive": prior[-1,-1]=1e-6
                    mu,cov,_=dense_update(np.r_[plant.nominal_profile(),np.log(5.)],prior,np.eye(11),
                                         np.zeros(11),np.zeros((11,11)),H,R,tape["y0"])
                point=mu.copy()
                if arm!="adaptive": point[-1]=np.log(5. if arm=="frozen" else tape["kla"][t])
                A,B,b=linearize(point,u[t]); off=b+B@u[t]
                if arm!="adaptive":
                    off=off+A[:,-1]*point[-1]
                    A=A.copy(); A[:,-1]=0.; off[-1]=point[-1]
                agree("pre_observation",H@(A@mu+off),pred[t],1e-8)
                after,c,_=dense_update(mu,cov,A,off,Q,H,R,obs[t])
                agree("mean",after,m[t],1e-8)
                agree("cov",c,P[t],1e-8)
                ncheck+=1
            if cond=="changing" and mode=="tape" and arm=="adaptive":
                changed=tape["kla"]!=5.
                truth=x[1:,HIDDEN][changed]; recorded=m[:,:10][:,HIDDEN][changed]
                primary[seed,lib]=float(np.linalg.norm(truth-recorded)/np.sqrt(truth.size))
                full_means[seed,lib]=m
    deltas=np.array([primary[s,"cffg"]-primary[s,"rxinfer"] for s in seal["seeds"]])
    rng=np.random.default_rng(89021)
    sampled=rng.integers(0,len(deltas),(10000,len(deltas)))
    ci=np.quantile(deltas[sampled].mean(axis=1),[.025,.975])
    scored=read(root/("analysis.json" if stage=="heldout" else "development/analysis.json"))
    if not scored: raise ValueError("Scored comparison required for independent audit")
    np.testing.assert_allclose(scored["primary"]["ci95"],ci,atol=1e-15,rtol=0)
    np.testing.assert_allclose(scored["primary"]["mean_difference"],deltas.mean(),atol=1e-15,rtol=0)
    max_belief_difference=max(float(np.max(abs(full_means[s,"cffg"]-full_means[s,"rxinfer"]))) for s in seal["seeds"])
    return dict(passed=True,stage=stage,archives=len(seal["runs"]),total_updates=total,full_sensor_replays=total,
                audited_commands_held=held_total,sampled_dense_and_transition_checks=ncheck,max_absolute_discrepancy=check_max,
                max_between_library_adaptive_tape_mean_difference=max_belief_difference,
                independent_effect=float(deltas.mean()),ci95=ci.tolist())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=DEFAULT_ROOT)
    p.add_argument("--stage",choices=("heldout","development"),required=True); args=p.parse_args()
    configure()
    result=audit(args.output,args.stage)
    save(args.output/("audit.json" if args.stage=="heldout" else "development-audit.json"),result)
    print(result)

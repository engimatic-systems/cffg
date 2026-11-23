"""Check the actual adapter interfaces, a cold real action, causal updates and affine oracles."""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from time import perf_counter,time

from study import DEFAULT_ROOT, HORIZON, LIBRARIES, configure, save


def run(lib):
    import numpy as np
    import plant
    import episode as e
    from cffg_native_check import fixture, dense_plan, dense_update
    from dynamics import linearize

    tape=plant.design(17001,600,"changing")
    start=perf_counter()
    backend=importlib.import_module(lib+"_native").Backend()
    setup=perf_counter()-start
    try:
        nominal=np.r_[plant.nominal_profile(),np.log(5.)]
        I=np.eye(11)
        filtered=backend.update(nominal,e.PRIOR*.5,I,np.zeros(11),e.PRIOR*.5,e.H,e.R,tape["y0"])
        Q=np.diag([.001**2]*10+[.006**2])
        first=e.plan(backend,*filtered,e.NOMINAL_U,tape["target"][:HORIZON],None,Q)
        clipped=np.clip(first,plant.FLOW_LOW,plant.FLOW_HIGH)
        cold=time()-float(os.environ["EXTRACTION_LAUNCH"])
        uargs,pargs=fixture()
        errors={}
        for missing in (False,True):
            args=(*uargs[:-1],None) if missing else uargs
            mean,cov=backend.update(*args)
            rm,rc,nll=dense_update(*args)
            evidence=backend.last_innovation_nll
            if evidence is None and missing: evidence=0.
            errors["missing" if missing else "observed"]=dict(mean=float(np.max(abs(mean-rm))),
                cov=float(np.max(abs(cov-rc))),nll=float(abs(evidence-nll)))
            np.testing.assert_allclose(mean,rm,rtol=1e-8,atol=1e-8)
            np.testing.assert_allclose(cov,rc,rtol=1e-8,atol=1e-9)
            np.testing.assert_allclose(evidence,nll,atol=1e-7)
        oracle=dense_plan(*pargs)
        command=backend.plan(*pargs)
        errors["first_flow"]=float(np.max(abs(command-oracle)))
        np.testing.assert_allclose(command,oracle,atol=1e-6,rtol=1e-9)
        data=dict(q=Q,flow=tape["flows"][:8],noise=tape["noise"][:8])
        native=plant.Plant()
        x=tape["initial"].copy()
        belief=filtered
        snapshots=[]
        before=[]; after=[]
        for t,u in enumerate(data["flow"]):
            A,B,b=linearize(belief[0],u)
            offset=B@u+b
            x=native.step(x,u,5.)+tape["process"][t]
            y=plant.observe(x,tape["noise"][t])
            nxt=backend.update(*belief,A,offset,Q,e.H,e.R,y)
            # Foreign goals can change a plan but must never change the observation-only posterior.
            if t==2:
                a=e.plan(backend,*belief,u,np.full(HORIZON,.3),None,Q)
                z=e.plan(backend,*belief,u,np.full(HORIZON,.4),None,Q)
                if np.max(abs(a-z))<1e-3: raise ValueError("Planning does not respond to goals")
            again=backend.update(*belief,A,offset,Q,e.H,e.R,y)
            for left,right in zip(nxt,again,strict=True): np.testing.assert_allclose(left,right,rtol=0,atol=1e-11)
            if t==3:
                snapshots=[v.copy() for v in nxt]
            if t>3: before.append((A,offset,y))
            after.append([v.copy() for v in nxt]); belief=nxt
        for j,(A,offset,y) in enumerate(before,start=4):
            snapshots=backend.update(*snapshots,A,offset,Q,e.H,e.R,y)
            for left,right in zip(snapshots,after[j],strict=True): np.testing.assert_allclose(left,right,rtol=0,atol=1e-11)
        return dict(passed=True,library=lib,backend_setup_s=setup,cold_real_action_s=cold,
                    cold_raw=first.tolist(),cold_applied=clipped.tolist(),dense_oracle_errors=errors,
                    reset_replay=True,goal_isolation=True,both_flows_live=True)
    finally:
        if hasattr(backend,"close"): backend.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("library",choices=LIBRARIES)
    p.add_argument("--output",type=Path,default=DEFAULT_ROOT)
    args=p.parse_args()
    configure()
    record=run(args.library)
    save(args.output/"qual"/(args.library+".json"),record)
    print(record,flush=True)

"""Observation-only filter and separate goal-conditioned controller for a single episode."""

from __future__ import annotations

import json
import os
import subprocess
from time import perf_counter

import numpy as np

import dynamics
import plant
from study import DEADLINE, HORIZON, MOTION_SD

H = np.eye(11)[[8,1]]
C = H[:1]
R = np.diag(np.array([.002,.003])**2)
U_COV = np.diag([15.**2, 8.**2])
R_GOAL = np.array([[.012**2]])
NOMINAL_U = np.array([75.,30.])
U0 = np.tile(NOMINAL_U, (HORIZON, 1))
PRIOR = np.diag([.035**2]*10 + [.1**2])
HIDDEN = np.array([0,2,3,4,5,6,7,9])


def transition(m, u, known):
    point = m if known is None else np.r_[m[:10], known]
    A,B,b = dynamics.linearize(point,u)
    if known is not None:
        # This diagnostic coordinate is independent of the plant in both known-Kla arms.
        b = b + A[:,10] * known
        A = A.copy()
        A[:,10] = 0
        b[10] = known
    return A,B,b


def ps_memory():
    tree = subprocess.check_output(["ps","-axo","pid=,ppid=,rss="],text=True)
    rows = [tuple(map(int, r.split())) for r in tree.splitlines() if r.strip()]
    ids = {os.getpid()}
    while True:
        children = {pid for pid, parent, _ in rows if parent in ids}
        if children <= ids:
            break
        ids.update(children)
    return sum(rss for pid, _, rss in rows if pid in ids)/1024


def proper(m, P):
    if np.shape(m)!=(11,) or np.shape(P)!=(11,11) or not np.isfinite(m).all() or not np.isfinite(P).all():
        raise ValueError("Nonfinite or wrong-shaped native filtered belief")
    if not np.allclose(P,P.T,rtol=1e-8,atol=1e-12):
        raise ValueError("Asymmetric native filtered covariance")
    np.linalg.cholesky(P)
    return m,P


def plan(backend,m,P,old_u,targets,known,Q):
    # Forecast along the last applied flow. The costs retain the fixed nominal flow prior.
    x=m
    As=[]; Bs=[]; bs=[]
    for _ in targets:
        A,B,b=transition(x,old_u,known)
        As.append(A); Bs.append(B); bs.append(b)
        x=A@x+B@old_u+b
    result=np.asarray(backend.plan(m,P,np.stack(As),np.stack(bs),np.stack(Bs),
                                   np.broadcast_to(Q,(len(targets),11,11)),C,
                                   targets[:,None],R_GOAL,U0[:len(targets)],U_COV))
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError("Invalid native two-flow plan")
    return result


def simulate(backend, simulator, tape, arm, mode):
    n=len(tape["kla"])
    x=np.empty((n+1,10)); x[0]=tape["initial"]
    u=np.empty((n,2)); raw_u=np.empty_like(u)
    obs=np.empty((n,2)); pred=np.full_like(obs,np.nan)
    pred_cov=np.full((n,2,2),np.nan)
    means=np.empty((n,11)); covs=np.empty((n,11,11))
    failed=np.zeros(n,bool); planner_failed=failed.copy(); held=failed.copy(); late=failed.copy()
    t_filter=np.zeros(n); t_plan=np.zeros(n); latency=np.zeros(n)
    times=[]; diagnostics=[]
    m=np.r_[plant.nominal_profile(),np.log(5.)]
    P=PRIOR.copy()
    Q=np.diag([.001**2]*10+[MOTION_SD**2 if arm=="adaptive" else 1e-6])
    if arm!="adaptive":
        P[-1,-1]=1e-6
    started=perf_counter()
    # Sum of these two independent Gaussians is the declared reset prior exactly.
    m,P=proper(*backend.update(m,P*.5,np.eye(11),np.zeros(11),P*.5,H,R,tape["y0"]))
    previous_update=perf_counter()-started
    last=NOMINAL_U.copy()
    targets=np.pad(tape["target"], (0,HORIZON-1),mode="edge")
    for t in range(n):
        if t % max(1,n//16)==0:
            times.append((t,ps_memory()))
        known=(None if arm=="adaptive" else
               float(np.log(tape["kla"][t])) if arm=="informed" else float(np.log(5.)))
        if mode=="control":
            tick=perf_counter()
            try:
                raw=plan(backend,m,P,last,targets[t:t+HORIZON],known,Q)
                proposed=np.clip(raw,plant.FLOW_LOW,plant.FLOW_HIGH)
            except (ValueError, ArithmeticError, RuntimeError, np.linalg.LinAlgError) as exc:
                raw=last.copy(); proposed=last; planner_failed[t]=True
                if len(diagnostics)<16: diagnostics.append([t,"plan",str(exc)])
            t_plan[t]=perf_counter()-tick
            latency[t]=previous_update+t_plan[t]
            late[t]=latency[t]>DEADLINE
            held[t]=late[t] or planner_failed[t] or (t>0 and failed[t-1])
            chosen=last if held[t] else proposed
        else:
            chosen=raw=tape["flows"][t]
        u[t]=last=np.array(chosen,copy=True); raw_u[t]=raw
        # Only the simulator and the privileged informed arm see the true coefficient.
        x[t+1]=simulator.step(x[t],last,tape["kla"][t])+tape["process"][t]
        if not np.isfinite(x[t+1]).all() or (x[t+1]<0).any():
            raise ValueError(f"Plant left physical domain at transition {t}")
        obs[t]=plant.observe(x[t+1],tape["noise"][t])
        tick=perf_counter()
        prior_m,prior_P=m,P
        try:
            A,B,b=transition(m,last,known)
            offset=B@last+b
            prior_m,prior_P=proper(A@m+offset,A@P@A.T+Q)
            pred[t]=H@prior_m
            pred_cov[t]=H@prior_P@H.T+R
            m,P=proper(*backend.update(m,P,A,offset,Q,H,R,obs[t]))
        except (ValueError, ArithmeticError, RuntimeError, np.linalg.LinAlgError) as exc:
            # Python assigns the fallback pair only after both prediction checks succeed.
            m,P=prior_m,prior_P; failed[t]=True
            if len(diagnostics)<16: diagnostics.append([t,"update",str(exc)])
        t_filter[t]=previous_update=perf_counter()-tick
        if mode=="tape": latency[t]=previous_update
        means[t]=m; covs[t]=P
    times.append((n,ps_memory()))
    return dict(x=x,u=u,raw_u=raw_u,m=means,P=covs,obs=obs,pred=pred,pred_cov=pred_cov,
                kla=tape["kla"],phase=tape["phase"],event=tape["event"],target=tape["target"],
                y0=tape["y0"],failed=failed|planner_failed,filter_failed=failed,planner_failed=planner_failed,
                held=held,late=late,latency=latency,t_filter=t_filter,t_plan=t_plan,
                rss_samples=np.asarray(times),diagnostics=np.asarray(json.dumps(diagnostics)))

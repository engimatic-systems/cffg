---
id: cpt-ydse
status: in_progress
deps: [cfp-okq2]
links: [cff-wt1c]
created: 2026-11-22T21:08:39Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym, benchmark]
---
# Time four inference libraries on short and sustained PCGym reactor control

## Background in Plain English — WHY
Cooper needs an estimate for a four-library comparison. The previous replay measured CFFG. The slowest peer and the cost of long forecasts govern the end-to-end runtime.

## Outcomes / Acceptance Criteria — WHAT
1. Run CFFG, RxInfer, NumPyro and GTSAM on the same development seed in fresh processes, for 59 and 599 uninterrupted control decisions apiece, with one second for each command after preparation.
2. Check the first-action mean, covariance and cross-time covariance against an independent frozen affine Gaussian reference. Record numerical exceptions and missed decisions.
3. Time a batch of open-loop forecasts through step 60 and compare predictions with the simulated plant. Publish setup, latency distributions, episode costs, any failed qualifications, and an updated 30-seed compute projection.

## Methodological Suggestions — HOW
Use CPU float64 and a single numerical thread, one measured arm at a time. Plant dynamics, intervals, physical parameters, data and goal preview follow the retained CSTR. Long episodes extend physical duration. Architecture path: isolated benchmark consumers of runtime numerical/status and Gaussian readouts. Engine, JIT, vmap, serialization, model math and production API changes are N/A. Exercise real public controllers with independent native libraries.

## Future work
cff-wt1c owns the scored 30-seed study. Parameter drift and high-dimensional actuator bounds have independent qualifications.

## Definitions
PCGym simulates process control. CSTR is its continuously stirred tank reactor. A rollout follows a trajectory over multiple decisions. A development seed is for testing implementations and timing rather than drawing comparative conclusions.


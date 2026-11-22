---
id: cpt-ydse
status: closed
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


## Notes

**2026-11-22T21:21:40Z**

Completed the isolated four-library PCGym pilot at P4/H15 on the Apple M2 Max. All 12 frozen Gaussian qualifications passed. All 2,632 actions in eight seed-0 episodes succeeded within the one-second warmed deadline. Long-run returns were -8.597292 for CFFG and RxInfer and -9.265972 for NumPyro and GTSAM. Warm long-run decision medians in ms were 7.10, 0.43, 5.72 and 0.61 respectively. Launch to warmed controller was 3.29, 13.93, 9.17 and 2.21 seconds. All 48 filtering origins passed with 512 matched particle draws per origin and four forecast leads. At lead 60 the CFFG RMSE was 0.000627 mol/L and 0.00728 K against persistence at 0.02196 mol/L and 2.58 K.

Independent library work established that native RxInfer streaming linearization reproduces the directed solution on archived nonlinear windows, native sparse GTSAM LM and a C++ RK4/AD factor pass the affine and transition checks, and NumPyro joint Laplace passes the affine gates where the tested full-covariance SVI budget fails. The separate forecast worker matches a scalar RK4 and interval-scoring reference and rejects improper posteriors. Evidence is under ~/agent-share/pcgym-four-library-timing/{rxinfer,gtsam,numpyro-development.json,measured} and tools/pcgym_timing/forecast_check.py. These results apply to the tested implementations, model and development seeds.

Wall time for all eight control processes and four forecast batches was 102.40 s. At these costs a serial 30-seed repeat requires 35.4 min for both control lengths and 15.8 min for twelve forecast origins per seed per arm. Duplicating control for a second deadline while sharing forecast results projects to 86.6 min. Offline reference sampling, varying physical regimes, parameter learning and installation are separate.

Report and full data: ~/agent-share/pcgym-four-library-timing/index.html and summary.json. Checked all 16 qualification/measurement manifests against the sources used by each arm, and 15 report links via HTTP. Ran tools/pcgym_timing/run_measured.sh through every timed process followed by tools/pcgym_timing/summarize.py --output ~/agent-share/pcgym-four-library-timing. The first offline analysis used ideal decimal observation bounds; the simulator stores Gym Box bounds in float32. Reconstructed the sensor noise with the actual bounds and verified it directly against the saved seed on every input, within 1.85e-16. No trajectories or score selections changed. The final offline summary and report pass.

Validation also included the standalone forecast_check, Python compilation, shell syntax, tk dep cycle and git diff --check. This ticket only adds opt-in research drivers and external report analysis. Production sources, shared dependency manifests, pytest discovery, test execution and its pass/fail criteria are untouched. These focused checks and actual benchmark runs are the isolated-tooling gate. The scored multi-seed study remains open under cff-wt1c.

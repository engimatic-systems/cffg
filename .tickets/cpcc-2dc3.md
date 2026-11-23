---
id: cpcc-2dc3
status: in_progress
deps: [cpc-4ntg]
links: [cff-wt1c]
created: 2026-11-23T01:05:08Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-xa6z
---
# Compare four planners on the reactor with online heat-transfer learning

## Background in Plain English — WHY
A gentler heat-transfer learner passed the stationary tracking allowance with the CFFG controller. We now need to know how the same estimate performs when each library plans actions from it. The fixed-model comparison could not answer this.

## Outcomes / Acceptance Criteria — WHAT
1. Freeze the same learner, model, information available to each controller, actuator limits and one-second decision deadline before opening fresh seeds.
2. Run CFFG, RxInfer, NumPyro and GTSAM for ten paired seeds (6601–6610). For each library run all 599 decisions in both the stationary and hidden-change conditions with the fixed, original (q=0.012), selected (q=0.003) and informed coefficient.
3. Compare selected-controller concentration tracking during the hidden drop. Primary measure is the mean paired per-seed difference in RMSE, CFFG minus peer. Report descriptive 95% whole-seed paired intervals, the absolute magnitudes, and all ten seeds. Report own stationary cost against a 1% tolerance and changed-period advantage of learning over fixed. Include original versus selected recovery, applied actions held on failure or timeout, and setup versus warmed decision latency.
4. Verify all adapters against affine ground truth and verify live parameter delivery against fresh models on development inputs. Preserve source and dependency digests, raw trajectories and a readable HTML report.

## Methodological Suggestions — HOW
Reuse the existing plant, estimator, masks, noise stream, latency boundary and native inference methods. Initialize each new episode from the same starting estimate. Shared code for the estimator isolates the choice of planner. Use the comparison machinery from cpc-4ntg and the four-library timing study. Architecture change-path index, runtime numerical or status invariant row, benchmark consumers only. Engine rules, objective, serialization, solver schedules, JIT/vmap production paths and shared dependencies are N/A.

## Future work
cff-wt1c owns broader comparisons. Independent native parameter-learning algorithms and response to other change schedules would require distinct experiments.

## Definitions
UA measures heat transfer. q is the assumed per-transition standard deviation of log UA. Root mean square error (RMSE) measures the size of concentration deviations from their target. Informed controllers receive true UA before acting. A paired seed supplies the same noise draws to every controller.


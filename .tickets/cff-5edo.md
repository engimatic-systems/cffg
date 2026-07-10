---
id: cff-5edo
status: open
deps: []
links: []
created: 2026-07-10T03:31:09Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Long-rollout planning: cost and stability beyond ~20 slices

Receding-horizon plan graphs measured at slices H in {5..25} on a 60-state aliased-corridor stack: success flat (0.786, beating a QMDP baseline at 0.643) through H=20 but DROPS to 0.571 at H=25 while per-env-step planning time grows linearly (0.61s at H=5 to 2.47s at H=25, sweeps dominated). Two threads: (1) diagnose the H=25 degradation — likely GFE outer-coordination limit-cycling at long chains under a fixed sweep budget (damping schedule? per-slice convergence detection?); (2) incremental replanning — successive env steps re-solve a nearly identical graph from scratch; warm-starting messages from the previous step's shifted solution should cut steady-state cost substantially.

## Acceptance Criteria

H=25 success within noise of H=20 on the same fixture (or a documented LIMITATIONS entry explaining why not); warm-started replanning >=3x faster than cold at H=20 with identical action choices.


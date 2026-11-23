---
id: cff-5o7r
status: closed
deps: []
links: []
created: 2026-09-25T08:32:03Z
type: chore
priority: 2
assignee: coopwilliams
parent: cff-xa6z
---
# Keep checked-collapse validation within the short-suite runtime budget

## Background in Plain English

The final cff-fkuj suite passed 2,535 tests but took 1,226.099 measured seconds, above the 1,200-second default-suite budget. Four repeated behavioral/calibration checks account for 30.054 seconds. They remain useful opt-in coverage; numerical correctness and hostile-input checks stay in the default suite.

## Outcomes / Acceptance Criteria

1. Move only these unchanged test bodies to the slow suite: estimator calibration/prediction (9.203s), repeated glucose benchmark determinism (6.658s), rolling-goal pending-checkpoint replay (7.902s), and full-engine unknown-dynamics preference/ablation (6.291s).
2. Retain default independent message/energy oracles, sparse/float32/large-scale checks, current/legacy checkpoint and atomic rejection cases, direct goal rollover algebra, and all new collapse tests.
3. Record measured 1,226.099s and estimated remaining 1,196.045s for 2,531 selected tests. Validate collection and reuse the passing full result under the project rule; do not rerun solely for markers.

## Methodological Suggestions

Add only pytest slow markers and the needed pytest import. Another sound budget adjustment is acceptable if these outcomes hold. This changes test selection, not numerical behavior; the full gate already passed before the marker-only adjustment.

## Future work

- cff-fkuj consumes this validation adjustment before its PR is published.
- cff-xa6z continues the PCGym implementation chain.

## Definitions

An ablation removes one modeled effect for comparison. Calibration measures approximate runtime. A checkpoint saves agent state. Slow tests are opt-in pytest cases. Float32 is 32-bit floating-point arithmetic; sparse support includes exact zero probability.



## Notes

**2026-09-25T08:32:36Z**

Completed marker-only adjustment: four unchanged behavioral/calibration cases moved to slow, totaling30.054s in the saved passing run. Measured1226.099s; estimated remaining1196.045s for2531 tests. Collection24 selected/6 deselected across affected files. Default retains direct FE/message/goal-rollover oracles, exact support, f32/uncomfortable scale, current/legacy replay elsewhere, and all27 new collapse tests. tk dep cycle and git diff --check pass. Reused passing2535-test gate; no second full run solely for markers.

---
id: cff-aqpz
status: closed
deps: []
links: []
created: 2026-09-25T10:40:44Z
type: chore
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [testing, pcgym]
---
# Keep the short suite within budget after stable mixture evidence coverage

## Background in Plain English

The final cff-d8jj suite passed 2595 tests but took 1272.832 seconds, above the 1200-second default budget. Saved per-test timings identify repeated checkpoint and example workflows that can remain available in the opt-in slow suite. New numerical tests must remain in the default suite.

## Outcomes / Acceptance Criteria

1. Unchanged long checkpoint and example replays move to the opt-in slow suite without deleting their assertions.
2. Numerical rule, solver, float32, sparse-support and gradient coverage remains selected by default.
3. Saved per-test timings justify an estimated remaining runtime below 1200 seconds; the measured total and estimate are recorded separately.
4. Collection and changed-artifact checks pass. The existing full passing result is reused without a second run solely for marker changes.

## Methodological Suggestions

Another sound selection is acceptable. Inspect saved dashboard timings and retain representative ordinary checkpoint, learning and float32 example coverage. This is test selection under the explicit suite-budget exception, not an isolated tooling claim. Engine execution pairs are unchanged and N/A.

## Future work

cff-d8jj publishes the stable evidence repair using the passing final gate and this bounded selection adjustment. No further behavior change is identified here.

## Definitions

The default short suite is the tests run before a pull request. Opt-in slow tests run only when explicitly selected. A replay repeats a complete workflow to check continuation or persistence. Float32 is single-precision floating-point arithmetic. A gradient measures sensitivity to a model input.

## Notes

**2026-09-25T10:41:30Z**

Measured final gate:2595 passed,103 deselected; pytest1257.68s, dashboard wall1272.832s. Fifteen unchanged workflow cases moved to slow: four Gaussian continuation combinations15.258s; two sparse continuation replays9.451s; functional configuration-mode replay9.937s; functional nonfinite-state checkpoint replay11.387s; legacy/base restore replay5.787s; three constraint migration rejection replays14.989s; goal-checkpoint corruption replay5.362s; float64 newcomer subprocess4.386s; legacy constraint-envelope replay5.758s. Total82.315s; estimated default wall1190.517s. Default retains float32 newcomer execution, current constraint reconstruction, ordinary schema/atomicity checks, direct sparse learning/readout tests, all37 new evidence checks and solver/rank/collapse regressions. Changed collection45selected48deselected; numerical selection105selected2preexisting slow. git diff --check and tk dep cycle pass. Reused passing gate under explicit budget exception; no numerical edits or second suite.

**2026-09-25T10:41:30Z**

Dependency snapshot from tk dep tree --full, tk ready and tk blocked. Legend: prerequisite --> dependent.

[x] cff-u9un --+
               +--> cff-d8jj --> cff-wt1c --> cff-xa6z
[ready] cff-aqpz+

Parallel ready work: cff-7kpx. cff-c5qf awaits the separate partial-goal scope decision. Budget adjustment is the immediate publication blocker for d8jj.

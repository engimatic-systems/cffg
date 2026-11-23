---
id: cff-xrut
status: closed
deps: []
links: []
created: 2026-09-30T16:34:05Z
type: chore
priority: 2
assignee: coopwilliams
parent: cff-fc2y
tags: [tests, runtime]
---
# Keep the default gate under budget by deferring repeated integration replays

## Background in Plain English

The passing input-readiness gate took 1,254.848 seconds, exceeding the 1,200-second default-suite budget. Retain the passing result and defer selected expensive integration replays while keeping direct numerical robustness and representative execution coverage in the default suite.

## Outcomes / Acceptance Criteria

1. Select deferred cases using the completed run timings; estimate the remaining suite below 1,200 seconds.
2. Keep float32, sparse support, uncomfortable-scale and direct readiness tests in the default suite.
3. Mark deferred cases slow, leaving their assertions unchanged and available for focused opt-in runs.
4. Record measured and estimated timings and reuse the passing result without a second unchanged full gate.

## Methodological Suggestions

Another sound selection is acceptable. Use the Runtime numerical/status architecture row; only test selection changes. All production execution and serialization pairs are unchanged. Validate selected/deselected collection and git diff --check. Do not rerun the full suite solely for marker edits.

## Future work

cff-36x7 must explicitly run any deferred nonlinear integration cases relevant to its behavior changes. No additional work is identified.

## Definitions

The default gate is the pytest selection excluding slow tests. An integration replay repeats a complete runtime workflow. Opt-in means running slow tests explicitly.


## Notes

**2026-09-30T16:35:30Z**

Completed passing gate: 2,610 passed, 135 deselected; wall 1,254.848 seconds. Marker-only deferral removes 11 replay cases totaling 66.246 measured seconds: mutually changing nonlinear site-root integrations (all four cases, 33.054); curved sensor flood replay (two methods, 9.653); neighboring hybrid auto replay (both dtypes, 13.209); large mixture rank f64 replay (6.335; f32 scale retained); Gaussian bundle schedule resume replay (two schedules, 3.995). Estimated remaining wall: 1,188.602 seconds. Representative curved auto, hybrid flood, affine energy, runtime vmap, float32 scale, sparse/null/invalid and ordinary checkpoint tests remain selected. Marker assertions unchanged. Focused collection: 61/86 selected, 25 deselected across four changed files. git diff --check and tk dep cycle pass. Reused full-suite result as required by AGENTS and Cooper; no repeat gate for these marker-only edits. Legend prerequisite --> dependent: [x]7kpx + [x]xrut --> [ ]fc2y --> [ ]36x7 --> [ ]qw7x --> [ ]toqa --> [ ]wt1c --> [ ]xa6z. Immediate blocker is readiness completion; independent ready roots ees0/qu2j/nd1q remain parallel.

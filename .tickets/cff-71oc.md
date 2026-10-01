---
id: cff-71oc
status: closed
deps: []
links: []
created: 2026-09-30T17:47:28Z
type: chore
priority: 2
assignee: coopwilliams
parent: cff-36x7
tags: [tests, runtime]
---
# Restore default-suite headroom after bounded startup validation

## Background in Plain English

The bounded-startup gate ran in 1,234.901 seconds. Its only failure was a stale static ownership assertion, while 2,627 tests passed. Restore the 1,200-second default budget using measured timings before the corrected gate.

## Outcomes / Acceptance Criteria

1. Defer measured integration replays whose core correctness remains covered in the default suite.
2. Retain readiness, callback guards, sparse support, float32, uncomfortable scale and independent dense affine evidence checks.
3. Preserve deferred assertions under the slow marker and record measured savings and estimated runtime.
4. Do not launch another full gate solely for marker changes after a passing result.

## Methodological Suggestions

Another sound selection is acceptable. Use Runtime numerical/status architecture row; only selection changes. Production execution and checkpoint pairs remain unchanged. Validate collection and git diff --check; tk dep cycle must pass. The unrelated stale ownership assertion is handled in cff-36x7 before its corrected final gate.

## Future work

Relevant deferred cases remain available for explicit focused validation in downstream cff-qw7x and other changes. No new feature work is identified.

## Definitions

Headroom means time below the default-suite limit. A replay repeats a complete inference workflow; opt-in slow tests run when requested explicitly.


## Notes

**2026-09-30T17:49:00Z**

Measured wall 1,234.901 seconds. Defer nine passed replay cases totaling 56.121 seconds: nonlinear affine engine parity (four f64 observed/unobserved/method cases, 20.755); f64 mixture specialized vmap replay (7.669; f32 remains); 2D switching-KF collapse parity (7.586; 1D parity and 2D approximation-gap check remain); hybrid direct-rule message replay (7.959; independent ledger/planning and local rule checks remain); 60-sweep hybrid vmap replay (6.253; direct batching/failure checks remain); K1 mixture-statistics-to-linear replay (5.899; independent 2D node-statistics oracle remains). Estimated wall 1,178.780 seconds. Default retains all new readiness/admission/callback/accounting checks, f32 dense affine evidence, both nonlinear methods local energy/rule coverage, sparse support and hostile scale. Assertions unchanged; collection and dependency-cycle checks follow. cff-36x7 corrects the stale ownership assertion separately before its final gate. No repeat gate is needed solely for marker changes after a passing result. Legend prerequisite --> dependent: [x]fc2y + [ ]71oc (ready) --> [ ]36x7 --> [ ]qw7x --> [ ]toqa --> [ ]wt1c --> [ ]xa6z; ees0/qu2j/nd1q are independent ready roots.

**2026-09-30T17:49:40Z**

Changed-file collection confirms 71/87 selected, 16 deselected. git diff --check and tk dep cycle pass. Production execution/serialization unchanged. Marker-selection work complete; cff-36x7 owns the corrected full gate.

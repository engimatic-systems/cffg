---
id: cff-gvwc
status: in_progress
deps: []
links: []
created: 2026-09-15T16:45:09Z
type: chore
priority: 2
assignee: coopwilliams
parent: cff-o7hs
---
# Keep cohort and large Gaussian scale regressions opt-in

## Background in Plain English — WHY

The current branch records a 1,233.077-second short-suite run, exceeding the 1,200-second cap. The full ten-adult, 24-hour glucose cohort and dimension-100 Gaussian batching regressions are long behavioral/scale checks. Keep them available under the opt-in slow marker so default validation has room for new coverage.

## Outcomes / Acceptance Criteria — WHAT

1. Those long cohort and Gaussian scale cases require explicit slow-test selection; smaller Gaussian correctness and robustness cases remain in the default suite.
2. The selected large Gaussian tests have focused passing evidence for the endpoint-schema change.
3. The dashboard-backed short suite passes; if its measured wall time exceeds 1,200 seconds, per-test timings justify further removals or slow-test selection until estimated remaining runtime is under budget. Record measured and estimated totals and reuse that passing result without rerunning solely for pruning.

## Methodological Suggestions — HOW

- Use pytest.mark.slow and collect-only checks for selection. Recorded shared-dashboard timings for matching test names are indicative only; the fresh final suite establishes actual wall time.
- Test-selection changes require the full gate. Host/JIT and single-instance/vmap are covered by the retained opt-in scale cases; flooding/scheduled and current/legacy decoding are N/A to marker-only edits.

## Future work

- cff-o7hs and cff-1o72 must retain the twenty-minute cap as they add tests.

## Definitions

- Cohort: the complete set of simulated patients in the behavioral check.


## Notes

**2026-09-15T16:51:37Z**

AGENTS.md now uses Cooper's measured-run protocol: let the suite finish; if it passes over twenty minutes, inspect its per-test timings, remove or mark slow enough cases to estimate a total below 1,200 seconds, report measured and estimated totals separately, and do not rerun solely for pruning. The glucose cohort and dimension-100 Gaussian cases are marked slow; collect-only validation selects 25 of 28 tests and deselects exactly those three cases. The two Gaussian scale variants passed in the 75-test focused run under cff-2dzo before their marker-only change. Final measured short-suite evidence remains due at the parent implementation head. Handoff: keep in_progress until that measured result and any required pruning are recorded.

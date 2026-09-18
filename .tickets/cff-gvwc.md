---
id: cff-gvwc
status: closed
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

**2026-09-15T17:58:09Z**

Measured current suite wall1305.784 seconds. Per-case timings recovered from the generated dashboard identify12 additional opt-in cases totaling184.805 seconds: the39.402-second NASCAR globalization regression,35.970-second attached nonlinear scan-island regression, and10 repetitive functional-checkpoint corruption variants. Missing PRNG key, non-finite state, and invalid importance weights remain in the short selection; all demoted cases remain available with -m slow and passed the completed run. Estimated remaining wall1120.979 seconds (18m40.979s), below1200. Collect-only confirms12 additional deselections (61/74 selected across touched files,13 deselected including one pre-existing slow case). No rerun is needed solely for this pruning; the independent topology fixture failure requires a fresh full run before publication.

**2026-09-15T18:27:00Z**

Final measured short-suite wall time is 1113.339 seconds (18 minutes 33.339 seconds), below the 1200-second budget. Pytest reports 2384 passed and 56 deselected in 1099.56 seconds. The prior timing-based estimate was 1120.979 seconds. The fresh run was required by the independent topology fixture correction; there was no run solely to recheck pruning. The dashboard and runtime history are refreshed. No additional budget work remains.

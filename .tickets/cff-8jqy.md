---
id: cff-8jqy
status: closed
deps: []
links: []
created: 2026-09-25T06:52:09Z
type: chore
priority: 2
assignee: coopwilliams
parent: cff-xa6z
---
# Keep the adaptive-diagnostics short suite within its runtime budget

## Background in Plain English

The final cff-d57a suite passed 2,486 tests, but measured wall time was 1,204.745 seconds. The project caps its default suite at 1,200 seconds. A complete functional-checkpoint manifest check took 7.721 seconds; retaining it as opt-in coverage brings the estimated default runtime to 1,197.024 seconds without changing solver coverage.

## Outcomes / Acceptance Criteria

1. The complete manifest enumeration remains executable in the opt-in slow suite with its assertions unchanged.
2. Default coverage retains current and legacy checkpoint replay, atomic rejection of corrupt state and incompatible mode, and all numerical correctness, derivative, float32, sparse-support and uncomfortable-scale cases.
3. Record the measured and estimated totals and validate test selection. Reuse the already passing full result as explicitly allowed for marker-only budget adjustments.

## Methodological Suggestions

Mark test_functional_checkpoint_has_one_exact_outer_and_runtime_manifest slow. Another sound selection is acceptable if the criteria remain true. This is test selection, not isolated tooling: the full gate has already passed at the pre-marker head. No numerical execution surface changes.

## Future work

- cff-d57a consumes this validation adjustment before publication.
- cff-xa6z continues the PCGym follow-up PR chain.

## Definitions

A manifest enumerates the fields stored in a checkpoint. A checkpoint is saved agent state. Atomic rejection leaves existing state unchanged. The slow suite is opt-in pytest coverage. Float32 is 32-bit floating-point arithmetic; sparse support includes impossible outcomes represented by exact zero probability.


## Notes

**2026-09-25T06:52:27Z**

Completed: only the 7.721-second manifest-enumeration test moved to opt-in slow; assertion body unchanged. Passing saved full run: 2486 passed, 82 deselected, pytest 1189.27 seconds, measured 1204.745 seconds. Estimated default selection after this change: 2485 tests, 1197.024 seconds. Collection confirms 19 selected/20 deselected in test_agent_mu3.py; current/legacy replay, mode mismatch and nonfinite-state atomicity remain selected. tk dep cycle and git diff --check pass. Reused the passing full result; no rerun solely for this marker change.

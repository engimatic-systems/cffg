---
id: cff-rnn1
status: closed
deps: []
links: []
created: 2026-09-25T04:19:20Z
type: chore
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym, testing]
---
# Keep the PCGym exact-inference validation within the short-suite budget

## Background in Plain English

The passing final suite for cff-v6th at 2226bea took 1,233.520 seconds of measured wall time (pytest reported 1,219.12 seconds), exceeding the 1,200-second short-suite budget. All 2,432 selected tests passed. The project protocol requires moving expensive cases to the opt-in suite using the retained timings, without rerunning solely for selection changes.

## Outcomes / Acceptance Criteria

1. The exhaustive tied-neural central-difference cases remain executable under the existing slow marker rather than being deleted or weakened.
2. The default suite retains independent float32 gradient finite differences, tied-parameter derivative accumulation, a complete hybrid directional-gradient oracle, invalid-state rejection, and the new exact/Colony regressions.
3. The ticket records measured removed case time and the resulting estimated short-suite wall time below 1,200 seconds.
4. Collection verifies the changed selection; the passing 2,432-test result is reused and is not misrepresented as a new run of the narrower selection.

## Methodological Suggestions

The dashboard retains per-test timings after its temporary XML report is removed. The complete tied-neural gradient test repeatedly resolves a nonlinear model for every parameter perturbation; its linearization and sigma-point cases took 17.794 and 21.276 seconds. Mark that exhaustive test slow. Total moved time is 39.070 seconds; estimated remaining wall time is 1,194.450 seconds and selected count is 2,430.

This changes test selection only, not assertions or execution code. No architecture change-path row applies. Host/JIT, flooding/scheduled, single/vmap and current/legacy decoding are N/A to marker placement; their existing tests remain unchanged. Use the explicit completed-suite selection-change exception in AGENTS.md, not the isolated-tooling exception. No new numerical run is required for this marker-only change.

## Future work

- cff-v6th uses this budget correction for its validated PR head.
- cff-csmz must run the exhaustive gradient cases explicitly when optimizing physical-mean training.

## Definitions

- **Central difference:** a derivative estimate from resolving the model after positive and negative parameter perturbations.
- **Directional-gradient oracle:** an independent derivative check along a chosen parameter direction.
- **Slow marker:** pytest's existing opt-in classification for long regressions.
- **Wall time:** elapsed real time, including process overhead.

## Notes

**2026-09-25T04:19:55Z**

Completed: marked only the two exhaustive tied-neural finite-difference cases slow. Retained 2,432-test passing result at 2226bea; measured wall 1,233.520s, moved case time 39.070s, estimated narrower suite 1,194.450s / 2,430 selected tests. Default-file collection confirms three retained cases and two deselected. Float32 central differences, tied injection, full hybrid directional gradients, runtime failure checks and all exact/Colony regressions remain in the default suite. git diff --check and tk dep cycle pass. No rerun solely for marker placement, per AGENTS.md.

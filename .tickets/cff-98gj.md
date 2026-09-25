---
id: cff-98gj
status: closed
deps: []
links: []
created: 2026-09-25T05:35:40Z
type: chore
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym, testing]
---
# Restore the short-suite budget after failure transport validation

## Background in Plain English

The passing final suite for cff-fbps at 67ab75f took 1,316.738 seconds of measured wall time. All 2,461 selected tests passed, but the project requires the default suite to stay below 1,200 seconds. Retained dashboard timings identify expensive replay and checkpoint combinations that can remain available as opt-in checks.

## Outcomes / Acceptance Criteria

1. Selected long replay/checkpoint cases use the existing slow marker without deleting tests or weakening assertions.
2. The default suite retains all new failure-transport regressions, numerical and gradient oracles, sparse/float32 checks, representative current and legacy checkpoint restoration, atomic invalid-state rejection, and sparse mixture learning continuation.
3. Measured removed case time and estimated remaining wall time are recorded, with the latter below 1,200 seconds.
4. Collection verifies selection; the passing result is reused under the explicit completed-suite selection-change exception and is not called a fresh narrower-suite run.

## Methodological Suggestions

Use the per-test timings in /Users/ravd/agent-share/cffg-tests.html. Move the large rslds continuation combinations, duplicate constraint-family legacy replay combinations, selected agent configuration/schema replay cases, full optimizer continuation, and the sigma-point neural bundle replay to the slow suite. Keep smaller Gaussian continuation, exactly_k current/legacy reconstruction, nonfinite-state rejection and both nonlinear numerical families in the default suite. The implementer may choose another sound selection.

This changes marker selection only. Architecture row: runtime/status validation coverage; no engine, message, objective or serialization implementation changes. Host/JIT, flooding/scheduled, single/vmap and current/legacy implementation pairs are N/A; retained tests still cover them. Reuse the completed passing suite as AGENTS.md explicitly requires for this adjustment, rather than running a second unchanged suite.

## Future work

- cff-fbps uses this adjustment before publishing its validated PR.
- cff-csmz must explicitly run optimizer continuation and nonlinear bundle replay when changing training.
- cff-iplr must explicitly run particle-runtime checkpoint checks when adding resampling.

## Definitions

- Slow marker: pytest's opt-in test classification.
- Replay: continuing from saved state with the same next result.
- rslds: recurrent switching linear dynamical system, a hybrid continuous/discrete model.
- Sigma-point: a nonlinear Gaussian approximation using deterministic sample locations.
- Oracle: an independent expected numerical result.
- Wall time: real elapsed time including process overhead.
- N/A: not applicable.



## Notes

**2026-09-25T05:36:14Z**

Completed marker-only adjustment: 13 passing cases moved, measured case time 128.476s; measured full wall 1316.738s minus moved time gives estimated 1188.262s / 2448 default cases. Moved totals: two rslds continuation cases 23.851s; four duplicate prepared constraint-family current/legacy combinations 21.411s; sample configuration mismatch 12.499s; missing-key/invalid-weight schema replay 22.674s; action RNG replay 11.326s; changed-budget checkpoint replay 11.498s; optimizer continuation 16.248s; sigma-point neural bundle replay 8.969s. No test body or engine code changed. Collection of the five affected files: 64 selected, 32 deselected; confirms remaining current/legacy exactly_k, all analytic constraint families, sparse component continuation, float32/64 Gaussian continuation, nonfinite checkpoint rejection, particle-invalidity checks, key representation and numerical/gradient oracles. Reused passing 2461-test result at 67ab75f per the completed-suite selection-change exception; not a fresh narrowed-suite run. git diff --check and tk dep cycle pass.

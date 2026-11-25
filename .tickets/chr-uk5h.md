---
id: chr-uk5h
status: closed
deps: []
links: []
created: 2026-11-25T07:24:31Z
type: chore
priority: 2
assignee: coopwilliams
parent: cff-pyy9
---
# Keep the hierarchical runtime short gate within twenty minutes

## Background in Plain English — WHY

The measured default suite took 1355.383 seconds at 8ec7ae2. The project limit is 1200 seconds. Keep numerical and status regressions in the default gate while moving repeated integration configurations to the opt-in suite.

## Outcomes / Acceptance Criteria — WHAT

1. Measured durations put estimated remaining wall time near 1110–1150 seconds.
2. Default coverage retains the new Gaussian precision, runtime statuses, hierarchical and contingent-policy oracles, and existing failure gates.
3. The union of collected default and opt-in test IDs stays identical. The before/after set differences identify every promoted slow case.

## Methodological Suggestions — HOW

Review assertions alongside the measured timing log. Prefer repeat workflows and parameter combinations covered by retained anchors. Walk Runtime numerical or status invariant in the architecture change-path index. Execution, learning, serialization, host/JIT and scheduler pairs are unaffected by marker-only edits.

## Future work

The parent coordinates the fresh gate for its separate compatibility-export correction.

## Definitions

A test ID names one collected pytest case, including its parameters. The short suite selects cases without the slow mark.


## Notes

**2026-11-25T07:28:56Z**

At 8ec7ae2 the short suite had 2843 passes and one compatibility-export failure. Dashboard wall was 1355.383 s (pytest 1339.60 s). All 93 moved IDs passed. Their rounded call/setup/teardown times in ~/agent-share/cffg-hierarchy-validation/short-suite-8ec7ae2.log sum to 211.750 s. Estimated wall with original process overhead is 1143.633 s, leaving 56.367 s against budget. Timing variation and JIT compilation moving to another test can change the actual wall. The parent owns a new run for the independent compatibility repair.

**2026-11-25T07:28:56Z**

Both pytest --collect-only -q selections (-m "not slow" and -m slow, under uv run --frozen --no-sync) were captured before and after. Short 2844 → 2751; slow 242 → 335. Their disjoint union has exactly the original 3086 full test IDs, and short still covers all 149 original files. All 25 files with test changes between 51ae6a7 and 8ec7ae2 are untouched; every new Gaussian tap/precision, hierarchy and checkpoint, contingent plan, ordinary and mixed controller, scalar tolerance, fan-out and transport gate retains its selection. The compatibility-export failure remains selected. The 36 edited files have identical ASTs outside pytest marker decorators/imports. Saved exact moved IDs, per-ID durations and four collection hashes: ~/agent-share/cffg-hierarchy-validation/short-suite-marker-audit.json.

**2026-11-25T07:28:56Z**

Selected existing, previously reviewed parameter and scenario marks from f74543f and d04e090 using current timings. Of 93 moved cases, 61 retain the same test function with other parameters in the default gate. Examples of retained coverage for longer replays: K=1 message/energy and quadrature, RTS smoothing and Kalman likelihood, recurrence pair and rollout oracles, conditional vmap plus failure cases, tied MNIW/PG direct statistical oracles (including BOTH f32 K32/d16 update gates), hybrid/dense root certificates, complete sparse neural gradient, deterministic T-maze and cue behavior, rotated rank and precision tests, three-regime recurrence oracle, exact bipartite matching and all-different marginals. The extended Sudoku, patient outcome, alternate fixed-point schedules and integration comparisons continue to run with -m slow. No files or assertions removed.

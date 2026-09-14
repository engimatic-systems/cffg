---
id: cff-3uq9
status: in_progress
deps: []
links: []
created: 2026-09-14T14:38:15Z
type: task
priority: 1
assignee: coopwilliams
tags: [testing, performance, tooling]
---
# Keep the short suite within twenty minutes without weakening qualification

## Background in Plain English — WHY

Cooper requires the default short suite to take no more than20 minutes and asked the supervisor to implement this directly. The last completed suite at75ac1d2 took3198.621 wall seconds. Five D12 files account for1926.496 reported test seconds; repeated compiled/eager parity and derivative cases are prominent. The old runner allowed unbounded runtime and expanded its quiet wait to4020 seconds. A later run at850db5f was canceled after1538.52 pytest seconds with1577 passes and41 deselections; it is incomplete and does not qualify the D12 repair.

This task enforces a real20-minute budget and reduces the measured costs. It preserves numerical assertions, dtype/support coverage, D12 qualification requirements and recovery thresholds. Cooper explicitly assigned this performance work to the root agent; D12 implementation remains sequential and paused during it.

## Outcomes / Acceptance Criteria — WHAT

1. AGENTS.md states a hard1200-second maximum for the default short suite. The standalone runner enforces the deadline, terminates its own child processes, returns failure on timeout and never publishes a timeout or partial report as a passing current gate. Suggested waits cannot exceed the budget.
2. Per-test and per-file timing evidence identifies the main overhead and distinguishes compilation/setup, execution, wall time and any cache assumptions. Interrupted diagnostics are retained usefully without certifying a complete suite.
3. The actual default suite completes successfully within1200 wall seconds on the development machine after the final relevant edits. Results disclose whether compilation caches were warm or cold and identify remaining limits; changing only the wait budget is insufficient.
4. Runtime reductions preserve required numerical assertions, host/JIT and single/vmap coverage, sparse/invalid inputs, float32, uncomfortable scale, derivatives, FE and learning checks. Do not silently deselect tests, change tolerances or move the registered recovery or A3WZ gates. Any necessary test-tier restructuring retains explicit required qualification commands and is justified before changing selection.
5. Runner deadline/failure/report behavior has focused tests using bounded child processes or controlled clocks, including timeout cleanup and nonzero failures. Cache changes, if any, cannot reuse results for changed code, values or dtype; compiled artifacts are distinct from test results.
6. Applicable architecture paths and duplicated surfaces are recorded before behavior changes. Test discovery/execution/pass interpretation requires the final complete gate; the isolated-tooling exception does not apply. Production math or scope changes are outside this ticket and require separate diagnosis.
7. Atomic commits reference this ticket; canonical docs and tk handoffs record exact commands, source boundaries, timings and limits. D12 ticket cff-8ois stays unqualified until a valid final gate passes with the performance work.

## Methodological Suggestions — HOW

- Start from /tmp/cff-short-suite-baseline.json extracted from the last complete dashboard; its2516-plus uniquely mapped test names give bounded timing evidence without another hour-long profile.
- Profile a few expensive cases for duplicate tracing/compilation and control-flow closures before optimizing. Reuse stable compiled functions and fixtures where they preserve input independence.
- Treat a persistent compilation cache as a transparent optimization with truthful cold/warm evidence, never a substitute for running assertions or honoring code changes.
- Preserve diagnostics from timeouts; a budget violation must remain visible even if earlier cases passed.

## Future work

cff-8ois completes its pending numerical qualification, then cff-y15d and cff-a3wz resume the early joint-solver checkpoint. cff-g7rc owns separate mixed-GFE solver latency work and is not automatically included. The remaining D12/compiler/recovery dependency chain is unchanged.

## Definitions

- Short suite: the default pytest selection, currently not slow.
- JIT/vmap: JAX compiled and batched execution.
- FE: free energy.
- Compilation cache: reusable machine code, not stored passing test results.
- Wall time: elapsed real time including setup and subprocess execution.
- D12/A3WZ: the hybrid mathematical contract and complete joint numerical checkpoint.



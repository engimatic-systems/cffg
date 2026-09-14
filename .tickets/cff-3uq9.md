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

Cooper requires the default short suite to take no more than 20 minutes and asked the supervisor to implement this directly. The last completed suite at 75ac1d2 took 3198.621 wall seconds. Five D12 files account for 1926.496 reported test seconds; repeated compiled/eager parity and derivative cases are prominent. The old runner allowed unbounded runtime and expanded its quiet wait to 4020 seconds. A later run at 850db5f was canceled after 1538.52 pytest seconds with 1577 passes and 41 deselections; it is incomplete and does not qualify the D12 repair.

This task enforces a real 20-minute budget and reduces the measured costs. It preserves numerical assertions, dtype/support coverage, D12 qualification requirements and recovery thresholds. Cooper explicitly assigned this performance work to the root agent; D12 implementation remains sequential and paused during it.

## Outcomes / Acceptance Criteria — WHAT

1. AGENTS.md states a hard 1200-second maximum for the default short suite. The standalone runner enforces the deadline, terminates its own child processes, returns failure on timeout and never publishes a timeout or partial report as a passing current gate. Suggested waits cannot exceed the budget.
2. Per-test and per-file timing evidence identifies the main overhead and distinguishes compilation/setup, execution, wall time and any cache assumptions. Interrupted diagnostics are retained usefully without certifying a complete suite.
3. The actual default suite completes successfully within 1200 wall seconds on the development machine after the final relevant edits. Results disclose whether compilation caches were warm or cold and identify remaining limits; changing only the wait budget is insufficient.
4. Runtime reductions preserve required numerical assertions, host/JIT and single/vmap coverage, sparse/invalid inputs, float32, uncomfortable scale, derivatives, FE and learning checks. Do not silently deselect tests, change tolerances or move the registered recovery or A3WZ gates. Any necessary test-tier restructuring retains explicit required qualification commands and is justified before changing selection.
5. Runner deadline/failure/report behavior has focused tests using bounded child processes or controlled clocks, including timeout cleanup and nonzero failures. Cache changes, if any, cannot reuse results for changed code, values or dtype; compiled artifacts are distinct from test results.
6. Applicable architecture paths and duplicated surfaces are recorded before behavior changes. Test discovery/execution/pass interpretation requires the final complete gate; the isolated-tooling exception does not apply. Production math or scope changes are outside this ticket and require separate diagnosis.
7. Atomic commits reference this ticket; canonical docs and tk handoffs record exact commands, source boundaries, timings and limits. D12 ticket cff-8ois stays unqualified until a valid final gate passes with the performance work.

## Methodological Suggestions — HOW

- Start from /tmp/cff-short-suite-baseline.json extracted from the last complete dashboard; its 2526 uniquely mapped test names give bounded timing evidence without another hour-long profile.
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



## Notes

**2026-09-14T14:44:20Z**

Before implementation: intended published ancestor cff-j191-joint-reference was checked clean and pulled ff-only (unchanged 9c55579); this work branch additionally carries the locally reviewed unpublished 8OIS repair whose gate was interrupted. This avoids changing or publishing an unqualified numerical branch merely to profile it. Work is performed directly by root as Cooper requested; no performance implementer was delegated.

Planning surfaces: runner process lifetime, deadline accounting, report completeness/failure interpretation, measured runtime history and AGENTS policy are applicable. This is test-execution work and not the isolated-tooling exception. If measured compilation overhead requires changes to pure local JAX compilation boundaries, apply the Runtime numerical or status invariant path: preserve numerical values/derivatives/status with explicit host/JIT and single/vmap checks; compiler selection, scheduling, lifecycle and serialization changes are N/A. Sparse/zero/improper/float32/large-shape assertions and all qualification thresholds stay intact. Performance diagnosis must distinguish cache reuse of machine code from reuse of test results. No default deselection or slow-marker movement is planned.

Baseline evidence: saved completed dashboard 75ac1d2 has 3174.420 summed test seconds; 2526 of 2537 test names map uniquely to current modules, 11 are ambiguous and kept separate in /tmp/cff-short-suite-baseline.json. Hybrid centering 572.766s, projection 539.366s, separator kernels 374.230s, affine 222.661s and separator numerics 217.473s total 1926.496s. Agent mu3 adds 263.650s. Last wall measurement 3198.621s. At current numerical source, isolated translated-chain constant-derivative case took 132.44s in a fresh process (/tmp/cff-3uq9-profile-baseline.log); default pytest capture hid compile logs, so a targeted logging diagnostic is needed before attributing the repeated-compilation mechanism. This is performance diagnosis, not a candidate optimizer or qualification run.

**2026-09-14T14:50:53Z**

Deadline implementation: the default runner creates a private process group, imposes 1200 wall seconds, and kills that group on timeout/interruption. Timeout/over-budget/incomplete results cannot replace the dashboard with a passing report; nonzero pytest exit also cannot publish an all-green XML report. Interrupted/invalid reports are retained. Runtime history records incomplete counts as unknown and caps suggested waits at 1200 without falsifying historical measured times. AGENTS now states the hard limit and preserves meaningful numerical coverage. Focused uv run pytest -q tests/test_dashboard.py -x passed 18 tests in 1.12s, including actual subprocess-child cleanup, over-budget completion, interruption and nonzero-result mechanisms. This is not a passing full-suite budget qualification.

Targeted logged profile at unchanged numerical source took 132.59s and emitted 3719 JAX compilation events plus 921 tracing-cache misses. 763 were scan compilations. JAX repeatedly identifies newly created accumulate functions in hybrid_affine.py:160/167 and hybrid_covariance.product, even at repeated array signatures. Evidence /tmp/cff-3uq9-jax-compiles.log, /tmp/cff-3uq9-profile-logged.log and /tmp/cff-3uq9-compilation-counts.json. Next ordinary performance change: stable JIT boundaries around these pure compensated accumulation functions, preserving their custom mathematical JVPs and dynamic value arguments. Re-run the same held diagnostic and required existing numerical assertions; no tolerance or selection change.

**2026-09-14T15:10:11Z**

The 67-minute figure was a suggested quiet wait with 25% headroom; the last measured complete suite actually took 53.31 minutes. The measured regression is still unacceptable. Stable JIT boundaries on compensated affine/product kernels reduced the same constant-derivative case from 132.59 seconds, 3,719 compile events and 763 scans to 33.83 seconds, 1,695 compile events and 8 scans. Adding stable boundaries for covariance resolution, solve, inverse and logdet made all three coordinate derivative cases pass in 40.83 seconds total (30.77 / 3.77 / 6.27 seconds). The equations, certificates and custom JVP rules are unchanged. Evidence: /tmp/cff-3uq9-profile-math.log and /tmp/cff-3uq9-profile-math-timings.jsonl.

An initially empty on-disk JAX executable cache reduced the unchanged 13 mu3 checkpoint corruption cases to 88.45 seconds; five cache entries occupied 1,135,700 bytes. Installed JAX cache source keys compiled programs by HLO, backend/version and compiler flags. This caches executables, never test results. The runner now provides an empty temporary cache per run and removes it afterward; caller-provided cache settings remain explicit and reported. Line-buffered per-test JSONL timings survive timeout. Runner tests passed 21 cases in 1.25 seconds after these changes, including fresh caches across consecutive invocations, caller cache preservation and actual child-process timeout cleanup. Immutable checkpoint source reuse is being validated separately; every corruption still gets fresh arrays and its own new target agent.

Legend: prerequisite --> dependent; tk records remain authoritative.

[ ] 3uq9 (ready, root implementing)
  --> [ ] 8ois (reviewed; valid full gate blocked here)
        --> [ ] y15d (solver implementation)
              --> [ ] a3wz (complete numerical checkpoint)
                    --> [ ] 7sw5 --> ... --> [ ] x0if --> [ ] zov5
[ ] k6g5 (independently ready) ------> [ ] 7sw5

No registered recovery fit or candidate optimizer ran during this work.

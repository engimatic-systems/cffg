---
id: cdc-bwta
status: closed
deps: []
links: []
created: 2026-09-14T17:25:15Z
type: task
priority: 1
assignee: coopwilliams
tags: [testing, performance, tooling]
---
# Keep the expanded joint-solver short suite within twenty minutes

## Background in Plain English — WHY

The default short suite passed in 1084.302 wall seconds at numerical source 875013f with three workers and an initially empty 1 GiB executable cache. Y15D adds complete-residual derivative and solver-policy checks. Their focused runs total 456.27 seconds; the prior suite reported 3172.694 test seconds. These measurements suggest that three workers may no longer leave enough room inside Cooper's hard 1200-second limit, although cache reuse and concurrent compilation can change actual costs. Cooper assigned all short-suite performance implementation to the root supervisor.

## Outcomes / Acceptance Criteria — WHAT

1. Preserve the default not-slow selection, every numerical assertion, the fixed D12 checkpoint thresholds, and the hard 1200-second process-group deadline.
2. Use measured file costs and the development machine's available CPU/memory to choose a bounded worker count. Do not claim that scheduling estimates are measured suite wall time or that compiler RSS is iteration storage.
3. Keep fresh-cache isolation, file locking, controller-owned artifacts, failed/partial-report handling, and one execution per selected test intact. Focused runner tests pass after the final relevant edit.
4. Run the default dashboard gate once after the final relevant code change. It must pass within 1200 wall seconds, disclose the worker count/cache conditions, and publish a complete report. Retain any failed attempt's timings and errors before ordinary fixes.
5. Record the final code boundary, exact commands, results, remaining limits, and this ticket's relation to Y15D in the ticket and PR. Test execution changes require the complete gate; the isolated-tooling exception does not apply.

## Methodological Suggestions — HOW

Start with the existing per-file timing report and evaluate four workers on the 12-core, 64 GiB development machine. Avoid running a second unchanged full suite for review or PR publication. Keep the implementation to runner scheduling unless observed evidence justifies a separate optimization.

## Future work

Y15D still requires independent review and its final full gate. A3WZ owns all native float32 moderate-horizon solver and actual-iteration-storage qualification. Compiler integration and registered recovery remain behind their existing dependencies.

## Dependency handoff

Dependency companion (prerequisite --> dependent; TK records are authoritative):

```text
[ ] cdc-bwta (ready; root scheduling/gate)
      |
      +--> [ ] cff-y15d (implementation done; review/gate pending)
                 |
                 +--> [ ] cff-a3wz (moderate qualification)
                            |
                            +--> [ ] cff-7sw5 (D12 component compilation)
                                      ^
[ ] cff-k6g5 (ready; declarations) -----+
```

## Definitions

- Short suite: the default pytest selection, not slow.
- Executable cache: compiled machine code; passing test results are never cached.
- Worker: an isolated pytest process assigned complete test files.
- Wall time: real elapsed time, including setup and child-process execution.

## Change paths and hostile inputs

Runtime numerical or status invariant: test execution and pass/fail reporting only. Check direct short-suite invocation and dashboard-managed execution; production numerical kernels, host/JIT and single/vmap mathematics, compiler/schedules, lifecycle, serialization and learning are N/A. Timeout, child failure, incomplete XML and artifact races remain applicable runner failure mechanisms. Sparse numerical support, message properness, numerical dtype and model-size semantics are unchanged; their existing tests remain selected.


## Notes

**2026-09-14T17:26:17Z**

Prospective implementation: increase the default worker count from three to four on the 12-core/64-GiB M2 Max and update the AGENTS.md runner description. Preserve file grouping, crash-retry prohibition, deadline, cache and all assertions. Root owns this scheduling change; independent reviewers inspect the pinned Y15D source commit 65b2dcd. Y15D remains blocked on this combined final gate. The planning estimate is not a measured four-worker result.

**2026-09-14T17:27:13Z**

Implemented the prospective scheduling change: four workers, with the same 1200-second deadline, loadfile grouping, no worker retries, initially empty locked 1-GiB cache and unchanged numerical selection/assertions. AGENTS.md matches the runner. Focused command: uv run pytest -q -n 4 --dist=loadfile --max-worker-restart=0 tests/test_dashboard.py --junitxml=/tmp/cdc-bwta-focused.xml; 21 passed in 1.98s, log /tmp/cdc-bwta-focused.log. git diff --check passed. The full combined gate is pending completion of both independent Y15D reviews and any resulting author fixes.

**2026-09-14T18:10:18Z**

Independent scheduling review is complete at 95f4046: no findings. The fresh read-only reviewer verified that only the worker default changed from three to four and that deadline, grouping, no-retry policy, cache rules, numerical selection/assertions and failure publication remain intact. Report: /tmp/cdc-bwta-review.txt; completed Codex CLI process exited 0. The reviewer ran no tests and made no numerical-solver claim. Focused evidence remains 21 passed in 1.98s. Y15D author repairs are now committed at d5e33ae and independently re-reviewing; the new-source checkpoint and combined final full suite remain pending. No timing estimate has been promoted to a measured four-worker result.

**2026-09-14T20:27:13Z**

Final gate passed at 96276c8 after the last code/test change e962d2c: uv run python tools/dashboard.py --run-suite. Exactly 2659 distinct tests executed once and passed, with zero failures, in 1198.26 pytest seconds / 1199.035 measured wall seconds. Four loadfile workers, no retries, initially empty locked 1-GiB executable cache, unchanged not-slow selection and hard 1200-second deadline. All 16 warnings are existing gym distutils deprecations repeated across four workers. Dashboard/runtime history refreshed. Log /tmp/cff-y15d-short-suite.log; call-count and file/worker audit /tmp/cff-y15d-short-suite-summary.json; original timings /var/folders/kg/pmjxhgy506n699dj2g96f2wh0000gn/T/tmpo6cnf5i8.timings.jsonl. Margin is only 0.965 seconds; do not claim robust headroom. Worker phase totals 974.80/993.47/977.02/1185.42 seconds support the root-owned follow-up cdc-c8kc before further harness growth. The current acceptance criteria pass; preserve this result for PR publication without an unchanged rerun.

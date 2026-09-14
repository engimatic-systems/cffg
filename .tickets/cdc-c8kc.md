---
id: cdc-c8kc
status: open
deps: [cdc-bwta]
links: []
created: 2026-09-14T20:25:51Z
type: task
priority: 1
assignee: coopwilliams
tags: [testing, performance, tooling]
---
# Restore short-suite margin using measured execution costs

## Background in Plain English — WHY

The expanded Y15D short suite passed at 96276c8 with 2659 distinct tests, four workers and a fresh locked 1-GiB cache in 1199.035 wall seconds. It met Cooper's hard 1200-second limit with only 0.965 seconds to spare. Worker phase totals were about 975, 993, 977 and 1185 seconds, indicating a substantial scheduling imbalance. New D12 harness tests will need room. Cooper assigned short-suite performance implementation to root.

## Outcomes / Acceptance Criteria — WHAT

1. Use the retained per-file/per-worker evidence to select a bounded scheduling or ordinary performance improvement. Preserve every selected test/assertion, the not-slow selection, failure handling, no retries, cache isolation and hard 1200-second process-group deadline.
2. Keep implementation small and maintainable. Any cost-based ordering must be deterministic and handle missing/new test files safely; it cannot omit or duplicate work. Avoid fragile special cases keyed to individual test incidents.
3. After focused runner checks and independent review, run exactly one final default dashboard gate after the last relevant edit. Report measured wall time, margin, worker/cache conditions, one distinct call per selected test and worker balance. Demonstrate improved margin over the retained 1199.035-second baseline; preserve failures before fixes.
4. Record final implementation/source and evidence in this ticket and PR. Root implements this work. It does not change D12 numerical or recovery thresholds and cannot use the isolated-tooling exception because scheduling affects test execution.

## Methodological Suggestions — HOW

Start with /tmp/cff-y15d-short-suite-summary.json and the original worker JSONL named in cff-y15d's final note. The largest file costs were joint coordinate 587.98s, projection 522.85s, precision 493.60s and centering 438.27s. Measured work ordering may address the 200-second tail without increasing concurrent compiler memory. Preserve the existing fresh-cache and process ownership behavior.

## Future work

Proceed after the immediate bounded compilation-scaling evidence requested by Cooper and before relying on the short suite for further harness growth. Moderate numerical qualification and registered recovery remain separate.

## Definitions

- Margin: 1200 seconds minus measured wall time.
- Worker balance: distribution of executed test/setup/teardown seconds across pytest workers, not a numerical qualification measure.

## Change paths and hostile inputs

Runtime numerical or status invariant row for test execution/pass-fail reporting only. Affected surfaces: dashboard/direct suite invocation, worker/controller scheduling and artifacts. Production numerical host/JIT, single/vmap, compiler, lifecycle, serialization and learning are N/A. Applicable hostile cases are missing timing data, new files, timeout, crashed worker, incomplete results, artifact races and duplicate/missing execution.



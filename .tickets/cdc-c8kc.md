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



## Notes

**2026-09-14T20:42:34Z**

Root read-only scheduling preflight after the Y15D gate (production/test code unchanged). Installed pytest-xdist3.8.0 LoadFileScheduling inherits LoadScopeScheduling. schedule() currently sorts file groups by descending TEST COUNT when loadscopereorder is true; --no-loadscope-reorder retains their collection order. _reschedule assigns another complete file when a worker has at most two pending tests. Sources: .venv/lib/python3.12/site-packages/xdist/scheduler/loadfile.py; loadscope.py:313,338,374; plugin.py:131,144. Existing tests/conftest.py has controller-owned phase timing and artifact cleanup, but no collection-order hook. tools/dashboard.py:169 controls the four-worker loadfile command and fresh cache.

Offline analysis grouped the retained 7977 setup/call/teardown records into 2659 distinct tests and their files, holding each observed phase duration fixed. A four-worker event-queue simulation with the installed low-water rule and current descending-count ordering reproduced the actual worker phase totals exactly: 993.466/974.803/977.019/1185.420 seconds. Reordering the same complete files by descending observed cost produced simulated totals 1032.674/1032.667/1032.668/1032.700 seconds, reducing the longest simulated phase sum by152.72 seconds. Artifact: /tmp/cdc-c8kc-scheduling-simulation.json. These are conditional scheduling-model results, NOT measured suite speed or promised wall time; actual cache/concurrent-compilation costs may change. No tests, compilation or solves ran.

A small candidate implementation is a deterministic collection-order hook using measured per-file costs plus --no-loadscope-reorder, while retaining xdist's existing complete-file scheduler and every discovered test. Avoid enumerating only known files on the pytest command line, which could silently miss new/nested discovery. Preserve within-file order; define safe behavior for absent/new/malformed timing data. Store only runtime-history evidence in the existing approved runtime-history mechanism, not repository session-state notes. Test actual scheduler/collection behavior on bounded synthetic cases and then use the required one fresh full gate after review. Root owns implementation after cdc-ogj9's immediate small measurement request.

Existing record_suite_runtime(tests, wall_seconds, pytest_code) in tools/dashboard.py:336 already receives each validated test's file and elapsed milliseconds. Per-file sums could be persisted as runtime-history metadata alongside the existing aggregate without adding another session-state file or needing an additional test run. Ordering is only a hint: preserve collection/selection and handle unknown filenames/costs safely. JUnit time precision and phase-timing sums differ slightly; disclose which source supplies ordering. A driver-provided cost mapping consumed by a deterministic conftest collection hook plus the existing --no-loadscope-reorder flag could avoid copying or subclassing xdist's scheduler. This remains a candidate, not implemented or qualified behavior.

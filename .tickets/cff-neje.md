---
id: cff-neje
status: closed
deps: []
links: [cff-b47g]
created: 2026-07-29T17:59:08Z
type: chore
priority: 3
assignee: coopwilliams
---
# Persist full-suite wall-clock history for agent waits

## Plain English

This ticket tells agents how long the repository full suite actually takes on this machine. The dashboard runner records recent wall-clock durations in a tracked repository artifact, and the repository test instructions point agents to that history before they launch or wait on a run.

## Design

Measure the pytest subprocess with a monotonic clock only in --run-suite mode. Persist a bounded JSON history at tools/test-suite-runtimes.json with wall time, summed JUnit case time, test count, result, completion time, commit, and a conservative suggested wait; reusing JUnit must not invent or overwrite a wall-clock measurement.

## Acceptance Criteria

AGENTS.md and README.md name the runtime-history path and require checking it before a final run. One --run-suite invocation writes exactly one bounded record and a suggested wait; --junitxml reuse does not overwrite measured history. Focused dashboard tests cover passing/failing metadata, history bounds, and one-pass behavior.

## Notes

**2026-07-29T18:10:11Z**

Dashboard --run-suite now measures the pytest subprocess with a monotonic clock and atomically updates tracked tools/test-suite-runtimes.json. The bounded record distinguishes wall time from summed JUnit case time, preserves pass/fail/test-count/commit metadata, and recommends the slowest recent-five wall time plus 25 percent headroom rounded to 30 seconds; --junitxml reuse never fabricates a measurement. AGENTS.md and README point agents to the repo file before running or waiting. Bootstrap final run: 1281 passed, wall_seconds=360.168, case_seconds=351.708, suggested_wait_seconds=480. Focused affected suite: 52 passed; git diff --check and tk dep cycle passed.

---
id: cff-kut1
status: closed
deps: []
links: []
created: 2026-07-14T04:04:56Z
type: chore
priority: 1
assignee: coopwilliams
tags: [performance, dashboard, tests]
---
# Reuse pytest results for dashboard refresh

The required end-of-session dashboard refresh always reruns the entire pytest suite, even when the agent has just completed a full green run. At 1,132 tests the suite took 516.44s, so the duplicate pass adds roughly eight minutes to every landing. Measure per-file timing from JUnit evidence and let tools/dashboard.py render an existing completed JUnit report without invoking pytest, while retaining an explicit standalone run mode.

## Acceptance Criteria

1. Dashboard can consume an existing JUnit XML report and does not spawn pytest in reuse mode. 2. It validates missing, malformed, empty, partial, and failing reports loudly enough that stale/partial results cannot masquerade as a full green suite. 3. Existing standalone behavior remains available explicitly. 4. AGENTS.md session protocol runs the full suite once with JUnit output, then reuses it for the dashboard. 5. Tooling tests cover argument handling, report validation, and no-rerun behavior. 6. Ticket records measured per-file/runtime concentration and recommends, but does not implement, any further test tiers without Cooper approval.


## Notes

**2026-07-14T04:06:35Z**

Timing evidence from the 2026-07-13 dashboard JUnit report: 1,132 cases, 517.441s suite wall / 516.163s summed testcase time. Top files: test_tmaze_learning 213.886s (41.4%), test_b_learning 55.524s (10.8%), test_glucose 44.192s (8.6%), test_b_epistemics 36.761s (7.1%), test_agent 31.011s (6.0%). Top 3 = 60.8%; top 5 = 73.9%. Recommendation for later discussion: a fast/default vs slow-simulation tier could have high leverage, but this ticket will not mark, skip, or alter those tests. It only removes the duplicate dashboard pass.

**2026-07-14T04:14:59Z**

Implemented validated one-pass dashboard workflow. tools/dashboard.py now requires either --run-suite (one live full pytest pass + render) or --junitxml PATH (no pytest; reuse completed result). Reuse validates XML syntax, declared test/failure counts, nonempty cases, coverage of every tests/test_*.py module, and freshness against src/tests/tools/pyproject/uv.lock. Failing reports render red and exit nonzero. AGENTS.md now prescribes focused iteration plus one full landing pass; README no longer advertises the stale 105-test count. Added tests/test_dashboard.py to M6 with malformed/empty/mismatch/partial/stale/no-rerun/single-run/failing coverage. Verification: focused 9 passed; full 1,137 passed with 4 existing dependency warnings in 224.98s, report at /tmp/cffg-tests.xml. No slow markers, skips, or tier changes were made; timing evidence/recommendation is in the prior note.

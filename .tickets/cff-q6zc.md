---
id: cff-q6zc
status: closed
deps: []
links: []
created: 2026-09-25T09:13:49Z
type: chore
priority: 2
assignee: coopwilliams
parent: cff-61ag
tags: [testing, runtime]
---
# Keep the short suite within budget after mixture-rank regressions

## Background in Plain English

The passing cff-61ag short suite took1254.886 seconds, exceeding the1200-second project limit. All30 new numerical-rank regressions must stay in the default gate. Several long behavior and replay scenarios can remain available in the opt-in suite.

## Outcomes / Acceptance Criteria

1. Keep all numerical-rank, sparse-support, float32 and uncomfortable-scale regressions in the default suite.
2. Use measured per-test timings to select unchanged long behavioral tests for the slow marker; record measured and estimated totals.
3. Preserve every assertion and keep the moved tests runnable with explicit slow selection.
4. Validate collection and whitespace. Reuse the passing2561-test result without rerunning solely for selection changes.

## Methodological Suggestions

Another sound selection is acceptable. Follow the test-discovery/execution change path; this marker-only budget adjustment is the explicit reuse exception in AGENTS.md, not a claim of isolated-tooling validation. Select long complete-agent replay, repeated fitting and renaming scenarios while retaining direct algebra, runtime failure/status, checkpoint corruption, parameter updates and numerical robustness cases.

## Future work

- cff-61ag publishes the numerical repair and this measured budget adjustment together in separate commits.

## Definitions

The short suite is the default not-slow pytest selection. A slow marker moves a test to an explicitly requested suite without deleting its assertions. A replay runs an agent again from saved state.


## Notes

**2026-09-25T09:14:17Z**

Measured full gate:2561 passed,87 deselected; pytest1239.47s, dashboard wall1254.886s. Moved unchanged long scenarios: full rSLDS factor-renaming14.493s; multi-coordinate sequence fit10.271s; two-window agent/enumerator6.427s; posterior refresh replay7.051s; runtime-configuration cache replay7.205s; version1 functional-agent migration8.251s; typed-key full-agent replay5.602s. Total59.300s, estimated remaining1195.586s. All30 new rank regressions and prior collapse/equality/failure/float32/sparse/large-scale tests remain default, alongside direct learning, fit rejection/certification, checkpoint corruption and runtime status tests. Marker-only validation: changed-file collection45selected/28deselected; git diff --check and tk dep cycle clean. Reused passing suite per explicit budget exception; no new full-suite result claimed after markers. Done; no follow-up required.

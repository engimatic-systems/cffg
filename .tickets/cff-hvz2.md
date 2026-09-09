---
id: cff-hvz2
status: in_progress
deps: []
links: []
created: 2026-09-09T15:26:32Z
type: chore
priority: 2
assignee: coopwilliams
tags: [housekeeping, tooling]
---
# Keep only three test-suite runtime results

## Background in Plain English — WHY

Agents read the runtime record before waiting for tests. Its 40-entry history limit makes a small scheduling aid unnecessarily long. Cooper requested a housekeeping branch from PR111 that renames recent to previous-3 and retains only the last three results.

## Outcomes / Acceptance Criteria — WHAT

1. The committed runtime record uses previous-3, has no recent key, and retains at most three completed results in chronological order, including latest.
2. The dashboard writes that bounded format on every measured run and accepts the old format when updating an existing record.
3. Latest result details remain unchanged during the one-time format migration. Suggested wait time uses only comparable retained runs, with the existing headroom and rounding policy.
4. Reusing a JUnit report does not append or replace measured runtime history.
5. Focused dashboard tests cover migration, rollover, and wait estimation; the applicable final validation passes. No inference behavior changes.

## Methodological Suggestions — HOW

Optional: version the renamed history format, retain a read-only legacy input path, and keep the existing atomic file replacement. Use tests/test_dashboard.py and validate the committed JSON without running the wording audit.

## Future work

cff-umag owns reducing wiggle-audit output. Other token-cost observations are recommendations only in this change; general process or document revisions require a separate decision.

## Definitions

- Runtime record: measured elapsed time and outcome from one test-suite run.
- JUnit report: the XML test-result file consumed by the dashboard.
- Rollover: dropping the oldest entry when a new result exceeds the three-entry limit.


## Notes

**2026-09-09T15:27:50Z**

Base bf6bbe9 from PR111, refreshed before creating agent/housekeeping-runtime-history. Change-path index: Runtime numerical or status invariant, dashboard consumer only; authority is tools/dashboard.py:record_suite_runtime rather than engine status. Affected pairs: measured execution versus reused JUnit report; old versus renamed runtime-history format. Host/JIT, flooding/scheduled, direct/batched inference and latest/legacy model checkpoints are all N/A because no engine or model state changes. Focused test file: tests/test_dashboard.py. Token-cost side research is read-only and reports recommendations, not additional changes. No wording audit will run.

**2026-09-09T15:30:59Z**

Implemented: runtime schema2 uses previous-3 (last three completed results, including latest); schema1/recent input migrates when recording a run. Three-result rollover and wait estimation are tested, and reused JUnit reports remain read-only for measured history. The committed one-time migration preserves latest and the exact old tail of three records: 430 lines/11,215 bytes becomes 50 lines/1,229 bytes (about 89% fewer bytes). Existing history remains in Git. Focused validation: all13 dashboard tests pass in0.04s; new tests pass Ruff, git diff --check and tk dep cycle pass. Five dashboard Ruff findings were checked against the base and are unchanged, not introduced here. Final gate pending the testing-policy choice; current rules otherwise require the full short suite.

Side-research question: where else do unnecessary tokens enter context? Answer: historical ticket wording-audit inventories, unscoped reads of large canonical docs and engine modules, tracked chat archives exposed to broad searches, duplicated startup/delegation context, and verbose tool output. Sources/counts from performance_review: 181 ticket files contain about182,000 words; cff-wbhu alone6,940 words with a945-word historical wording inventory; six canonical docs about41,600 words; tracked chats about9.8MB; AGENTS227 lines/1,797 words. These are sizes, not measured token use; files cost context only when loaded. Confidence high for counts and visibility, qualitative for savings. Recommendations only: retain short current handoffs and useful research conclusions; explicitly opt into archived-chat searches; use indexed section/function reads; reuse supplied instructions unless changed; compact stdout with detailed artifacts on disk. No broad cleanup, rule change, transcript read, or wording audit was performed.

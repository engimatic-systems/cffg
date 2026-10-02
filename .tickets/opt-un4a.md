---
id: opt-un4a
status: closed
deps: []
links: []
created: 2026-10-02T05:18:10Z
type: chore
priority: 2
assignee: coopwilliams
parent: opt-ewtf
tags: [tests, performance]
---
# Keep the default gate below twenty minutes after the adaptive runtime change

## Background in Plain English
The final opt-ewtf gate passed all 2651 selected tests, but its measured wall time was 1202.469 seconds, slightly over the repository limit of 1200 seconds. The dashboard retains each testcase duration. The three-state recurrence ownership diagnostic repeats the same assertions as the retained two-state case and took 5.595 seconds; exact-zero, float32/float64 and 100-step/eight-state robustness checks remain separate default tests.

## Outcomes / Acceptance Criteria
1. Move only the three-state parameter of test_enumerator_and_engine_comparison_issues_variational_ownership to the opt-in slow selection; preserve its assertions and the two-state default case.
2. Record the measured 1202.469-second gate and estimated 1196.874-second remaining default runtime, with the 5.595-second subtraction identified as an estimate rather than a rerun.
3. Reuse the passing gate which already exercised the moved case. Validate marker selection by collection only, plus git diff --check and tk dep cycle. Do not rerun an unchanged numerical suite just for this marker edit.
4. Include this completed discovery ticket in the single opt-ewtf PR.

## Methodological Suggestions
Architecture change path: Runtime numerical or status invariant, regression-anchor selection only. Affected surface is pytest default/slow selection; engine, objective, numerical formulas, serialization, host/JIT and Colony execution are N/A. Keep the authoritative measured runtime history unchanged. The existing budget rule explicitly permits reusing the passing run after selection-only edits.

## Future work
No new runtime implementation belongs here. Any further gate growth should be evaluated using a fresh measured run.

## Definitions
Default gate means pytest selection not slow. Wall time includes collection and process overhead. Estimated remaining time subtracts the measured duration of the moved testcase and is not a fresh run.


## Notes

**2026-10-02T05:18:31Z**

Completed: only the card=3 ownership diagnostic is now opt-in slow; its assertions are unchanged. The passing gate at ef71254 already exercised it:2651 passed,192 deselected,1202.469s measured wall (pytest1188.05s). Dashboard testcase duration5.595s gives estimated default1196.874s; this is an estimate, not a rerun. Collection verifies13 default/1 slow in this file and retains exact-zero, f32/f64 and100-step/eight-state cases. git diff --check and tk dep cycle pass. Reuse the passing result per the explicit short-suite budget exception. Include this discovery in the primary opt-ewtf PR; no engine or numerical behavior changed.

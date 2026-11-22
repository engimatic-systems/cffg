---
id: cpr-stp2
status: closed
deps: []
links: []
created: 2026-11-22T02:01:05Z
type: bug
priority: 2
assignee: coopwilliams
tags: [premerge, runtime, diagnostics]
---
# Stop repeating an invalid first adaptive proposal and preserve the original failure attempt

## Background in Plain English — WHY

PR 185 accelerates adaptive planning. With a previously converged scalar cubic state and a finite sensor parameter of 1e200, its first proposed update is invalid. Rejection halves the distance to a retained point that is identical to the proposal. With budget 40 the new loop evaluates 32 proposal rounds and reports the soft sensor failure at attempt 33. The base reported that same failure at attempt 1.

## Outcomes / Acceptance Criteria — WHAT

1. The first rejection cannot consume the proposal budget by repeating an identical state. Fallback retains the offending sensor, failure code, stage, originating logical attempt and correct prior state.
2. A later rejected proposal with a distinct accepted point still permits recovery. Cover insufficient budget, a repeatable invalid input, and a successful proposal run.
3. Exercise float32 and float64 with finite stress inputs appropriate to each dtype. Preserve the intended treatment of vacuous or improper inputs. Exact categorical zeros are outside this Gaussian proposal loop.

## Methodological Suggestions — HOW

Use the Runtime numerical or status invariant row. Inspect adaptive_proposals.proposal kernel and outer_strategy fallback accounting. Compare optimized JIT and original host/fallback behavior; check logical provenance separately from physical evaluations. No serialization change is needed.

## Future work

Independent of the cfp-9806 reference path. Validate the corrected numerical head with the project gate.

## Definitions

A proposal is a candidate for the next set of iterative solver coordinates. Fallback invokes the existing solver after the fast path fails to certify a result.


## Notes

**2026-11-22T02:02:35Z**

Pinned CPU/JAX probes at 92d6acc and f9df9d6 agree on improper_tilt / sensor / soft. Under budget 40 the reported failure attempt changes from 1 to 33. Under budget 9 it changes from 1 to 2. Both outcomes are still diverged. Source, commands and JSON are in ~/agent-share/cffg-premerge-review/late/.

**2026-11-22T02:30:14Z**

The applicable architecture row is Runtime numerical or status invariant. The affected surfaces are the compiled Gaussian proposal loop and host trust fallback, including its public failure attempt, retained state, counters and remaining phase budget. Exact/discrete schedules, functional updates, serialization, ledger and readouts are N/A. Test a finite overflow-inducing sensor parameter in float32 and float64, at budgets that skip, limit and admit proposals. Check recovery from a later invalid trial and ordinary certification. Existing startup checks cover vacuous and improper cavities. Exact categorical zeros are N/A.

**2026-11-22T02:33:47Z**

The compiled loop stops after the first invalid map if it has no valid point for backtracking. Trust fallback then evaluates the original state with the remaining budget. A terminal recheck there keeps the originating attempt. Discovery now includes its map call in the physical evaluation count. Once a valid point exists, rejected trials can still backtrack. The scalar regression retains 1 when the third round fails at 2, then certifies 1.5 with one more round.

The original CPU pr185.py probe now reports improper_tilt on sensor in the soft phase at attempt 1 for budgets 9 and 40, with one rejection, one charged iteration and two map evaluations each. Regression tests also verify float32 (1e20), float64 (1e200), equality of every returned state leaf to the prior state, and remaining trust budgets of 8, 8 and 39 for requested budgets 8, 9 and 40.

uv run pytest -q tests/test_adaptive_runtime.py tests/test_adaptive_startup.py tests/test_outer_failures.py passed 71 tests with 2 slow tests deselected. A final focused rerun covering first rejection, later recovery and fallback passed 5 tests. Cooper owns the integrated short-suite gate.

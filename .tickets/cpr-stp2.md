---
id: cpr-stp2
status: open
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

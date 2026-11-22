---
id: cpr-rpya
status: open
deps: []
links: []
created: 2026-11-22T02:01:05Z
type: bug
priority: 2
assignee: coopwilliams
tags: [premerge, pcgym, validation]
---
# Verify that the reported reactor action comes from the certified site messages

## Background in Plain English — WHY

In the PR 186 reference tool the checker accepts messages and a separate list of beliefs. On the default window, increasing the current coolant mean by 4 model units changes the reported action by 14 kelvin. Both checker modes accept the modified result. The normal command derives its beliefs from the correct messages, so the retained measurements still have independent support. The gap matters when the pilot connects the checker to another caller.

## Outcomes / Acceptance Criteria — WHAT

1. Changing just the reported action or its source belief by 14 K is rejected, or made impossible by deriving the returned action from the certified messages. Exercise both the independent and stock-rule checker paths.
2. A wrong or missing belief that a caller can supply cannot silently affect an accepted output. Cover altered means and covariances wherever the checker still accepts them as inputs.
3. The nine retained float64 reactor windows still pass, and the stale-history negative control still fails. The exported result identifies the action that was checked.

## Methodological Suggestions — HOW

For the Runtime numerical or status invariant row, inspect tools/benchmark_cstr_chain.py functions solve, posterior, check_candidate and main, and the consumption planned in cfp-9806. Compare the independent host check and original compiled local rules. This is a fixed-model reference tool. General engine, Colony and serialization changes are outside scope.

## Future work

The closed-loop pilot cfp-9806 consumes the corrected checker. Dispatch recognition cfp-hlqd can proceed independently.

## Definitions

A site is a variable together with its incoming Gaussian messages. A belief is the posterior mean and covariance obtained by combining those messages. A negative control deliberately changes a result to see whether validation catches it.


## Notes

**2026-11-22T02:02:35Z**

At 80ae1f2 both checkers accept 296.459723 K and a perturbed 310.459723 K with identical residuals and energies within each mode. Source, commands and JSON are in ~/agent-share/cffg-premerge-review/late/.

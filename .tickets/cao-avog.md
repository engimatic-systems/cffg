---
id: cao-avog
status: closed
deps: []
links: []
created: 2026-11-24T14:51:37Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-xfnx
tags: [learning, performance, numerics]
---
# Offer rounded-only mixed updates for callers choosing lower CPU cost

## Background in Plain English — WHY

Mixed component learning retains rounding remainders to accumulate small observations. This is the accepted default. On the six measured CPU workloads it cost 21.7–54.2% more than the earlier rounded-only route. A caller may prefer the earlier arithmetic and its lower cost when small lost updates are acceptable.

## Outcomes / Acceptance Criteria — WHAT

1. Public scoring and updating accept an explicit rounded-only bank. Existing banks keep compensated precision and their current results.
2. Conversion makes the loss of any retained remainders explicit. Switching back starts from the rounded posterior. Document how to select the same arithmetic at restart.
3. On the previous public workloads, compare numerical values and statuses to fd8bfd1, including float32, float64, masks, invalid inputs, zero weights and the known translated streaming loss. Run compiled and eager paths and check that a compiled stream keeps the smaller state layout.
4. Measure the recurring speed of both public routes on matched inputs with coordinated CPU access. Report the workload and cost alongside the accuracy tradeoff.

## Methodological Suggestions — HOW

The relevant architecture paths are Learnable or checkpoint feature and Runtime numerical or status invariant. Compare public eager, compiled, scanned and batched execution and compact/exact checkpoint handoffs. Standalone mixed learning runs outside graph schedules, the ledger and declared parameter dispatch. Preserve the old operation order at large offsets and near the input precision limit.

## Future work

cff-xfnx and caoc-0dvd retain the default precision decision. This library option will be available to callers choosing arithmetic for specific experiments.

## Definitions

A rounding remainder is the small difference lost when a value is rounded to the next representable value. A responsibility is the fraction of one observation assigned to a component.


## Notes

**2026-11-24T14:56:02Z**

MixedBank.as_rounded() explicitly discards residuals and returns seven primary fields. Public score/update dispatch on the pytree type, including under jit, vmap and scan. RoundedBank.as_compensated() begins with zero residuals. Both snapshot options store the complete rounded state as v2; RoundedBank.from_state_dict accepts validated v1/v2/v3 and selects the visible values. This option is scoped to standalone mixed learning. Lifecycle and acted AXIOM runs use compensated banks.

Qualified against independent pinned source trees fd8bfd1 and f3ab2da on 40 recurrent inputs each for N=1/K=32 and N=32/K=500, in float32 and float64. Each step scored all fields with mixing enabled and applied the result to the next bank. Every byte of the scores, responsibilities, statuses and final posterior matched its respective reference, including all default residuals. The focused mixed accumulation, learning and scoring tests passed 48/48 in 18.15 seconds. Fractional mean and scatter loss at a translated one-ULP feature in both precisions is reproduced as an explicit rounded-mode limitation. Malformed v3 remainders reject on restore even in rounded mode.

The coordinated CPU comparison used source float32, d=6, categorical widths 3 and 5. Each alternate-order pair timed eight device-resident public score/update calls, starting at the 32-tick learned state for that arm. Final bank, score, responsibilities and validity synchronized; no host result copies or compilation inside timer. Three warmups and 12 paired blocks. N=1/K=32 median rounded 0.09253 ms versus default 0.12273 ms, median paired ratio 0.7563, 95% paired-block bootstrap interval [0.7420,0.7689]. N=32/K=500 median 1.28426 versus 1.65594 ms, ratio 0.7788, interval [0.7692,0.7820]. Raw output and reference comparison scripts: ~/agent-share/axiom-online-precision/rounded-option/{speed.json,math.json,measure.py,check.py}. Parent owns dashboard and short-suite integration gate.

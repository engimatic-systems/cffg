---
id: cao-1ggb
status: closed
deps: [cao-fs5s]
links: []
created: 2026-11-24T14:51:40Z
type: task
priority: 2
assignee: coopwilliams
parent: car-zieb
tags: [axiom]
---
# Run and resume scheduled interaction reductions on the real action clock

## Background in Plain English — WHY

The online actor learns continuously across game resets. It must periodically combine redundant interaction rules and preserve the synthetic experiences used for that choice when it restarts.

## Outcomes / Acceptance Criteria — WHAT

1. Every 500th completed real action, including action 10000, invokes the qualified reducer with the full interaction model, the seeded dormant data, 2000 fresh generated records and up to 2000 candidate pairs. Ordinary actions use constant work and keep the input objects.
2. Successful passes return the updated device bank, continued random key, accumulated records and a reviewable receipt. Failed passes keep the previous model and do not advance the completed reduction clock.
3. The checkpoint validates the reduction clock and generated history against the actor step. Compare all bank fields, key, tape and decisions with direct host execution on an evolving sparse model and after restoration.
4. Qualify cost and memory at realistic record and component counts. Keep any short development measurements separate from the acted benchmark.

## Methodological Suggestions — HOW

Use the Learnable or checkpoint feature and Runtime numerical or status invariant change paths. Compare the native actor/device entry point with exact host execution. Exercise empty support, f32, the last reduction and a corrupted checkpoint. Graph iteration, IR and ledger changes are N/A.

## Future work

car-zieb owns the acted game pilot, campaign timing and final suite gate.

## Definitions

A reduction pass combines rules using retained synthetic data. A dormant row is a component slot available for future rules.


## Notes

**2026-11-24T14:55:12Z**

The actor boundary is available in tools/axiom/evaluation_reduction.py. The synthetic actor check compared every exact bank/checkpoint field, both scores, every greedy trial and all future generated records against direct host calls after real release learner updates. Tested clock 499/500, 1000 and the 10000 terminal boundary with 40000 records, plus empty support, skipped reductions and incomplete archives. Command: uv run --frozen --no-sync python tools/axiom/evaluation_reduction_check.py (2 passed). The integrated actor and checkpoint owners are using this interface. Next qualify an exact likelihood bound on fixed dormant components to avoid scoring guaranteed losers at full capacity.

**2026-11-24T14:59:12Z**

All 22 focused core cases passed, including both opt-in 5000-slot tests and two new tests for exact bounds, ties, numerical range and a dormant row becoming the winner after a trial. Fixed diagonal precision makes every term in the centered quadratic nonnegative; the row constant plus the four maximum categorical scores is an upper bound in the same arithmetic. Score only when the current winner fails to exceed that bound strictly. Cache evaluated fixed winners for the rest of the pass and rebuild the bounds whenever the global likelihood repair changes. General precision or extreme operands use the exhaustive path. The deterministic K=5000 development work-count check used the actual seeded dormant bank, all six head widths and six live rows learned using learn_frame. At 500 actions it performed 60000 instead of 10048000 context comparisons, 5 pairs and 3 merges. The entire bank, state, score and decisions matched dense computation. At 10000, after further learned updates, 40000 retained rows required 560000 evaluations, 5 pairs and 1 merge. The terminal tape used 38000 tiled earlier generated records plus 2000 fresh records and occupied 2080000 bytes. Pair proposal still compares its sampled live sources against all slots. This is a work count; busy or saturated bank wall times are unmeasured. Reproduce with uv run --frozen --no-sync python tools/axiom/evaluation_reduction_check.py --scale.

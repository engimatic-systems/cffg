---
id: cml-6ba0
status: closed
deps: []
links: []
created: 2026-11-28T04:28:22Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-cj5h
---
# Expose a per-owner free-energy breakdown reconciled to the scalar ledger

## Background in Plain English — WHY

Engine.bfe returns a total in nats. Model authors need the contributions from the local factors, preferences, information terms and entropy corrections that the runtime actually scored. The independent sensor-query calculation in cff-gul8 answers another question.

## Outcomes / Acceptance Criteria — WHAT

1. Meet cff-cj5h criteria 21–58, 87–89, 92 and 94–97. Provide a deterministic, read-only Engine.fe_breakdown(state) for supported sum-product and GFE graphs with stable owner and component labels, explicit unsupported domains and numeric owner totals.
2. Reconcile all owned terms to Engine.bfe without double counting explanatory components. Reuse ledger primitives across both surfaces and preserve the scalar execution path. Check against independent fixed-goal channel algebra, sparse/improper cases, float32 and the 80-state fixture.
3. Identify where claims are exact, optimistic bounds or approximations. Cover all current GFE owners at implementation time and document exclusions. Keep predictive port information under cff-gul8 distinct.

## Methodological Suggestions — HOW

The Runtime numerical or status invariant and New factor or rule family change paths identify host/JIT, single/vmap and status surfaces to inventory. On a774a54 Engine.bfe delegates to ledger.py and ledger_terms.py. Inspect those owners before factoring out component sums.

## Future work

cff-gul8 owns expected information gain from opening candidate DATA ports.

## Definitions

A ledger owner is a node or correction contributing once to the stored objective. A nat measures information using natural logarithms. GFE is generalized free energy.


## Notes

**2026-11-28T04:37:05Z**

Change path: Runtime numerical or status invariant and factor energy. Factor/edge row values come from the existing scalar ledger before group reduction. A separate opt-in compiled readout returns values, signs and stable owner labels. Flood/scheduled consume the same state and kernel; scalar/vmap solver math is unchanged; public breakdown accepts one state; serialization and checkpoint forms are unchanged. Qualify ordinary and composite owners, sparse and unsupported cases, mu3 preparation, nonfinite local terms, f32/f64, and 80-state channel plus independent mutual information.

**2026-11-28T07:04:05Z**

Integrated closure. This review head includes lint, ledger readout and exact MAP ties as separate commits. The scalar ledger was bit-identical to origin/main across 16 float32/float64 comparisons. The combined short suite at e6ad56d passed all 2858 tests in 444.02 seconds.

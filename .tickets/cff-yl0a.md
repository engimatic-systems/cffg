---
id: cff-yl0a
status: open
deps: [cff-d56d]
links: [cff-wbhu]
created: 2026-07-27T05:44:29Z
type: feature
priority: 2
assignee: coopwilliams
tags: [performance, scheduling]
---
# Chain slice fusion: composite scan kernels to close the schedule 20x gap

cff-d56d's two-pass schedules cut the hostile 256-step K=8 chain from 83.24 ms to 0.99 ms warm (recorded M2 Max CPU/x64 protocol), but the recorded dedicated-SSM reference is 0.0198 ms, so the kernel sits at ~50x against the 20x acceptance target. The remaining gap is per-step operation count, not scheduling: each scan step fires 2-3 factor kernels (gather, pad, rule, normalize, register bookkeeping) where a dedicated smoother does one matrix-vector alpha update. Loop unrolling and registerization were measured and do not close it.

## Acceptance Criteria

A fusion pass recognizes repeating (transition, equality, emission) slice patterns inside a registerized ScanStage and emits one composite update per step (algebraically identical: composition of the registered rules, normalization preserved). Hostile T=256 warm kernel within 20x of the recorded 0.0198 ms reference. Parity tests: bit/tolerance-identical marginals and FE against the unfused schedule and flooding. Fusion is an execution detail: Schedule.describe() unchanged.


## Notes

**2026-07-27T14:13:01Z**

Scope update after the 2026-07-27 review round: the fused status path plus live matched-shape referencing brought every benchmark configuration inside the ticket's 20x envelope (hostile K=2 run 13-19x, ordinary K=8 7-8x, LGSSM d=4/m=2 17-19x), so slice fusion is no longer an acceptance blocker for cff-d56d. It remains the right lever if the envelope must tighten (LGSSM mid-sizes sit at ~19x with little margin) or if the stricter recorded reference figure (~2x below live) becomes the standard.

**2026-07-27T15:47:43Z**

BOUND AFTER cff-d56d CLOSURE (2026-07-27). This is the strict-kernel continuation, not a reopening of the scheduling P1. Keep the public matched-model/full-posterior reference and the minimal-output low-level reference as separate reported columns: the latter projects only the requested smoother output inside JIT (and precomputes HMM likelihoods), so it is an optimization ceiling rather than the same output contract. Scope stays HMM chain lowering: role-specific directed outputs plus fused transition/equality/emission scan, with generic-vs-fused marginal/FE/status/gradient parity and hostile-input regressions. Do not absorb readout vectorization (cff-3kdk) or a speculative LGSSM fast path; profile and file the latter separately only if the product-facing LGSSM envelope remains a priority after readout is fixed.

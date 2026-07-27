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


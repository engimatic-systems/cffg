---
id: cff-3kdk
status: open
deps: []
links: [cff-d56d]
created: 2026-07-27T05:44:29Z
type: task
priority: 2
assignee: coopwilliams
tags: [performance]
---
# Vectorize engine readouts: marginals and gaussian_marginals loop per edge in Python

Engine.marginals and Engine.gaussian_marginals iterate edges in Python with per-edge numpy/LAPACK work. The cff-d56d benchmark measured gaussian_marginals at 46 ms (T=16) to 773 ms (T=256) on the 2-state LGSSM - it dominates end-to-end time once inference itself is O(E). Bucket edges by cardinality/dimension (the interior_slot_groups pattern) and compute moments batched.

## Acceptance Criteria

Readout phase on the cff-d56d benchmark drops at least 10x at T=256 with identical values; no per-edge Python loop in the readout hot path; existing readout tests unchanged.


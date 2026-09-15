---
id: cff-ll6y
status: open
deps: [cff-v53k]
links: []
created: 2026-09-15T18:28:19Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-1o72
---
# Train neural means with versioned optimizer continuation

## Background in Plain English — WHY

Certified fixed-point gradients need an optimizer and a restartable training state. The saved template remains the authority for the network and its positive noise.

## Outcomes / Acceptance Criteria — WHAT

1. Training applies only certified finite gradients and reports refused updates without mutating parameters.
2. A frozen seeded positive-noise nonlinear regression benchmark beats its untrained network and linear baseline on a held-out grid.
3. A separate versioned optimizer checkpoint binds structure, target blob layout, fixed parameters, numerical settings and complete continuation arrays.
4. Restoration is validated before publication and preserves the next update/objective; missing or legacy schema fails specifically unless explicitly migrated.
5. BFE training is supported; GFE training is admitted only within the complete graph capability boundary, with unsupported combinations rejected.

## Methodological Suggestions — HOW

Use Learnable or checkpoint feature and Runtime numerical or status invariant. Audit tying, reconstruction, dtype, fresh/stale certificates and atomic load errors. Freeze data generation, split, seeds, step rule and numeric acceptance before final comparison. Optimizer state is not a conjugate Learnables posterior.

## Future work

cff-3zzo provides deterministic amortized initialization; exact zero noise remains cff-u06e.

## Definitions

Optimizer continuation: weights, moment arrays, step count and any owned stochastic/data-position state needed to reproduce the next update. Held-out data: examples not used to fit the weights.

---
id: cff-3zzo
status: open
deps: [cff-v53k]
links: []
created: 2026-09-15T18:28:19Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-1o72
---
# Learn deterministic amortized Gaussian message initialization

## Background in Plain English — WHY

A recognition model can predict useful starting messages from observations. The normal inference solver must still certify the final beliefs and energy.

## Outcomes / Acceptance Criteria — WHAT

1. A deterministic trained proposal changes only declared message coordinates and preserves model parameters, observations and exact support.
2. Improper proposals are rejected without covariance floors; accepted proposals still use the ordinary certified adaptive solver.
3. Paired held-out proposals and neutral starts agree on final beliefs and energy, and a 95% confidence interval for sweep-count reduction excludes zero.
4. Training uses certified target states; architecture, feature/coordinate schema, normalizers and weights have versioned persistence.
5. The API offers no importance-sampling proposal, making particle-weight and ESS requirements explicitly N/A.

## Methodological Suggestions — HOW

Use Runtime numerical or status invariant and Learnable or checkpoint feature. Audit host/batched numerical proposal evaluation, proper/vacuous inputs, both dtypes, stale structure and support. Freeze the paired held-out experiment and confidence-interval calculation before final measurement.

## Future work

No additional work is identified within cff-1o72. Multiple-root models retain their normal inference limitations; no universal speedup is promised.

## Definitions

Amortizer: a learned function that predicts an inference starting point. Paired comparison: the same held-out observation evaluated with both initializations. Confidence interval: uncertainty in the average observed sweep-count difference.

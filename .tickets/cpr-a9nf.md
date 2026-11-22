---
id: cpr-a9nf
status: open
deps: []
links: []
created: 2026-11-22T02:01:05Z
type: bug
priority: 1
assignee: coopwilliams
tags: [premerge, numerics, nonlinear]
---
# Detect when float32 sigma points lose a direction that contributes to output variance

## Background in Plain English — WHY

At the PR 186 head, the sigma-point approximation introduced in PR 156 reports success when every evaluation point loses the same input direction to rounding. An affine function that depends on that direction then reports output variance 1 where the exact answer is 2. That can make a controller look more certain than its own model permits.

## Outcomes / Acceptance Criteria — WHAT

1. For mean (0, 64), covariance diag(1, 2^-40), output g(x) = 2^20 (x[1] − 64), and unit noise, obtain the affine variance 2 within dtype accuracy or return the documented unrepresentable-points failure. Preserve the successful linearization control.
2. In float32 and float64, check that rescaling or permuting the input directions cannot hide a lost direction. Exercise a scaled-up case and a well-resolved two-direction positive control.
3. The local status and the public solver report the failure consistently when the points lose spread. Invalid or vacuous Gaussian inputs keep their documented classification. Exact discrete zeros are outside this factor family.

## Methodological Suggestions — HOW

Use the New factor or rule family and Runtime numerical or status invariant rows in docs/ARCHITECTURE.md. Inspect nonlinear._sigma_geometry, approximation and local_result. Consider measuring the reconstruction error against the spread in the affected direction. Compare eager and JIT execution, local and grouped updates, and public return status. No serialized model changes are needed.

## Future work

Independent of the Taylor-linearized reactor pilot cfp-9806. Include the regression in the corrected numerical head’s validation.

## Definitions

Sigma points are selected input values used to estimate the mean and covariance of a transformed distribution. Affine means a linear map plus a constant. Spread means the variation represented by those points.


## Notes

**2026-11-22T02:02:35Z**

First reproduced at PR 156 head 0080676 and again at 80ae1f2. The variance was 1 in the sigma-point rule and 2 in the linearization control. The guard originates in 20bb3c2. Evidence and commands are in ~/agent-share/cffg-premerge-review/early/.

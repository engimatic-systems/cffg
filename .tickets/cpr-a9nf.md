---
id: cpr-a9nf
status: closed
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

**2026-11-22T02:30:12Z**

Change-path review before implementation. The New factor or rule family and Runtime numerical or status invariant rows apply. The nonlinear rule owns point construction for single updates, grouped updates, readouts and energy. Check eager and jitted/batched geometry, local failure and affine controls, and public auto and flood failure forwarding. Preserve the existing vacuous/improper classifications. Exercise float32 and float64, swapped axes and rescaled units. Exact discrete zeros, serialization, and current/legacy checkpoints are N/A. Baseline reproduction with uv on this branch gave five identical second coordinates of 64, sigma-point variance 1 with success, and linearization variance 2.

**2026-11-22T02:33:57Z**

The rule now measures point-rounding error in the scaled Cholesky basis of the input covariance. Each off-center point has unit displacement there, so both coordinate and correlated directions are checked against a dtype-relative tolerance. The original float32 reproducer (uv run python ~/agent-share/cffg-premerge-review/early/sigma_collapse.py) now reports represented=False and local status 2. Linearization still succeeds with affine variance 2. Both public schedules identify sensor with unrepresentable_sigma_points. Regressions cover float32 and float64, permuted axes, independent 32x and 64x unit changes, eager and JIT/vmap on correlated inputs, the resolved two-direction variance of 3, and vacuous/improper classifications. uv run pytest -q tests/test_nonlinear_rules.py tests/test_nonlinear_engine.py tests/test_adaptive_startup.py --tb=short passed 74 tests with 12 slow cases deselected in 96.49 seconds. The integrator owns the final suite gate.

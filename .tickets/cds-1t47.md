---
id: cds-1t47
status: open
deps: [cff-dlrj]
links: []
created: 2026-11-28T04:23:32Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-ahps
tags: [rslds, hybrid, derivation]
---
# Qualify a Gaussian softmax bound that preserves source odds under label replication

## Background in Plain English — WHY

D18 gives a computable upper bound on the normalizer of a destination-softmax transition. Repeating every destination with equal sublabel weights leaves the exact odds between two source regimes unchanged. At 32 destinations, the global quadratic bound instead moves a source probability from 0.40744 to 0.000335. A production factor would treat this artificial preference as evidence.

## Outcomes / Acceptance Criteria — WHAT

1. State a gauge-invariant alternative for fixed affine coefficients and a proper Gaussian drive. Prove the sign of the objective bound. Derive all coordinate messages from that same objective, including any nonconjugate solve.
2. Reproduce the D18 replication case at 2, 4 and 32 destinations. Show which quantities must remain invariant under equal replication of every destination. Compare source odds, absolute node objectives and Gaussian message precision to a quadrature oracle at small drive dimension.
3. Check weak and strong drive variation, exact categorical zeros, absent Gaussian information, float32 and float64, and K=32 with drive dimension 16. State worst-case work and temporary storage. Distinguish objective accuracy from fixed-point convergence.
4. Decide whether fixed inference merits implementation. If it does, open separate fixed-inference, learning and integration tickets under cff-ahps and assign each the validated equations and remaining gates.

## Methodological Suggestions — HOW

A candidate upper bound applies Jensen to log of the exponential sum after removing a shared affine logit. For fixed coefficients the necessary Gaussian exponential moments are analytic. Compare its induced Gaussian coordinate problem with D18 and with quadrature. Quadrature can diagnose a surrogate without certifying its sign.

## Future work

Production factors, coefficient learning and public APIs depend on the decision here. The independent binary-tree work continues under cff-0f3g.

## Definitions

A destination is a categorical outcome. A source is the preceding regime. Gauge invariance means the conditional distribution is unchanged when all destination logits gain the same affine function. Equal replication splits every destination into the same number of equally likely labeled copies.


## Notes

**2026-11-28T04:27:02Z**

Start from D18 and tests/test_destination_softmax_derivation.py. Reuse the 32-label counterexample, which isolates destination replication from genuine source evidence.

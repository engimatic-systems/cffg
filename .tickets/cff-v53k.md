---
id: cff-v53k
status: open
deps: [cff-n60h]
links: []
created: 2026-09-15T18:28:19Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-1o72
---
# Expose and differentiate the certified adaptive fixed-point system

## Background in Plain English — WHY

Neural training needs the derivative of the settled inference result. The derivative must include how changing weights and beliefs change the local Gaussian approximation.

## Outcomes / Acceptance Criteria — WHAT

1. Inference and training share one owner for chart coordinates, exact-island reconstruction, the complete residual, and the raw energy.
2. A canonical parameter plan preserves blob tying through JAX-only parameter injection.
3. A matrix-free adjoint solve certifies its original residual and rejects non-finite, exhausted, singular or unsupported solves before exposing an applicable gradient.
4. Gradients require a matching certified inference result, finite energy and fixed support; stale weights or observations cannot reuse an old certificate.
5. A scalar neural fixture with at most 20 parameters agrees with independent central differences; both local approximation methods include total coefficient derivatives.

## Methodological Suggestions — HOW

Use Runtime numerical or status invariant and New factor or rule family. Audit flooding/mixed reconstruction, host/JIT numerical kernels, ties, both dtypes and stale state. Differentiate R_u transpose lambda=E_u and dE/dtheta=E_theta-R_theta transpose lambda. Do not use the trust normal-equation subproblem as the adjoint certificate.

## Future work

cff-ll6y applies certified gradients; cff-3zzo learns proposals from certified roots.

## Definitions

Adjoint: an auxiliary linear solve for the derivative of a scalar objective. Gauge: equivalent message coordinates removed before solving. Matrix-free: derivative-vector products without a whole-graph Jacobian.

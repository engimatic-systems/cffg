---
id: cff-o7hs
status: open
deps: [cff-4e5z, cff-ri8m]
links: [cff-1o72, cff-fxmj, cff-ri8m, cff-l6i5]
created: 2026-07-08T16:49:34Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Nonlinear Gaussian factors: linearization + sigma-point moment matching (EP)

Deterministic nonlinear function nodes g(x) with Gaussian message approximations: (a) Jacobian linearization at the incoming mean (EKF-style, jax.jacfwd — free with JAX), (b) unscented/sigma-point moment matching (UKF-style), (c) this activates the reserved MOMENT_MATCHING ConstraintKind as an EP bead: site approximations refined per sweep. Care: EP messages can have negative-definite Lambda transiently — decide and document the projection policy. FE becomes approximate at these nodes; ledger labels the term. DELEGATION: Fable derives message forms + EP update/damping policy and the FE surrogate; Opus implements sigma-point utilities + tests vs Monte Carlo ground truth; Sonnet surveys how RxInfer/ForneyLab handle nonlinear node FE (specific questions, short report distilled into this ticket).

## Acceptance Criteria

Nonlinear observation model (e.g. range-bearing) posterior matches dense-grid/MC ground truth within documented tolerance; EP iteration stable with damping; MOMENT_MATCHING bead validated end to end.


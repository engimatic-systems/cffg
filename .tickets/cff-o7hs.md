---
id: cff-o7hs
status: open
deps: [cff-4e5z, cff-ri8m, cff-209l]
links: [cff-1o72, cff-fxmj, cff-ri8m, cff-l6i5, cff-pksr, cff-lquu, cff-k9ws, cff-209l]
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


## Notes

**2026-07-15T01:45:14Z**

Scope boundary (2026-07-14, from the recurrence-link design discussion): this ticket remains valid for its own domain — deterministic nonlinear maps between GAUSSIAN edges (nonlinear dynamics/emissions, EKF/UKF perception), with the approximate FE term labeled in the ledger as already specified. It is explicitly NOT the route for continuous→discrete links (e.g. a recurrence factor p(z'|z,x) with a softmax/stick-breaking link): those get a dedicated hybrid factor with a provable variational bound (Pólya-Gamma / Jaakkola-Jordan), per the cff-fxmj audit directive to prefer declared subclasses with bounds over universal approximate machinery. Epistemic (GFE) scoring through nonlinear nodes stays out of scope here (cff-dfpi owns that question).

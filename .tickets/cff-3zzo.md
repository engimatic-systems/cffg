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

## Notes

**2026-09-15T18:56:17Z**

Freeze paired qualification before fitting the recognition model: scalar prior N(0,1), equality-shared input, two observed neural sensors with theta=[0.8,0,1.25,0], linearization, positive output variance0.4, observations (t+0.15,t-0.15). Training t=-0.4+0.05*j, j=0..16; held-out32 JAX Threefry2x32 uniform draws key3201 on [-0.4,0.4]. Recognition is an affine output layer over powers0..7 of s=(y1+y2)/0.8, least-squares fitted to certified roots in one Gaussian chart anchored at t=0. Use dtype-scaled rank admission without ridge. Paired solves use25 iterations and tolerance1e-8. Every pair must converge and all Gaussian means/covariances agree atol=rtol=1e-6; energy difference<=1e-6*(1+abs(F)). Primary work counter is outer_map_evaluations, report iterations separately. Bootstrap paired differences with NumPy default_rng3202,10000 resamples of32 complete pairs, linear quantiles0.025/0.975; lower95% endpoint must exceed zero; never drop failures. Research source: agent pilot checked three neutral roots in4/3/4 iterations against independent reduced-site roots; confidence high for model suitability, learned proposal and CI remain unqualified.

---
id: cff-4ejb
status: closed
deps: [cff-gsd5]
links: []
created: 2026-09-15T17:06:22Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-o7hs
---
# Globalize adaptive Gaussian fixed points and label their energy

## Background in Plain English — WHY

Nonlinear Gaussian approximations change with their input beliefs. Even a tree therefore requires a consistent fixed point. A frozen Gaussian model has a Bethe energy, but its adaptive updates need not minimize the total changing energy.

## Outcomes / Acceptance Criteria — WHAT

1. Configured iterative Gaussian factors use the shared nonlinear callbacks in flooding and mixed schedules, including exact batch and scan islands attached to them.
2. Runs globalize the fully recomputed fixed-point residual with matrix-free derivatives, accept only proper finite beliefs, and require finite small residual and energy change to converge.
3. Local numerical failures, rejected trials, exhausted budgets, and vacuous final beliefs remain distinguishable in run diagnostics and readouts.
4. The ledger labels each approximate term, its normalized frozen Gaussian convention, conditional stationarity, and absence of an established evidence bound.
5. Affine and curved models agree across scheduling and local host/JIT/vmap execution; float32, weak/strong precision, and an uncomfortable-scale case have executable evidence.
6. Public construction, equations, execution integration, resource accounting, and limitations are documented with test anchors.

## Methodological Suggestions — HOW

- Applicable ARCHITECTURE change paths: New factor or rule family; Runtime numerical or status invariant. Audit flooding/scheduled batch/scan, host/JIT, single/vmap/Colony, ledger/readouts, and estimator. Latest/legacy schema belongs to cff-gsd5; support on discrete edges is N/A for this Gaussian factor.
- Reuse the existing gauge-reduced trust-region residual solver and its matrix-free linearization. Disable uncertified Picard candidate advancement for these factors. Initialize from actual messages without a made-up mean or covariance floor.
- Full host run control follows existing GFE behavior; local updates and trust kernels remain JIT/vmap compatible. No whole-run JIT claim is implied.

## Future work

cff-1o72 differentiates the full adaptive fixed point for neural mean training and amortized initialization. Exact zero noise remains cff-u06e.

## Definitions

- Fixed point: messages whose next full update returns the same messages.
- Globalization: shrinking or rejecting a proposed update using a scalar residual merit and propriety checks.
- Matrix-free: applying a derivative to a vector without constructing the whole graph Jacobian.
- Bethe energy: the sum of local belief energy terms and shared-edge entropy corrections.
- Surrogate: an explicitly named approximation to an objective.
- Conditional stationarity: stationary energy with approximation coefficients held fixed, separate from coefficient self-consistency.

## Notes

**2026-09-15T17:25:25Z**

Research question: does globalization survive genuinely changing nonlinear cavities? Answer: an equality joins two cubic sensors with opposing Gaussian evidence, so each cavity includes the other site. Independent NumPy/SciPy four-site roots give latent variances 0.156421040551 (Taylor) and 0.139520070577 (sigma) for evidence +/-0.3, conditional/evidence variances0.2, prior N(0,1). Sources: independent GPT-6 Astra xhigh calculations and nonlinear_fixture.adaptive_pair. Confidence high. Strong opposing evidence +/-1.5 with variances0.05 exposed non-finite shadow residuals entering least squares; invalid history is now discarded. Initially exact equality equations also received inverse-epsilon weights; adaptive Gaussian solves now retain unit residual weights after Gaussian whitening. Final-head focused engine validation:19 tests passed in75.25s, covering both methods/routes, changing-cavity roots, rejected-budget outcomes, affine full energy, invalid beliefs, float32 units, distinct JIT/vmap members, and Colony. Exact attached scan coverage and full parent gate remain.

**2026-09-15T17:27:52Z**

Final focused integration check:42 tests passed in61.64 seconds, including the attached eight-link Gaussian scan island versus flooding, Gaussian scheduled kernels, shared execution adapters, and existing run strategies. The complete nonlinear engine file passed19 tests in75.25 seconds before the final metadata/scan additions; those additions passed in the42-test check. Full short-suite certification is still due on parent cff-o7hs. Global objective_contract now marks adaptive Gaussian energy approximate, while node metadata retains analytic labels elsewhere. Prior precision rank checks scale with actual matrix magnitude, so float32 changes of physical units do not manufacture impropriety. Implementation is complete; parent owns final suite, runtime budget, and PR publication.

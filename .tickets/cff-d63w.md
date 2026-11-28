---
id: cff-d63w
status: in_progress
deps: [cff-qu2j]
links: [cff-wt1c]
created: 2026-09-30T21:28:34Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym, numerics]
---
# Reduce avoidable float32 Gaussian posterior and readout accuracy loss

## Background in Plain English

The saved first-order PCGym controller now passes its exact-message certificate in float32. That does not mean its posterior is as accurate as it could be. On retained candidate v3a2, CFFG's largest mean error is about 1.17e-4; an independent float32 Kalman smoother is about 4.57e-7. Both use the same runtime-rounded model. Reading CFFG's computed information parameters in float64 reduces its mean error to about 3.32e-5, so both message arithmetic and readout arithmetic contribute.

The certificate problem itself is covered by cff-qu2j and was repaired by exclusive equality cavities in cff-vxza. This ticket concerns avoidable posterior and objective accuracy loss, not relaxing convergence tests.

The fully unobserved variant exposes a larger objective error: its normalized Gaussian tree has exact free energy zero, but the float32 ledger returns about -0.204666. Its posterior-moment check still passes. A narrowly marked, strict expected-failure regression retains this mechanism in tests/test_first_order_precision.py.

## Outcomes / Acceptance Criteria

1. The retained three-dimensional, past-4/horizon-15 fixture compares source rounding, message arithmetic and readout arithmetic independently in both precisions.

2. Gaussian mean readouts avoid unnecessary loss from explicitly multiplying a computed covariance by the information vector.

3. On the retained conditioned and fully unobserved fixtures, mean error scaled by posterior uncertainty, relative covariance spectral error and relative evidence-weight error are at most the square root of the dtype's machine epsilon. This independent accuracy target preserves at least half the dtype's available precision in dimensionless posterior quantities; it is not inferred from message consistency.

4. Changes preserve proper and partially informative Gaussian semantics, exact zero-information messages, loud invalid-input handling and the existing unregularized mathematical model.

5. Float32, float64, sparse information, improper inputs and at least a tenfold horizon pass focused checks; host/compiled, flooding/exact and single/batched paths agree within their declared accuracy contract.

6. No global convergence threshold is weakened to make an inaccurate posterior appear successful.

## Methodological Suggestions

Optional: start with direct Cholesky solves for Gaussian moments and compare to the current inverse-times-vector implementation. Then isolate repeated covariance/precision conversions before changing linear-message formulas. Use an independent NumPy Joseph-form Kalman filter and smoother, dense conditioning and backward-error diagnostics. Architecture change paths: Factor rule or energy, Runtime numerical or status invariant. Inventory gauss, efrules, readouts and registered-kernel statistics before implementation. An alternative sound method is acceptable.

## Future work

- cff-wt1c — Report the selected precision and remaining accuracy limits in matched PCGym comparisons.

## Definitions

Float32/float64: 32-/64-bit floating-point arithmetic. Posterior: the distribution after conditioning on data. Information parameters: a Gaussian precision matrix and precision-weighted mean vector. Kalman smoother: a linear-Gaussian reference calculation using all observations. Joseph form: a covariance update arranged to reduce cancellation. Cholesky solve: a linear solve using a positive-definite matrix factorization. Backward error: how much the model equations must change to explain a computed answer. Free energy: CFFG's inference and model-scoring objective. Certificate: a consistency check for the computed messages. Compiled/batched: JAX-compiled and vectorized execution. Machine epsilon: the spacing of representable numbers near one. Spectral error: the largest directional error after covariance scaling. Relative evidence-weight error: the absolute difference from one of exp(reference free energy minus computed free energy).


## Notes

**2026-09-30T21:29:17Z**

Research question -> answer -> sources -> confidence: independent f32 Joseph Kalman/RTS vs runtime-rounded dense conditioning on retained v3a2: absmean4.567e-7,abscov1.682e-7,whitenedmean1.239e-5,whitenedcov3.881e-5,FEerror-9.923e-7. CFFG: absmean1.165e-4,abscov3.074e-5,whitenedmean4.482e-4,whitenedcov6.490e-5,FEerror-7.919e-5. Reading computed CFFG naturals in f64 reduces absmean to3.320e-5; directf32Cholesky solve gives7.983e-5. Equilibrated global backwarderror KF2.303e-8 vs CFFG9.966e-6. Sources: tests/first_order_fixture.py; gauss.linear_forward/linear_backward, efrules.GAUSSIAN.moments, readouts/gaussian.py. Bounded read-only independent agent calculation; high confidence on mechanism separation, not a universal condition-number cutoff. Legend prerequisite --> dependent: [in_progress]cff-qu2j --> [ ]cff-d63w; downstream matched-comparison reporting cff-wt1c. Precision improvement is not part of qu2j certificate closure.

**2026-09-30T21:34:47Z**

New hostile-case evidence, same accuracy scope before implementation: setting all20 DATA ports to None leaves a normalized Gaussian generative tree with exact -log Z=0. Both precisions certify; posterior-scaled moments pass. f32 current ledger gives -0.2046661376953125; f64 passes zero-energy accuracy. Added a strict expected-failure energy-only opt-in regression alongside a passing separate moment/vacuous/improper-readout check, so the issue is not hidden by a broadened tolerance or an entire-test expected failure. Completion must remove that marker and restore the analytical identity, including reported RunResult.free_energies. This discovered accuracy ticket remains separate from the cff-qu2j PR and is not closed with it.

**2026-11-28T04:26:08Z**

Start on published main a774a54. Review Factor rule or energy and Runtime numerical or status invariant. Trace affine and prior messages, Gaussian natural-to-moment conversion, node-local Gaussian energy, equality and edge entropy, exact/flood executors, full/selected host readouts and Colony. Retain strict sparse, vacuous, invalid and derivative behavior. Compare both dtypes to dense source-rounded and rounded-message controls, unobserved evidence zero, and the 10x horizon. No changes to statuses, tolerances, persistence, or IR planned.

**2026-11-28T04:44:38Z**

Integrated onto review base #208 (d07e8cf), whose Gaussian numerical sources match published main a774a54. On v3a2, present-main f32 already certifies with residual zero. Relative errors (whitened mean, spectral covariance, evidence weight) are (4.314e-4,2.043e-4,4.760e-4) conditioned and (3.015e-9,5.368e-5,0.21126) unobserved. Unobserved F is +0.237320. Replacing the forward or backward messages separately with a float64 oracle gives f32-backward-only wm=4.306e-4, forward-only 6.714e-6. Summing/solving the computed naturals in float64 gives wm=4.315e-4, so current host readout is no longer the principal error. With Cholesky channels, QR proper prediction, direct EF moment solves and conditional Gaussian KL ledger, conditioned errors are (1.044e-4,2.102e-5,1.236e-5); unobserved errors (6.55e-11,2.17e-6,3.05e-5). At horizon 150, wm/wc/weight conditioned (7.314e-5,3.090e-5,2.836e-4), unobserved (8.56e-10,5.842e-5,0). All pass sqrt(eps) in f32/f64. The rank guard checks whether input and projected output likelihoods span every input direction before accepting a partial joint. No rank floor enters a solved precision. Retained accuracy/certification suite including all slow checks: 15 passed. Gaussian rules, tap oracle, readiness, mixture and nonlinear local suites: 114 passed. Reproduce with uv run pytest -q -o addopts= tests/test_first_order_precision.py and the named test_gauss_rules.py, test_gaussian_tap_oracle.py, test_gaussian_readiness.py, test_mixg.py, test_nonlinear_rules.py files. Study artifacts live in ~/agent-share/cff-d63w. Queued A/B and final canonical gate pending; leave ticket in progress until gate.

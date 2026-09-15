---
id: cff-gsd5
status: closed
deps: [cff-2dzo]
links: []
created: 2026-09-15T16:50:20Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-o7hs
---
# Add configured nonlinear Gaussian means and local approximation rules

## Background in Plain English — WHY

Nonlinear Gaussian factors need portable function identity and parameters, plus a mathematically explicit approximation to their noisy conditional. This ticket implements local Taylor and sigma-point Gaussian approximations before global solver integration.

## Outcomes / Acceptance Criteria — WHAT

1. A pure-data mean-function declaration records namespaced identity, compatible version, configuration and approximation method; core loading/editing/hashing stays JAX-free and blobs hold numeric parameters and positive noise covariance.
2. Compiler-side resolution groups equivalent immutable configurations while varying parameter arrays retain a leading member axis. Missing/incompatible implementations, malformed configuration, function shape mismatch and invalid noise produce specific errors.
3. Taylor evaluates the cavity mean and Jacobian; sigma points use alpha=1, beta=2, kappa=0 and construct residual covariance from weighted outer products without a ridge. Both return Gaussian tilted moments and sites via cavity subtraction, plus the normalized frozen-factor Gaussian Bethe node energy.
4. Affine messages, marginals and node energy agree with gaussian_linear in float32 and float64. A declared curved fixture compares posterior error to an independently refined dense grid before final acceptance.
5. Vacuous input produces neutral dependent messages; invalid required cavities/covariances, non-finite function values and singular noise produce truthful diagnostics and non-finite energy. Indefinite finite sites are allowed when their combined beliefs are proper.
6. Host/JIT and single-instance/vmap results agree; dimension-32 or 128-factor evidence demonstrates bounded local matrix storage. JSON and bundle reconstruction preserve function identity, configuration, values and compiled outputs.

## Methodological Suggestions — HOW

- Use New IR field and New factor or rule family in ARCHITECTURE. Audit core serialization/edits/plates/composition, topology, configured registry snapshots, source requirements, grouping/packing, runtime adapters, readouts, manifests and estimator accounting.
- Flooding/scheduled globalization is owned by a separate child of cff-o7hs; local callbacks must be shared. Current/legacy decoding is affected and requires explicit errors. Sparse discrete support is N/A on nondegenerate Gaussian edges; float32, vacuity and uncomfortable scale are acceptance criteria above.
- Record equations in DERIVATIONS and rule docstrings, including conditional frozen-coefficient stationarity and the full adaptive fixed-point contract. Do not claim total adaptive-energy stationarity.

## Future work

- cff-o7hs integrates globalized residual/status handling, schedule classification and labeled objective metadata. cff-1o72 supplies neural architectures and full-fixed-point gradient training.

## Definitions

- Cavity: the belief with the current local site removed.
- Site: an unnormalized Gaussian local factor represented by information vector and precision.
- Sigma points: a deterministic weighted set that reproduces selected Gaussian moments.
- Configuration: portable static function settings, separate from numeric parameter blobs.
- Surrogate: the explicitly named approximate energy used by the local Gaussian model.

## Notes

**2026-09-15T16:59:41Z**

Freeze curved-posterior qualification before production comparison: X~N(0.25,0.5), g(X)=X+0.3 X^3, conditional noise variance 0.16, and Gaussian evidence N(Y;0.8,0.09). Use float64 trapezoidal reference on [-8,8] with 131,073 points and 65,537-point refinement. Primary metric is Hellinger distance between the normalized dense-grid input posterior and the inferred Gaussian. Thresholds: Taylor <=0.16; sigma-point <=0.10; sigma error <= Taylor error plus the maximum measured coarse/fine Hellinger discrepancy. Those discrepancies measure grid sensitivity, not a rigorous error bound. Exploratory screening by GPT-6 Astra xhigh considered thirteen setups before selecting this monotone curved sensor; a wider-prior cubic case made sigma points worse, so no universal superiority is claimed. Independent exploratory means/variances: reference 0.516872942816/0.111931710783, Taylor 0.606501293489/0.154735136965, sigma 0.510302853248/0.137267504139. Exploratory Hellinger 0.148485457385 and 0.086808425279. Final acceptance must use production rules, not these exploratory values.

Derivative research: proper numeric gauss.moments derivatives agree with analytic scalar derivatives. Keep boolean propriety/status outside the differentiated numerical output (use auxiliary data); a zero derivative on an invalid NaN branch does not certify a valid state. Sources: gauss.moments, efrules.GaussianFamily, independent temporary finite-difference and JAX experiments. Confidence: high. Full fixed-point derivatives must recompute mean coefficients, never freeze them.

**2026-09-15T17:18:46Z**

Local math and configured source/schema tests pass: 35 tests in 17.91 seconds, including core runtime-import exclusion. Research question: does local arithmetic survive strong float32 priors and large offsets? Answer: naive cavity subtraction erased weak sites; Schur elimination fixes cancellation. Computed covariance/message validation rejects underflow, and sigma geometry rejects unrepresentable spread. Sources: nonlinear.py, independent scalar affine experiments by GPT-6 Astra xhigh, tests/test_nonlinear_rules.py. Confidence high on reproduced mechanisms. Portable FunctionSpec JSON/.cffgz and grouping round-trips preserve separate parameter rows; the compatibility source-shape helper now resolves configured bindings. Persistent storage matches both actual State leaves and an independent scalar/dimension-32 formula. D13 records equations and conditional stationarity. Runtime final gate remains owned by cff-4ejb and parent cff-o7hs.

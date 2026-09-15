---
id: cff-gsd5
status: in_progress
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

---
id: cff-fc2y
status: open
deps: [cff-7kpx]
links: []
created: 2026-09-25T11:23:56Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-toqa
tags: [pcgym, nonlinear]
---
# Classify partial Gaussian input readiness without inventing a belief

## Background in Plain English

The reconstructed CSTR planning chain rejects a partially informed Gaussian input before the forward prior arrives. Its precision can be positive semidefinite with a null direction, even though the complete model has a proper posterior. Such an input is not a normalized Gaussian and cannot be propagated using a pseudoinverse covariance. A later bounded initializer needs a trustworthy classification, separate from the ordinary nonlinear rule.

## Outcomes / Acceptance Criteria

1. A small shared classifier distinguishes proper input information, finite partial information with flat uninformed directions, and invalid input information.
2. Partial information is admissible for future startup deferral only when precision is symmetric positive semidefinite and the information vector lies in its informed range. Negative curvature, nonfinite values and incompatible nullspace information are invalid.
3. Numerical guards scale with dtype and coordinate magnitudes. No global precision floor, ridge or fabricated missing-direction mean is introduced.
4. The classifier does not construct a normalized belief or certify noise, parameters, output potentials or physical-function evaluations. Those remain separate factor obligations.
5. Float32/float64, exact vacuity, rotated nullspaces, mixed coordinate units, nearly dependent directions, and at least tenfold dimension/component scale have explicit tested outcomes.
6. Host/JIT and single/vmap classifications agree. The helper is nondifferentiated admission metadata; normal nonlinear equations and the differentiated residual map remain unchanged.
7. DERIVATIONS explains the classification and why a pseudoinverse covariance is not a substitute probability distribution.

## Methodological Suggestions

Another sound method is acceptable. Use the New factor or rule family and Runtime numerical/status architecture rows. Reuse strict propriety admission without tightening unrelated Gaussian rules. Derive range and numerical-rank guards before implementation; compare rotated analytic potentials with independent NumPy spectral checks. Flood/scheduled policy and current/legacy checkpoint decoding are N/A for this classifier-only step. Run focused hostile-input checks and one final default suite.

## Future work

cff-36x7 consumes this metadata in the bounded startup runtime. Parent cff-toqa delivers the supported nonlinear planning contract; cff-wt1c later measures matched control outcomes. This ticket alone does not make a previously failing plan succeed.

## Definitions

Precision is the inverse covariance for a proper Gaussian. Positive semidefinite means no negative quadratic direction. A nullspace is the set of directions with no quadratic information; the informed range is its complement for a symmetric precision. A pseudoinverse inverts only nonzero directions. Dtype is numeric precision; JIT compiles a function; vmap batches it. CSTR is a continuous stirred-tank reactor.

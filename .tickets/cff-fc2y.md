---
id: cff-fc2y
status: closed
deps: [cff-7kpx, cff-xrut]
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
2. Partial information is admissible for future startup deferral only when it is numerically consistent with symmetric positive-semidefinite precision and an information vector in its informed range. Resolved negative curvature, nonfinite values and resolved incompatible nullspace information are invalid. The documented resolution band is not an exact-rank theorem and cannot authorize a belief or convergence.
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

## Notes

**2026-09-25T11:26:39Z**

Pre-implementation handoff: New factor/rule family and Runtime numerical/status rows. New internal numerical classifier only; no registry, message, energy, scheduling or serialization changes in this ticket. Host/JIT and single/vmap apply; flood/scheduled and current/legacy decoding checked N/A until cff-36x7. Preserve READY exactly when the existing strict Gaussian propriety predicate and finite information vector admit the input. The fallback classifies only rank-deficient partial information; an otherwise full-rank matrix rejected by the existing predicate cannot be relabeled partial. Proposed coordinate normalization: Lambda=D C D using square roots of positive diagonals, with exact zero rows/coordinates checked separately; classify the transformed vector D^-1 eta against the retained informed subspace. Classification is nondifferentiated metadata and never constructs moments. Guards and their numerical-resolution envelope must be justified before publication; side review is checking mixed-unit and rotated-nullspace cases. Invalid model noise/function parameters remain separate factor obligations for the runtime ticket.

**2026-09-25T11:30:25Z**

Review narrowed the claim: floating-point spectral/range tolerances certify numerical consistency at stated resolution, not exact PSD/range membership. Tiny negative eigenvalues and tiny incompatible null components can be indistinguishable from roundoff. Preserve strict READY first; retain exact zero-row/eta checks; validate the equilibrated matrix; include eigendecomposition residual and coordinate-normalization error in the nullspace uncertainty. Refuse poorly separated fallback subspaces instead of widening compatibility to order-one errors. Mantissa/exponent normalization avoids subtracting large logs when scaling D^-1 eta. An exactly missing coordinate with a proper informed principal block can be classified structurally, without a spectral rank cutoff on its weak proper directions. Sources: failure_provenance counterexamples and bounded guard review; high confidence. The helper never constructs moments, and deferred startup cannot certify convergence.

**2026-09-30T16:03:05Z**

Independent review: question — can fallback arithmetic erase real information? Answer — JAX CPU comparisons/frexp flush stored subnormal eta and precision to zero in both f32/f64; eta=[tiny/2,-tiny/2] with ones(2,2) loses an entirely incompatible nullspace direction. Integer-bit checks now conservatively reject every subnormal fallback entry while preserving READY and exact signed-zero behavior. Sources: readiness_review minimal host/JIT probes, test_gaussian_readiness regression. High confidence. This refusal is documented in D13 and does not fabricate a belief.

**2026-09-30T16:04:05Z**

Implemented internal classifier with strict READY parity, coordinate equilibration, frexp-scaled information direction, exact missing-coordinate checks, residual/orthogonality-aware eigenspace resolution and conservative subnormal fallback refusal. The helper remains unused by the solver until cff-36x7. Focused validation after numerical changes: uv run pytest -q tests/test_gaussian_readiness.py tests/test_nonlinear_rules.py tests/test_cstr_planning.py — 54 passed, 17 deselected; final readiness-file check after added guard assertions — 26 passed. D13 and ARCHITECTURE describe the numerical band and ownership. Final default-suite gate remains pending.

**2026-09-30T16:35:31Z**

Final validation: default suite 2,610 passed, 135 deselected, wall 1,254.848 seconds; dashboard refreshed at /Users/ravd/agent-share/cffg-tests.html. cff-xrut marker-only deferral estimates 1,188.602 seconds and reuses the passing result; no unchanged rerun. Independent review found no additional normal-input defect after subnormal guard, including mixed-unit JIT/vmap dimensions 4/40. Classifier complete; next work cff-36x7 consumes metadata, controls once-only bounded startup and must explicitly exercise the deferred nonlinear integration regressions relevant to its change.

---
id: cff-ej67
status: open
deps: [cff-ocxl]
links: [cff-ffsg]
created: 2026-09-14T04:34:41Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-zov5
tags: [rslds, inference, hybrid, numerical]
---
# Preserve small Gaussian message increments through hybrid projection

## Background in Plain English — WHY

Outgoing Gaussian messages can be small differences between large precisions. Review of cff-b0s3 found that rounding a projected covariance before forming this difference loses useful information. A one-ulp eager/JIT covariance difference also becomes a message error of 0.0009765625 for a cavity precision of 10000, exceeding the existing arithmetic tolerance while both statuses report success.

The translated affine repair is committed in the same branch. This ticket isolates the newly diagnosed projection and quotient mechanism for a fresh implementation worker. A bounded independent prototype retains a covariance correction and meets the existing tolerance, so an ordinary numerical repair is justified before requesting any contract decision from Cooper.

## Outcomes / Acceptance Criteria — WHAT

1. The repeated local projection, canonical conversion, and signed quotient path preserves small outgoing message increments for the retained-region reproductions from cff-b0s3 review. All message blocks in a common fixed basis satisfy the existing atol=rtol=512*4*eps(dtype) against an independent reference computed from the identical stored region. Host/JIT and single/vmap satisfy the same budget; translation invariance remains covered separately with matching execution modes.
2. Covariance projection retains sufficient numerical information until the precision difference is formed. If a high part and correction are stored, document their exact meaning and propagate them through both marginal directions, ordinary Gaussian/categorical adapters, repeated message evaluation, objective/statistics consumers, and differentiation. Stored metadata must not silently change the represented belief. Exposed physical covariance remains consistent within its dtype rounding allowance.
3. Precision and linear differences are computed without first rounding away the small increment through avoidable inverse/subtract cancellation. Constants and gauges preserve labeled normalization and the same D12 potential. Signed or vacuous cavities remain legal when the complete belief is proper; no clamp changes a message or precision to force success.
4. Independent references reproduce the exact stored-region counterexample, both marginal directions, multiple labels, d=2 off-diagonal covariance, float32 with x64 disabled, float64, and at least one composed repeated evaluation. Check probability, centered mean, covariance, all outgoing coefficients, full free energy, auxiliary targets, and learning statistics. Mathematical/finite-difference derivative checks cover any new retained covariance path. The D12 temporal approximation falsifier and registered recovery thresholds remain unchanged.
5. The planning inventory states behavior at exact log zeros, empty support, support changes, tiny-positive underflow, malformed/non-finite inputs, improper complete beliefs, float32, and 10–100 times the small fixture scale. Literal absence stays distinct from arithmetic failure. Dtype-scaled guards fail explicitly when a required residual or solve cannot be represented; no hidden tolerance increase is permitted.
6. The storage inventory counts every retained correction and temporary in local T=100, K=20, and d=10 shapes. Storage stays polynomial in the existing O(T*K*K*d*d) grouped-belief and O(T*K*d*d) message bounds. No global covariance, dense global Jacobian, unbounded tape, or path enumeration is introduced outside tiny references. cff-a3wz still measures the complete iterative buffers separately.
7. Before implementation, record the applicable Derived component coordinate, New factor or rule family, and Runtime numerical or status invariant architecture rows and affected duplicated local surfaces. Compiler, schedules, production lifecycle, persistence, public learning activation, and source-recovery fits are N/A. Update existing D12 derivations/docstrings/architecture and dashboard registration in place.
8. Focused tests and independent review pass after the final relevant edit. One combined final short-suite run at the proposed PR head may satisfy this ticket, cff-b0s3, and cff-ffsg. Keep their completion blocked until this numerical defect is resolved. If bounded diagnosis and ordinary fixes instead require a mathematics, supported-scope, or threshold revision, return evidence and a recommendation to Cooper before proceeding.

## Methodological Suggestions — HOW

These are optional implementation guides, not new approximation semantics.

- Inspect /tmp/cff-b0s3-quotient-stages.py, /tmp/cff-b0s3-quotient-candidates.py, and /tmp/cff-b0s3-quotient-split.py, then preserve the independent reference and mechanism in asserted repository tests.
- Accumulate projected covariance around a retained conditional reference: C0 plus weighted conditional-covariance differences plus the centered mean variance. Form the quotient through C*J_out = I - C*J_cavity, assembling the right side from C0 and the retained correction before they are combined into one rounded covariance.
- Use compensated affine arithmetic from cff-b0s3 for cancellation-sensitive local products. An equally sound simpler approach is welcome. Retaining a low part is a numerical representation of the same covariance, not an extra stationarity variable.

## Future work

cff-b0s3 and cff-ffsg finish the local kernel gate after this repair. cff-a3wz then qualifies the complete message and auxiliary solver with frozen tiny/moderate cases and actual iterative storage. Compiler integration and cff-x0if registered recovery remain downstream and unchanged.

## Definitions

- D12: the hybrid-regime-ec-v1 approximation specified in docs/DERIVATIONS.md.
- Projection: replacing a local mixture by its declared label-conditioned mean and covariance.
- Natural coefficients: the constant, linear coefficient, and precision matrix of a quadratic log potential.
- Precision: the inverse of a Gaussian covariance.
- Cavity: the incoming product excluding the outgoing message being computed.
- Quotient: the projected belief divided by its cavity, represented by subtracting natural coefficients.
- ULP: one spacing between adjacent floating-point values at the stated scale.
- High part and correction: two retained arrays whose sum represents one mathematical quantity more accurately than one rounded array.
- JIT/vmap: JAX compiled and batched execution.
- Epsilon: the dtype machine resolution near one.



## Notes

**2026-09-14T04:35:11Z**

Completed independent review at 8e201c3: identical stored regions give projected source covariance 9.999987378250808e-05 eager versus 9.999988105846569e-05 JIT. Identical projected beliefs give matching canonical/quotient outputs, locating parity loss in collapse_label. For the eager covariance, exact inverse-minus-cavity is 0.012621765123... while direct conversion returns 0.013671875. Reference-covariance summation removes that parity difference; compensated inverse-difference residual assembly reduces error for identical stored covariance to about 1.8e-9. However, rounding reference+correction into one array leaves about 0.00035775 error against projection from the original region. Retaining the correction through residual assembly gives eager/JIT 0.0129798809066 versus exact rational stored-region values 0.0129795203159 and 0.0129795192562, errors below 3.62e-7. Sources: independent b0s3_reviewer /tmp stage/candidate/split scripts named above, hybrid_natural.py collapse_label/canonical/quotient. Confidence high for mechanism and measured scalar prototype; general directions/adapters/derivatives still need implementation validation. The predecessor's fixed-origin and compensated-affine code is available in branch ancestry; this ticket depends on accepted ocxl mathematics, not completion of b0s3, avoiding a dependency cycle. Fresh worker starts after the predecessor finishes its separate underflow correction and review.

**2026-09-14T04:35:36Z**

Dependency refinement built from tk dep tree --full cff-a3wz, tk ready, and tk blocked. Legend: prerequisite --> dependent; tk records are authoritative.

[x] cff-ocxl
  +--> [ ] cff-ej67 (ready; next fresh implementation worker)
  |       --> [~] cff-b0s3 (completion blocked; affine underflow fix in progress)
  |              --> [~] cff-ffsg
  |                     --> [ ] cff-a3wz (complete solver checkpoint)
  +--> [ ] cff-k6g5 (parallel ready declaration root, implementation deferred)
       a3wz + k6g5 --> [ ] cff-7sw5 --> [ ] cff-idhk --> [ ] cff-brcf
                           --> [ ] cff-f3jz --> learning/composition
                                 --> [ ] cff-x0if --> [ ] cff-zov5

The new dependency represents completion order; all predecessor local code is already committed in the shared branch. Implementation remains sequential. No numerical code changed in this refinement; tk dep cycle and git diff --check are the artifact checks.

**2026-09-14T04:49:58Z**

Predecessor underflow correction is committed at 2b7fd93, clean tree; independent re-review completed with no findings. Fresh implementation may now start on this bounded projection/quotient repair. Fixed origins and compensated affine code are available in ancestry; use current APIs and preserve their explicit underflow boundary.

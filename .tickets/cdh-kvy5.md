---
id: cdh-kvy5
status: open
deps: [cdh-urn0]
links: [cdh-urn0]
created: 2026-09-14T23:56:51Z
type: bug
priority: 2
assignee: coopwilliams
parent: cff-zov5
tags: [hybrid, numerical]
---
# Audit recomputed affine inputs at native binary exponent boundaries

## Background in Plain English — WHY

The bounded cdh-urn0 repair centers D12 mixtures through compensated affine evaluation. Its diagnosis also found a smaller composed-affine expression whose result changes when callers request different intermediate outputs. This needs a separate audit before anyone claims that arbitrary generated inputs to the compensated helpers have consistent stored-word semantics. It is not evidence that built-in D9 or fit_rslds uses this unbound D12 path.

On native float32 CPU, compute a weighted conditional mean, subtract it using ordinary floating subtraction, square the differences through affine, and decode those squares for another weighted affine sum. For the preserved K4 weighted fixture, returning the square yields 0.0625 with bits 1031798784. Returning decoded mantissa and exponent separately yields mantissa 0 and exponent 122 for one entry, corresponding to 0.03125 rather than exponent 123. The weighted sum can then lose half a component contribution. The cdh-urn0 _mix repair avoids that ordinary-subtraction composition; the general helper audit is not included in its scope.

## Outcomes / Acceptance Criteria — WHAT

1. Preserve a bounded local reproducer for the inconsistent native decoded fields and determine which operand is recomputed differently, using operation-level evidence. Do not name a compiler fault or contraction pass without proof.
2. Inventory actual D12 callers that can generate affected helper inputs before proposing any shared-primitive change. Keep mathematical identities, approximation, support, dtype and thresholds unchanged.
3. Specify the supported rounding boundary and test it on native host/JIT and single/vmap paths, analytic forward/reverse derivatives, nonfinite/subnormal inputs, powers-of-two neighbors and a bounded larger domain. No universal exactness claim is required.
4. If ordinary repairs cannot meet the existing numerical requirements, record exact evidence and pause for root to bring the required decision to Cooper. Do not change qualification semantics or run a complete optimizer as a diagnostic.

## Methodological Suggestions — HOW

Reproduce with `uv run python /tmp/cdh-urn0-decode.py` in cffg-d12-harness. It builds p=arange(1,17,float32).reshape(4,4)/sum(p), m=(arange(64,float32).reshape(4,4,4)-32)[...,:2]/16, then mean=affine(p[...,None,:],swapaxes(m,-1,-2))/sum(p,axis=1)[...,None], c=m-mean[:,None], and v=affine(c[..., :,None,None],c[...,None,:,None]). Compare independently compiled outputs v, bitcast(v,uint32), and the pair (bits & 0x7fffff, (bits >> 23) & 255). The differing exponent occurs at [0,3,0,0]. Returning extra arrays changes the execution context, so preserve separate-output cases.

The optimized local HLO saved at /tmp/cdh-urn0-words-hlo.txt rewrites mean division as reciprocal multiplication and distributes decoding over multiple fusions. That is evidence of execution context, not yet identification of the exact transformation. /tmp/cdh-urn0-boundaries.py found ordinary optimization barriers ineffective. No whole-program compilation or solve was used.

## Future work

cdh-urn0 owns the narrow weighted/rare projection repair and independent review. cdh-m75a owns the separate fixed-coordinate reference adapter; cdc-ztuf owns harness integration, and cff-a3wz owns later qualification. This audit authorizes none of their solve, storage-compilation, public-integration or evidence-revision gates.

## Definitions

Affine means a weighted sum plus offsets. A mantissa holds a floating-point number's significant binary digits; the exponent scales them by a power of two. A binade is the range between two adjacent powers of two. JIT is just-in-time compilation, and vmap is JAX batching. A fusion combines compiler operations into one execution kernel. A stored word is one float32 value with a single consistent bit pattern.


## Notes

**2026-09-15T00:10:10Z**

Independent source inventory at published d22a43a, accounting for the active branch's compensated _mix centering repair. Question: which actual D12 paths outside that repaired subtraction feed generated values into weighted/compensated products? Answer: there are bounded candidates, but this read-only inventory establishes no additional failure.

Priority1: hybrid_separator._conditional_at_risk.merge (205) divides to obtain mixture fractions, subtracts m-old_mean, and forms ordinary f_old*f_new*delta*delta.T. The updated mean later enters centered_affine in auxiliary_statistics (255) and transition_energy (291). The square itself is ordinary multiplication, not integer-significand _two_product. This path affects xi targets, auxiliary certificates and FE even without learning output.

Priority2: hybrid_natural._mix's divided projected mean enters from_centered_moments (341), where center+mean is squared by _second_moment (159) through rounded_product. This is a finiteness/status check, not the repaired projected covariance; the same helper supports Belief.second readouts. A composed evaluation may retain the preceding mean division in the same program.

Priority3: rebase (222) computes new_origin-old_origin before precision contractions through affine. Belief.relative_mean (115) computes displacement+(center-requested_center) before expectation's compensated precision contraction (143). These reach cavities/messages and site FE; fixed-origin cases can have exactly zero origin differences, so displaced-origin cases are the useful separate boundary.

Lower priority: hybrid_covariance.resolved (38) divides covariance words by power-of-two scaling before compensated matrix products in spectral checks; quotient (451) passes -mean/2 into affine. Such binary scaling differs materially from subtractive cancellation. Remaining _mix covariance subtraction is an additive offset; divided increments enter combine via multiplication by one, neither is the repaired centered self-product. Learning precision (hybrid_separator239) uses an ordinary outer product of source-relative means and does not directly consume the suffix covariance.

Separately executed kernels receive materialized arrays; nested Python/jit helpers alone do not establish that boundary inside joint.evaluate (hybrid_joint219). Sources above are src/cffg files in cffg-d12-scaling. Confidence high for reachability and distinctions, untested for numerical/compiler failure. No edits, tests or probes occurred. Use this to bound the separate audit or independent fixed-coordinate checks; it is not evidence by itself for revising scope, tolerances or blocking qualification.

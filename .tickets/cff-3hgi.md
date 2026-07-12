---
id: cff-3hgi
status: open
deps: []
links: [cff-fxmj, cff-l6i5]
created: 2026-07-10T18:06:32Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-ahps
tags: [adversarial-reviewer, numerics, gaussian]
---
# Gaussian BP uses an absolute ridge and treats negative precisions as valid beliefs

ADVERSARIAL REVIEWER FINDING (2026-07-10). gauss.py uses _RIDGE=1e-10 in every solve/inverse, contrary to the dtype-scaled guard contract and without matrix-scale normalization. This breaks an exact message invariant: with invertible A and a vacuous input, linear_forward must emit a vacuous output for every noise scale. For W=1e-12 it emits Lambda=9.901e-13 (99.0% of W) in both f32/f64; at W=1e-20 it emits 100% of W. A downstream weak likelihood at the same scale therefore receives roughly double the correct precision. At W=1 f32 the ridge disappears while f64 emits 1e-10, so behavior is dtype- and unit-dependent. Separately _neg_entropy discards slogdet's sign: Lambda=[[-1]] returns finite -1.41894, and gauss.moments returns covariance=[[-1]], rather than marking an improper belief. Cancellation in W-WA solve(...) and user-mutated runtime params can therefore produce plausible finite FE/readouts from invalid Gaussian states.

## Acceptance Criteria

1. Replace absolute ridge/PD tolerances with dtype- and matrix-scale-derived guards whose protected invariant is documented; use stable factorizations/pseudoinverse limits where exact vacuous propagation requires them. 2. linear_forward emits exactly/tolerance-vacuous messages across >=12 orders of precision scale in f32/f64 and under coordinate rescaling. 3. Every Gaussian energy/readout checks symmetry and positive definiteness using sign+logdet/Cholesky; improper beliefs yield +inf/status or explicit improper readouts, never negative covariance or finite entropy. 4. Hostile tests cover sparse/vacuous inputs, ill conditioning, rank deficiency, float32, d>=100, and mixed scales. 5. Kalman, hybrid, BFE tree identity, and conjugate learning parity remain green; quantify any deliberate regularization bias.


## Notes

**2026-07-10T18:08:57Z**

ADDITIONAL REPRO: mixg._pd_ok is dtype-derived but not matrix-scale-relative: it requires lambda_min > sqrt(eps). A valid scalar precision 1e-4 is rejected in float32 (threshold ~3.45e-4) but accepted in float64; 1e-6 likewise. Thus diffuse but proper hybrid components fall back to vacuous messages/+inf solely by dtype/units. Include mixg._pd_ok/_ridge and relative Cholesky guards in this ticket, not just gauss.py.

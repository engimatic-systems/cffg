---
id: cff-rgty
status: closed
deps: []
links: [cff-6tek]
created: 2026-08-09T04:22:43Z
type: bug
priority: 2
assignee: coopwilliams
tags: [gaussian, numerics, learning]
---
# Stabilize MNIW scatter symmetry after large-statistic cancellation

## Background in Plain English — WHY

A valid observed-only rSLDS coordinate pools hundreds of proper Gaussian-mixture statistics. The MNIW update forms its posterior scatter by subtracting two large matrix products. Roundoff leaves an antisymmetric residue near 1e-13 even though the symmetrized scatter is well-conditioned and positive definite. The current check scales its symmetry tolerance from the small final scatter, rejects the valid coordinate, and blocks cff-l7y4. Generic MNIW posterior algebra in `cffg.conj` owns this defect; the rSLDS fitter must not weaken its improper-statistics checks to hide it.

## Outcomes / Acceptance Criteria — WHAT

1. MNIW updates accept a posterior scatter whose antisymmetric residue is within a dtype-derived bound scaled to the matrix operations that produced it.
2. MNIW updates still reject materially asymmetric, non-finite, or non-positive-definite statistics.
3. The accepted posterior scatter is explicitly symmetric and positive definite before rollover.
4. A float64 regression reproduces the 240-step NASCAR coordinate mechanism: final scatter scale near one, intermediate products much larger, and antisymmetric roundoff near 1e-13.
5. A float32 regression exercises the same cancellation mechanism with a dtype-appropriate bound.
6. A materially asymmetric hostile input remains loud in both dtypes.
7. Existing NIW, MNIW, Gaussian-learning, and rSLDS fitting tests pass.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound method that satisfies the outcomes.

- Use the **Runtime numerical or status invariant** change-path row for the dtype-scaled guard and the **Learnable or checkpoint feature** row for the MNIW consumer. The authority is `src/cffg/conj.py`; `src/cffg/gaussian_learning.py` should continue to delegate posterior algebra.
- Bound acceptable antisymmetry from the magnitudes of the additive and subtractive terms, then symmetrize the accepted mathematical covariance before the SPD check. Do not add an absolute ridge, covariance floor, or probability floor.
- Host/JIT is N/A because MNIW updates are NumPy host coordinates. Flooding/scheduled and single-instance/`vmap` are N/A because the defect occurs after engine readout pooling. Current/legacy checkpoint decoding is unchanged.

## Future work

No future work is identified. cff-l7y4 resumes its pilot after this generic blocker lands.

## Definitions

- **Antisymmetric residue:** The difference between a computed matrix and its transpose, caused here by floating-point evaluation order.
- **MNIW:** Matrix-Normal-Inverse-Wishart, the conjugate posterior used for a linear matrix, offset, and covariance.
- **Posterior scatter:** The positive-definite matrix hyperparameter updated from pooled second moments.
- **SPD:** Symmetric positive definite.


## Notes

**2026-08-09T04:25:33Z**

Completed 2026-08-08. The MNIW scatter update now bounds antisymmetric roundoff from the summed magnitudes of the four matrix terms that undergo cancellation, using the active dtype epsilon and matrix dimension. It symmetrizes only a residue inside that bound, then retains the existing positive-definite check. No ridge or covariance floor was added. The 240-step NASCAR mechanism had final scatter scale 1.1–3.3, intermediate sufficient statistics near 4,000, and antisymmetric residue 1.1e-13–2.3e-13; the old final-scatter tolerance rejected three of four regimes. Fixed float64 statistics reproduce that cancellation. The float32 case injects 1e-4 skew within its operation-scaled rounding bound, while 1.0 skew is rejected in both dtypes. Focused validation: 37 passed and 1 slow test deselected across test_conj.py, test_gaussian_learnables.py, and test_rslds_fit.py.

Retained wiggle-audit candidates and meanings: generic=the MNIW family-wide host algebra; N/A=the named structurally irrelevant JIT, schedule, batching, and checkpoint surfaces; same=the cancellation mechanism in both dtypes; remaining=continued loud rejection; appropriate=bound derived from the active dtype; material=skew above the explicit operation-scaled tolerance; near=the stated numeric orders of magnitude; check=runtime input guard; pass=automated pytest success; reproduce=fixed-statistic regression; sound=an optional method satisfying the outcomes; stable=numerical stability; test=the named automated files; fit=the cff-l7y4 fixed-dataset loop; learn=the MNIW parameter-update primitive; learnable=the architecture row; rollover=point-parameter installation; scale=numeric magnitude; small=final numeric magnitude; may=method permission; should=design intent; explicit_owner=cffg.conj; only=the observation-only boundary; current=the pre-fix guard and checkpoint pair; existing=the named regression suites. Capability ownership was checked separately. Intelligible-ticket review: complete.

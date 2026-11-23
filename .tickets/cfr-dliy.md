---
id: cfr-dliy
status: closed
deps: []
links: []
created: 2026-11-23T03:13:48Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [premerge, numerics, readout]
---
# Reject rank-deficient beliefs in single-edge and bulk Gaussian readouts

## Background in Plain English — WHY

A controller can ask the engine for one Gaussian edge or for every edge. At the cumulative PR 195 head, the single-edge call returned a mean above one billion and a finite covariance after one observation of three linear combinations of four unknowns. The fourth direction had no information. The bulk call correctly returned NaNs for that same state in float32. A different singular observation can fool both calls in float64.

## Outcomes / Acceptance Criteria — WHAT

1. Both public readouts return all-NaN means and covariances for the observed rank-deficiency mechanism in float32 and float64.
2. Both public readouts retain the dtype, shape, and accurate unregularized moments of proper beliefs, including correlated and mixed-scale coordinates.
3. Vacuous, indefinite, asymmetric, and nonfinite information fails explicitly in both public calls.
4. The single-edge call examines only its named edge. The bulk call keeps the existing Gaussian edge names and agrees with its previous numerical results on proper inputs.
5. The focused validation records single-edge cold and warm timings in the same process conditions as a baseline.

## Methodological Suggestions — HOW

The architecture index row is “Runtime numerical or status invariant.” The affected duplicate surfaces are Engine.gaussian_marginal and Engine.gaussian_marginals, both host readouts in readouts/gaussian.py. Colony callers use these same Engine methods. JIT, vmap, engine steps, scheduling, persistence, and bounded-action scoring are N/A. Compare the coordinate-scaled positive-definiteness test with cfr-zc4v while retaining host execution. Exact zero information from rank-deficient sensors and vacuous messages must stay detectable. Check float32 and a mixed-scale case at 10 times the fixture dimension.

## Future work

cfr-yuhl owns directed certification. cfr-zc4v owns admission for bounded decisions. Neither depends on this readout.

## Definitions

A Gaussian mean locates the center of a belief and its covariance measures uncertainty. Precision is inverse covariance. A rank-deficient precision leaves a direction without enough information to define a Gaussian distribution. NaN is the floating-point marker for an undefined value. JIT compiles a calculation. Vmap applies it to a batch.


## Notes

**2026-11-23T03:22:20Z**

At 5c4c43b, a float32 three-row sensor with rows [-6,2,-4,10], [11,11,3,-7], [-4,13,-6,15] produced exactly H.T @ H after one public step. The selected edge reported a mean component of 1.247e9 and maximum covariance 2.594e15; the bulk path returned NaN. The new cases also cover a float64 exact null direction and a rounded float32 sensor Gram with a positive eigenvalue only after promotion.

Both APIs now use the same NumPy host readout. The single-edge path indexes one edge and bulk obtains the arenas once. Both check diagonal-normalized precision at its stored dtype, solve the unshifted system in float64, and validate the covariance after casting back. This follows the dimension-times-machine-epsilon margin used by the bounded solver owner at cfr-zc4v. An unresolved matrix in either direction returns NaN. No JAX dispatch is needed for a host readout.

Final focused validation: uv run --frozen --no-sync pytest -q tests/test_readout_owners.py tests/test_kalman.py -k "not renders and not engine_contains_no_family_rule_or_local_readout_calculation" passed 35 cases. The remaining facade ownership test passed separately after the independent Engine correction. Three gallery render cases were deselected. Five new cases cover both dtypes, positive definiteness after conversion, partial observation, recovery after prior propagation, and forty-dimensional correlated mixed-scale beliefs. A five-step conditioned Kalman model gave the same 18 edge names, shapes, and dtypes as the original #195 bulk implementation. Maximum abs(new-old)/(1+abs(old)) was 1.04e-6 in float32 and 8.30e-16 in float64.

A local single-threaded CPU microbenchmark queried one proper 4D s4 edge in the P4/H15 fixture using fresh processes before and after. First call and subsequent 1000-call median measured 196/40.8 microseconds originally and 206/58.9 microseconds with both guards. These are host readout times only. tk dep cycle and git diff --check passed. The parent owns final integration and suite validation.

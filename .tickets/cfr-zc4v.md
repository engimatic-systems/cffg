---
id: cfr-zc4v
status: closed
deps: []
links: []
created: 2026-11-23T03:03:35Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [premerge, numerics, bounds]
---
# Reject a singular joint Gaussian consistently in compiled bounded decisions

## Background in Plain English — WHY
The #190 bounded-action API can issue a command and free energy from an invalid joint Gaussian under JAX compilation. The same input fails the ordinary call. This came from a direct probe at 5c4c43b.

## Outcomes / Acceptance Criteria — WHAT
1. For BoundedGaussianPlan((1,),(-1.,),(1.,)), zero mean of length two and baseline energy zero, test float64 covariance [[9,3],[3,1]] and float32 covariance [[9,15],[15,25]]. Both have determinant exactly zero. At the reviewed head solve_joint returns INVALID_GAUSSIAN eagerly and SUCCESS under jax.jit, with an apparent free energy near 1.342785.
2. Reject invalid joint covariance under both execution modes. Return the documented failure diagnostics and preserve valid, correlated inputs in both dtypes.
3. Include near-singular valid and invalid cases, rescaled coordinates, and state lifting. Check the public bounded region readout which calls the compiled dense method. Existing empty-support and nonfinite behavior must still hold.

## Methodological Suggestions — HOW
Review the runtime numerical or status invariant row in docs/ARCHITECTURE.md. Inspect bounded_actions.py:158–160 and its conditional covariance check around line 213. A tiny positive remainder after cancellation can pass a factorization. Validate the original joint geometry against a scale- and dtype-aware criterion. Check eager, jit and batched behavior.

## Future work
cff-fl7h owns the original bounded-region scope. Larger action blocks and new reactor benchmarks are separate.

## Definitions
A singular covariance has a direction of exactly zero variance. JIT compiles the same JAX function before evaluating it.

## Notes

**2026-11-23T03:10:00Z**

Change path: Runtime numerical or status invariant (bounded D15 decision). Check dense and factorized covariance entry points, lifted state moments, eager/JIT/vmap, and BoundedChainRegion.evaluate staging. Engine scheduling, serialization, model graph and suite configuration N/A. Test both dtypes and coordinate signs/scales, exact-zero covariance directions, small positive and negative conditional variances, and existing empty or nonfinite support. The physical baseline requires a proper Gaussian.

**2026-11-23T03:17:03Z**

Confirmed the reported rank-one matrices at f70f8eef gave INVALID_GAUSSIAN eagerly and SUCCESS when staged. The dense solve now tests positive definiteness in correlation units after reserving dimension times machine epsilon on the diagonal, and derives the gain and conditional covariance from the unshifted joint Cholesky factor. This admission test also covers independently supplied covariance blocks. If a valid conditional variance disappears when assembling the output joint, the result is NUMERICAL_FAILURE with masked values. Empty boxes retain their original status priority.

**2026-11-23T03:17:03Z**

Validation: uv run --frozen --no-sync pytest -q tests/test_bounded_action_numerics.py tests/test_bounded_action_derivation.py tests/test_bounded_chain.py --tb=short passed 70 tests in 27.80 s, including the graph-backed public decision. A separate CPU probe with JAX_ENABLE_X64=0 returned INVALID/SUCCESS/INVALID for exact rank one and positive/negative variance perturbations. The regressions cover eager, jit, compiled vmap, signed unit changes, exact and nearly singular matrices, and a proper block with 47 correlated nuisance coordinates. git diff --check passed. The parent owns the integration suite.

**2026-11-23T03:17:03Z**

Independent numerical review approved the implementation. Initial review prompted removal of an excessive norm multiplier and a check for rank lost in the lifted output. The reviewer verified 48- and 256-coordinate equicorrelated matrices against analytical covariance (max absolute error 1.39e-7 float32 and 4.45e-16 float64), both ticket examples, and independently probed 1,024 singular integer Gram matrices across dimensions 3–48. Cost: same-process alternating CPU float64 JIT calls, 180 warmed synchronized solves per implementation per shape, covariance 0.6^|i-j|, finite box +/-1. Median microseconds for one bounded coordinate at dimensions 2/28/80 were 65.1/69.4/145.5 before and 65.4/76.2/197.0 after. For two bounded coordinates they were 277.3/301.8/363.8 before and 255.0/275.4/378.3 after. These are local kernel timings; episode timing is outside scope.

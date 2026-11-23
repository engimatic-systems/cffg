---
id: cfr-zc4v
status: open
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

---
id: cff-qr92
status: closed
deps: []
links: []
created: 2026-07-30T01:09:38Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-mwjo
---
# Make GFE Newton convergence floor cover recomputation roundoff

## Plain English

The GFE inner solver will recognize a stationary solution when its residual is at the reproducible float64 recomputation floor instead of reporting budget exhaustion one rounding unit above the current cutoff. The guard remains derived from machine precision, while a factored exact-support regression proves that the relaxed certificate does not hide a moving outer message state.

## Technical description

A cff-2gev numerical-feasibility check reaches stable finite outer messages but `_gfe_solve_result` plateaus at residual 1.28e-13 to 1.70e-13, just above the current `512 * eps(float64) = 1.14e-13` cutoff. Use the smallest reviewed dtype-scaled ULP allowance that covers the reproducible softmax/Jacobian recomputation plateau, preserve the distinct diverged and budget-exhausted statuses, and anchor it with the exact-support factored corridor mechanism rather than an incident-only scalar assertion.

## Hostile-input plan

Exact-zero transition/observation support: retain finite messages and certify only when both inner and outer residuals pass their dtype-scaled thresholds. Vacuous input: keep existing uniform/vacuous solver coverage. Float32: the floor must come only from `finfo(dtype)` and existing float32 hostile tests must remain finite. Scale: retain the 32-position x 4-selector GFE-TM test and ensure the new constant does not depend on the small corridor cardinality.

## Acceptance Criteria

The exact-support factored boundary-belief regression converges with finite FE/messages and `RunStatus.CONVERGED`; a zero-iteration solve remains `BUDGET_EXHAUSTED`; NaN input remains `DIVERGED`; the tolerance is derived from `jnp.finfo(dtype)` with its invariant documented; focused GFE, GFE-TM, and factored tests pass.


## Notes

**2026-07-30T01:10:39Z**

Implemented a 1024*finfo(dtype).eps Newton certificate (the smallest power-of-two ULP allowance above the observed 1.70e-13 float64 plateau), documented the invariant beside the solver, and added an exact-support 8x2 factored-corridor regression that requires both inner and outer convergence plus finite FE/messages. Focused GFE/GFE-TM/factored hostile set: 21 passed. No absolute floor was introduced; zero-iteration and NaN status tests remain in tests/test_gfe.py.

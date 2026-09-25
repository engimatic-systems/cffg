---
id: cff-qu2j
status: open
deps: []
links: []
created: 2026-09-25T03:46:25Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Resolve float32 exact Gaussian-chain certification failures against an independent oracle

## Background in Plain English

Replaying the saved first-order controller with its normalized observation [-1,-0.6] gives budget_exhausted in float32 after one exact pass (residual 0.0006103515625), while float64 converges (about 9.1e-13). This is evidence for numerical investigation, not a reason to declare all Gaussian models float64-only or loosen every certificate.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. The saved three-dimensional, past-4/horizon-15 problem is reduced to a self-contained numerical regression.

2. Posterior means, covariance and free energy are checked against an independent dense-Gaussian or Kalman reference in both dtypes.

3. The investigation distinguishes message recomputation error, conditioning and genuine certificate failure, and records the resulting supported precision contract.

4. A correction preserves protections against stale schedules and unrelated scale inflation in tests/test_schedule.py; a non-root cannot become converged by relaxing a global threshold.

5. Sparse/zero information, improper/vacuous messages and a horizon at least ten times larger are exercised with dtype-derived guards.

6. README and LIMITATIONS explain any remaining precision sensitivity and how to select precision before constructing arrays.

## Methodological Suggestions

Inspect exact_strategy.py and Gaussian information-form operations. Architecture: Runtime numerical or status invariant. Pairs: host/JIT checks, flooding/exact scheduled comparison, Engine/Colony parity; checkpoint N/A. Retain independent moment/energy checks before accepting a certificate change.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-wt1c — Validate the PCGym follow-up chain with matched control and runtime comparisons.

## Definitions

Float32/float64: 32-/64-bit floating-point numbers. Certificate: a numerical check supporting convergence. Kalman reference: an independent linear-Gaussian filtering/smoothing calculation. Residual: the measured mismatch used to assess convergence.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

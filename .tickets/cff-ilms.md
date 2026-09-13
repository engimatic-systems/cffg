---
id: cff-ilms
status: in_progress
deps: []
links: []
created: 2026-09-13T15:56:39Z
type: bug
priority: 1
assignee: coopwilliams
tags: [learning, numerics]
---
# Restore component-MNIW posterior mean reporting

## Background in Plain English — WHY

The all-family bundle resume proof in cff-jilz exposed an existing failure: ComponentMNIWFamily.mean calls MNIW.mean_cov, but MNIW has no such method. Learnables.expected therefore raises AttributeError for component dynamics before it can report any parameter role.

## Outcomes / Acceptance Criteria — WHAT

1. Component-MNIW reporting returns each matrix, offset, and posterior-mean noise covariance.
2. The covariance equals psi / (nu - d - 1), where d is the output dimension, and differs from the inference rollover psi / nu.
3. A proper posterior with no finite covariance mean raises a clear ValueError instead of returning an invalid value.
4. Single and component MNIW use one covariance-mean implementation. Float32 inputs retain dtype, exact zero entries remain zero, and a 64-component case works.
5. Focused conjugate and learning tests pass, and the cff-jilz final short suite validates the integrated PR.

## Methodological Suggestions — HOW

The implementer may choose another sound method. Use the Learnable or checkpoint feature change path: MNIW owns the covariance formula and family adapters expose it. Message updates, free energy, checkpoint formats, and compiler plans are unchanged. Host/JIT, flooding/scheduled, single/vmap and current/legacy decoding changes are N/A because this fixes a host-side reporting method. Test the undefined-mean domain separately from the valid distribution domain; no floor or ridge is needed.

## Future work

This fixes the immediate blocker to cff-jilz reporting-mean equivalence. No further work is identified.

## Definitions

- MNIW: matrix-normal inverse-Wishart distribution over a linear map and its noise covariance.
- Posterior mean: the distribution expectation used for reporting.
- Rollover: the parameter value installed into the next inference run, which uses inverse expected precision for noise covariance.


## Notes

**2026-09-13T15:58:58Z**

Fixed the missing MNIW covariance-mean method and made ordinary MNIW reporting delegate to the same implementation already called by component MNIW. The analytic covariance regression distinguishes reporting from rollover, rejects undefined means, preserves float32 and literal zeros, and exercises 64 components. Focused validation: 48 tests passed, 1 slow test deselected across test_conj.py, test_hybrid_learnables.py and test_gaussian_learnables.py. The integrated cff-jilz short suite remains the final gate.

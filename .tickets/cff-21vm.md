---
id: cff-21vm
status: open
deps: []
links: []
created: 2026-09-25T03:46:26Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Add reproducible joint NIW and MNIW posterior sampling

## Background in Plain English

The experiment hand-wrote inverse-Wishart sampling for uncertainty-aware action selection. The conjugate posterior classes expose updates and point summaries but not joint parameter draws. Reusable sampling should preserve the correlation between covariance and mean or regression matrix.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. NIW and MNIW expose joint draws with explicit random keys and documented scalar/batch shapes.

2. Draws follow the declared posterior, including conditional mean/regression covariance rather than independently sampled marginals.

3. Seed reproducibility, independent keys and joint moment checks agree with an independent distributional reference where those moments exist.

4. Invalid degrees of freedom, shapes, non-finite scales and non-positive-definite inputs fail explicitly; proper heavy-tailed posteriors are not rejected merely because a moment is undefined.

5. Float32 and larger matrix dimensions use dtype-aware validation without fixed tiny ridges.

6. A standalone example distinguishes draws from expected-precision rollover and posterior mean covariance; public exports are tested.

## Methodological Suggestions

Derive/verify the standard joint sampling contract in a docstring or DERIVATIONS if not already covered; use uv for dependencies. Architecture: Learnable or checkpoint feature. Pairs host/JIT equality if API jittable (otherwise document host only); flooding/scheduled N/A; single/vmap batched draws; current/legacy checkpoint N/A (posterior schema unchanged). Zero support/improper cases are covered by parameter validation.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-wt1c — Validate the PCGym follow-up chain with matched control and runtime comparisons.

## Definitions

NIW: normal-inverse-Wishart distribution over Gaussian mean and covariance. MNIW: matrix-normal-inverse-Wishart distribution over regression matrix and noise covariance. Joint draw: correlated parameters sampled together. Rollover: substituting expected parameters for the next inference call.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

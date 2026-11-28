---
id: cff-21vm
status: closed
deps: []
links: [car-ne0a, cff-pyy9]
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

## Notes

**2026-11-28T04:03:07Z**

Architecture handoff: Learnable or checkpoint feature, host conjugate classes only. Add keyed scalar and batched draws using a shared inverse-Wishart covariance factor and conditional Gaussian noise. Explicit JAX keys, source-dtype scale validation, finite and proper hyperparameters, positive-definite returned covariance, fractional degrees of freedom including undefined covariance moments. Test translated and disparate units, correlated regression columns, invalid/nonfinite scales and sparse singular matrices, float32 and float64, higher dimensions and empty sample batches. Sampling runs on the host between trials. JIT/vmap, flooding vs scheduled, Engine/Colony, learnable fields, and current/legacy checkpoint schemas are unaffected. Parent coordinates numerical runs and owns the combined gate.

**2026-11-28T04:14:28Z**

Delivered NIW.sample(key, sample_shape=()) and MNIW.sample(key, sample_shape=()) through cffg.conj and the package facade, plus examples/joint_gaussian_draws.py. The host call derives a local PCG64 stream from one scalar JAX key. A reverse Bartlett factor draws the covariance and supplies the same factor to conditional mean or regression noise. Source-dtype correlation admission precedes scaled float64 solves. Unrepresentable returned draws fail explicitly instead of resampling and biasing the distribution. No new package or checkpoint schema.

Verification on a774a54 plus this change: with UV_PROJECT_ENVIRONMENT=/Users/ravd/projects/cffg/.venv and PYTHONPATH=src, run uv run --frozen --no-sync python -m pytest -o addopts= -q tests/test_conj.py tests/test_public_api.py::test_top_level_facade_reexports_only_canonical_owners tests/test_gaussian_agent.py::test_unresolved_gaussian_prior_at_execution_precision tests/test_hybrid_learnables.py::test_component_mniw_compact_updates_preserve_resolved_directions_and_resume. Final result 50 passed in 12.24 s. The standalone example ran. Coverage includes 8,192 joint draws per family against analytic covariance, precision and log determinant, SciPy log density, and whitened conditional moments in low/high noise groups. Replays span float32/float64, x64-disabled JAX, 32-dimensional disparate units, scalar, batch and empty draws, invalid inputs, rank loss and proper fractional tails. Parent owns the combined gate.

Oracle detail: for d=2, nu=11.5, psi=[[3,-.9],[-.9,1.5]], entropy from the normalized density is -3.463426 nats. SciPy 1.18 draws scored with its logpdf yielded -3.469799 with SE .01393. The CFFG sample yielded -3.439165 with SE .02100. scipy.stats.invwishart.entropy alone gave -2.423705 (its formula subtracts log(2) where d*log(2) is required). The checked expectation uses the analytic normalizer and trace. No upstream dependency changes.

**2026-11-28T04:33:20Z**

The integrated sampling and finite-hypothesis branch passed the complete 2,855-test short suite at 2bfe84b in 484.55 seconds wall. It is ready for review.

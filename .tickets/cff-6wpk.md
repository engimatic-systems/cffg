---
id: cff-6wpk
status: open
deps: []
links: [cff-7t4j]
created: 2026-11-24T02:37:08Z
type: bug
priority: 1
assignee: coopwilliams
tags: [learning, numerics]
---
# Preserve Gaussian spread through declared NIW and linear-regression learning

## Background in Plain English — WHY

The ordinary Gaussian and linear-Gaussian factors can infer the right variance while the registered learner receives identical evidence for two different variances. The public mixture readout fix cff-7t4j is specific to a prior mixture. The conventional NIW and MNIW families consume different statistics. These families need end-to-end coverage through Learnables.accumulate.

## Outcomes / Acceptance Criteria — WHAT

1. Run the converged public two-factor example with scalar coordinates at 4096, two independent Gaussians of variance 0.25 or 1, and a zero transition slope. Marginal variances agree with each input. With matched float64 learner priors of scale 3 and mean/intercept 4096, preserve the distinct posterior scales 3.25 and 4.0. The original float32 Engine path returns 3.0 in both families for both cases.
2. Preserve within-record input and output covariance and their cross-covariance on a correlated regression fixture. Check Gaussian prior, linear Gaussian, observed/missing linear output and switching linear components separately. Separate directly measured failures from extensions inferred by inspection.
3. Use centered statistics through the family adapters, pooling and posterior solves, including fractional component mass, zero-weight rows, stationary points, representable translations in float32/64, rank-deficient observations and invalid inputs. Test host/compiled, single/batch, supported schedules, and legacy bundle restore with one further update.
4. Compare old and new work, payload sizes and wall time before choosing a default for each family. Keep current public raw-statistic callers working.

## Methodological Suggestions — HOW

Start with the two-factor public fixture recorded in the linked notes. A correct marginal establishes the inference side of the boundary. Verify the learner separately against the analytic posterior. Reuse cff-7t4j for univariate or NIW centered pooling and derive the joint centered regression update before implementing it. Follow the architecture rows for a learnable or checkpoint feature and a registered rule. Confirm persistent semantics and migration only where they change.

## Future work

The AXIOM prior-mixture learner car-9ccf can use its own centered path. Establish the dependency before extending this ticket to an AXIOM pixel-regression implementation.

## Definitions

A posterior scale is the matrix that carries evidence about noise in the Gaussian conjugate posterior. NIW learns a mean and covariance. MNIW learns linear-regression coefficients and noise covariance. A cross-covariance measures how input and output vary together.


## Notes

**2026-11-24T02:39:18Z**

Verified through the public converged Engine and Learnables.accumulate with JAX 0.10.2 on CPU, source 8d46be4 and centered-pooling candidate b763744. Each prior and linear factor reported proper. Both scalar ground-truth marginals correctly distinguish 0.25 and 1. The two raw node records are identical and both posterior scales are 3.0. Independent callback and scalar update corroborate the mechanism. The observed-output and switching-linear callbacks use raw latent moments, with observed coordinates fixed where applicable. The adjacent AXIOM prior-mixture path has its own centered mode. Full scripts and results live under ~/agent-share/axiom-centered-checkpoint/other_families*. This follow-up is related work, not a blanket dependency on unrelated graph families.

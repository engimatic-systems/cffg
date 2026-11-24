---
id: cff-6wpk
status: in_progress
deps: []
links: [cff-7t4j, cad-1bat]
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

**2026-11-24T05:17:01Z**

Implementation handoff on fd8bfd1, fix/centered-declared-learning. Architecture rows: Learnable or checkpoint feature and New factor or rule family (readout only). Check registered prior, latent and DATA linear callbacks, specialized switching-linear callback, shared host reduction and checkpoint continuation. Flood/scheduled inference feed the same readout. Messages, FE, graph encoding, parameter addresses and disk posterior layout N/A. Cover rank-deficient point observations, exact-zero mass, missing/improper rows, both float precisions and large translated coordinates. Wall-time measurements and the short gate await a coordinated slot.

**2026-11-24T05:20:14Z**

Reproduced the two scalar float32 cases on fd8bfd1: correct marginal variances 0.25 and 1, both declared scales 3.0. Ordinary NIW now returns 3.25 and 4.0, also at float64 origin 2^27. Existing raw readout remains available. 68 focused Gaussian learner, readout ownership and registered kernel tests passed (2 opt-in cases excluded).

**2026-11-24T05:25:15Z**

Registered latent and DATA regression now use joint-centered moments and QR. Exact scales at 4096 and at 2^27: 3.25/4.0. Dense, fractional, rank-one and point conditional data agree with an independent regression oracle. 85 focused conjugate, Gaussian-learning and readout-ownership cases passed (one opt-in excluded). New P2 cad-1bat reproduces an unrepresentable column precision with two translated predictors; the learner explicitly refuses that update.

**2026-11-24T05:27:25Z**

Switching-linear centered readout is tested against the independent full Gaussian joint, including its cross-covariance and responsibility. Translated sparse f32 flood and f64 auto public learning cases passed, as did the existing K=32,d=16 write path. Fused centered callbacks are candidates for coordinated timing; raw Engine and direct raw-update APIs retain their original algorithms.

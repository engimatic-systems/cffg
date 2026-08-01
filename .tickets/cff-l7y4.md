---
id: cff-l7y4
status: open
deps: [cff-y4nx, cff-6tek]
links: [cff-y4nx]
created: 2026-08-01T05:05:47Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
tags: [rslds, validation, learning]
---
# Canonical NASCAR learned-model validation for rSLDS

## Plain English

Fit the canonical synthetic NASCAR observations with CFFG rather than supplying the generating parameters. The fitted recurrent model will be compared with a matched ordinary SLDS on the state-dependent partitions and long periodic rollouts that motivated rSLDS. The full benchmark remains an external uv project, keeping the default test suite small.

## Evidence and scope

The independent cff-y4nx consumer run at CFFG 6c01812 proved T=10,000 fixed-parameter inference but correctly declined to claim the canonical learned-model result. The packaged RSLDSLearning does not learn the shared emission, accumulates between trials rather than replacing one-sequence coordinate statistics, and provides no PCA/ARHMM-style label-breaking initializer. This ticket owns those missing learned-fit surfaces and the external matched-baseline experiment; it does not own generic non-discrete LearnableSpec registration (cff-6tek) or cold compilation performance.

## Architecture handoff

Applicable change-path row: Learnable or checkpoint feature.

- Authority: cffg.learn Family/Learnables after cff-6tek, cffg.conj MNIW, cffg.models.rslds, and the external NASCAR harness.
- Derived surfaces: Gaussian and hybrid member_of targets, factor-local emission/dynamics/recurrence statistics, replacement versus accumulation cadence, parameter writes, and model-fit status/history.
- Serialization/checkpoints/bundles: model parameters remain serialized blobs; generic fit checkpoints are N/A unless introduced explicitly.
- Duplicated pairs: host/JIT is host-only fitting around JIT inference; current/legacy checkpoints N/A; sequential/vmap N/A; mixed schedule remains the inference authority.
- Focused tests: compact synthetic replacement-coordinate recovery only. The T=10,000 NASCAR fit, SLDS baseline, and long rollouts stay outside pytest.
- Durable docs: README fit API; ARCHITECTURE cadence and initialization boundary; LIMITATIONS approximation and identifiability claims.

## Hostile inputs

- Exact-zero regime support must remain exact within inference; the fitter must not silently revive it with a probability floor.
- Improper Gaussian, recurrence, or emission statistics terminate the fit loudly and do not write parameters.
- Float32 uses dtype-derived guards in inference and finite host-side posterior validation.
- T=10,000 must reuse one compiled engine and fixed-shape state/parameter trees; no per-iteration recompile or history-sized device allocation.
- Affine latent and label non-identifiability are handled by explicit initialization/alignment, never by scoring raw labels as though identified.

## Acceptance Criteria

1. Shared gaussian_linear emission matrix, offset, and covariance learn from soft factor-local MNIW statistics alongside dynamics and recurrence parameters.
2. A single-sequence fit API replaces each variational-EM coordinate from fixed priors on every outer iteration; it does not count the same sequence repeatedly as continual evidence. Termination distinguishes converged, budget-exhausted, and diverged from the inner Engine status and a finite falsifiable objective.
3. A documented PCA plus ARHMM-equivalent or comparably justified initializer breaks affine/label symmetry without ground-truth states or regimes.
4. In the pinned stick-breaking NASCAR experiment, learned rSLDS and matched learned SLDS consume the same observations and parameter budget. Report label/affine-aligned recovery, held-out prediction, and long-rollout periodicity; rSLDS must preserve the state-dependent oval behavior materially better than SLDS with pilot-set thresholds and headroom.
5. Compact sparse-support, improper, and float32 recovery tests pass. The long experiment is external and adds no default pytest case.
6. The result names the upstream NASCAR variant, CFFG commit, seed, commands, environment, run status, and artifact provenance. A not-ready outcome remains acceptable only with a concrete new blocker.


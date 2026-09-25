---
id: cff-iplr
status: open
deps: [cff-wi2o]
links: []
created: 2026-09-25T03:46:26Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Implement the validated nonlinear parameter-particle learning contract

## Background in Plain English

After the particle design establishes its statistical and persistence contract, users need a reusable implementation for nonlinear model uncertainty. This work extends the successful finite-bank route without substituting point training or undocumented heuristic jitter.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. The public API implements exactly the approved v1 target, resampling and rejuvenation contract and rejects unsupported targets explicitly.

2. Explicit random keys reproduce draws and continuation; no hidden global random generator is used.

3. Sequential observations are counted once and posterior/evidence comparisons pass the independent design oracles.

4. Degenerate weights, exact-zero support, non-finite likelihoods, float32 and larger particle sets have explicit tested outcomes.

5. A resumed serialized model/data/checkpoint produces the same continuation as an uninterrupted run.

6. An example explains approximation, uncertainty calibration, memory costs and non-identifiability without promising exact continuous inference.

## Methodological Suggestions

Follow the preceding derivation, using existing registered function and graph likelihood authorities where appropriate. Architecture: Learnable or checkpoint feature plus New IR field as dictated by the design. Pairs host/JIT, flooding/scheduled likelihoods, single/vmap and current/legacy checkpoint required.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-wt1c — Validate the PCGym follow-up chain with matched control and runtime comparisons.

## Definitions

Target: the declared posterior distribution. Degenerate weights: probability mass concentrated on too few or no valid particles. Continuation: subsequent learning after restoring saved state.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

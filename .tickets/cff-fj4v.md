---
id: cff-fj4v
status: open
deps: []
links: []
created: 2026-09-25T03:46:26Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Publish continuous and hybrid control recipes covering experiment onboarding gaps

## Background in Plain English

The newcomer reached working inference quickly but wrote four controllers and read internals for basic continuous-learning patterns. Existing features need executable examples that connect modeling choices to public APIs and honestly distinguish supported conditioning from tight-Gaussian approximations.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. Standalone examples explain augmented [state; action] variables, future goal observations and learning only executed transitions.

2. A tied MNIW example demonstrates expected(name, role), and an observed-selector component-MNIW example uses a list of component priors and the supported DATA-emission pattern.

3. The examples show selective per-arm accumulation, status-gated decisions and a bounded categorical-action alternative.

4. Known continuous regressors have an explicit conditioning contract; a narrow Gaussian is labeled an approximation and its sensitivity is demonstrated, not called an exact clamp.

5. README maps these recipes to existing feature rows, and LIMITATIONS distinguishes missing capabilities from existing workarounds.

6. Examples run through public imports in supported precision modes and avoid long benchmark/training dependencies.

## Methodological Suggestions

Reuse the maintained examples directory; no new markdown guide. Architecture: Learnable or checkpoint feature for consumer contracts only; no new IR/rules intended. Pairs host/JIT and flooding/scheduled are exercised through public API; single/vmap N/A for standalone recipes; checkpoint example must use current supported decode, legacy N/A. Sparse selectors, missing observations and float32 outcomes are explicit.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-wdhj — Implement the continuous and hybrid rolling-agent facade.

## Definitions

Regressor: known input used to estimate a transition. Component-MNIW: one regression posterior per mixture component. Selector: categorical variable selecting a regime. Per-arm: updating only the option that was actually executed.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

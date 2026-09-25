---
id: cff-qtub
status: open
deps: [cff-v6th]
links: []
created: 2026-09-25T03:46:25Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Reuse compiled structure when Colony spawns parameter-only variants

## Background in Plain English

Colony caches Engine by structure hash but still calls compile_template for every parameter-only instance. PCGym policy evaluation repeatedly paid setup cost for an already known graph. The API docstring currently promises compilation only for a new structure.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. Spawning a validated parameter-only variant of a cached structure does not rerun structural compilation.

2. All initial parameter families use the same validation/resolution authority as fresh compilation; invalid arrays cannot poison the cache.

3. Two instances with different valid parameters retain independent state and observations and match separate fresh Engines.

4. Changes to structure, shape, approximation identity or other executable semantics cannot collide in the reuse key.

5. A benchmark separates first-structure setup, cached spawn, binding and inference for 1/40/400 policies and records compile counts and synchronized timing.

6. The exact Gaussian Engine/Colony regression stays green, including singleton and batched policies.

## Methodological Suggestions

Separate reusable structural artifacts from per-instance array resolution rather than copying an incomplete subset of groups. Architecture: Runtime numerical or status invariant plus New factor or rule family derived resolution/assembly surfaces (no new factor). Pairs host/JIT parameter parity; flooding/scheduled unchanged; single/vmap required; checkpoint N/A. Exercise sparse categorical arrays, vacuous/improper Gaussian input rejection, float32 and large policy counts.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-wt1c — Validate the PCGym follow-up chain with matched control and runtime comparisons.

## Definitions

Structure hash: identity derived from model declarations rather than parameter values. Parameter-only variant: identical executable graph with different allowed numeric arrays. Cache: reuse of a previously constructed artifact.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

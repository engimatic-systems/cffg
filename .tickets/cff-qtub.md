---
id: cff-qtub
status: in_progress
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

## Notes

**2026-11-28T04:25:12Z**

Revalidated on published main a774a54. Colony still compiles topology, validates bindings and assembles an Arena for each same-structure spawn. Worktree perf/colony-parameter-spawn. Change paths are Runtime numerical or status invariant and New factor or rule family (resolution and assembly adapters only). Checked pairs: host numeric resolution and JAX kernel shape admission, flooding and scheduled, singleton and grouped variance, and exact Gaussian. Engine sweeps and ledger terms are reused. Serialized IR, checkpoints and bundle formats are N/A. Cached plans must preserve fresh source loading, covariance and family validation, owner isolation, kernel and nonlinear declaration identity, dynamic constraint table layouts, and exact zeros.

**2026-11-28T04:33:30Z**

Implemented the cached prepare/resolve split on the published main base. A successful first spawn publishes the validated compiler plan and Engine. Later spawns consume current source declarations and run existing numeric packers and validation. Topology, observation rows, address ownership and arena assembly are reused. Registry replacement or a different JAX precision context receives its own executable; dynamic parameter shapes continue to partition run batches. If a new abstract pack layout appears, the same registered shape validators run. Changed plate initializer specs and sparse overrides are resolved live. Failed validation leaves the cache and instance IDs unchanged. Focused regression selections: 169 passed in compiler/assembly/resolution/Colony/constraint/plate files and 16 passed in final test_colony_spawn.py. Timing and combined final gate follow.

**2026-11-28T04:34:33Z**

Combined with cff-b2fz on the #208 integration branch. Cached binding uses the same shared-ownership check and copy-specific BlobRef construction as fresh grounding. An integration regression restores a scoped compact template from JSON, changes its tied initializer, overrides one copy with sparse support and binds one sensor. It compares all packed parameters, observation messages and learning plans to fresh flat compilation, checks the original member remains independent and rejects inconsistent shared source paths on a cache hit.

**2026-11-28T05:19:01Z**

Combined gate on #208 code revision 34374ae passed. The canonical dashboard recorded 2,866 passing cases, zero failures, four workers, 451.364 s wall. Isolated follow-up 1908dd3 fixes a missing import in the standalone benchmark argument parser; its --help check passes. It changes no library code, pytest configuration or outcome interpretation. The source owner is running the comparative timings separately.

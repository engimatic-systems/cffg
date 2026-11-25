---
id: cff-4yr6
status: in_progress
deps: []
links: []
created: 2026-09-25T03:46:25Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Batch observation binding while preserving partial DATA updates

## Background in Plain English

The experiment reports about 14 ms binding 20 Gaussian DATA ports versus about 5 ms of inference. StateLifecycle currently performs eager array scatters per port. Binding must become cheaper without changing cached observations, missing-data behavior or validation.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. Numeric observation and mask updates are batched rather than launching separate eager scatters per Gaussian port.

2. Omitted ports preserve prior values; None unbinds a named port; all supported domain validators retain their public errors.

3. An invalid update cannot return or mutate a partially bound state.

4. Mixed categorical/Gaussian/exponential-family batches match the existing scalar behavior, including float32, missing values and 10-100 times more ports.

5. A synchronized same-process benchmark records cold/warm binding latency, dispatch counts and hardware for 20 and 200 ports before/after; the new path reduces per-port dispatch without regressing small binds materially.

6. Preserve the existing Colony.bind delegation to Engine.bind_data while batching the underlying writes. Document and check the public update semantics on both entry points.

## Methodological Suggestions

Precompute immutable packing indexes and validate host inputs before one grouped update per storage family. Architecture: Runtime numerical or status invariant (StateLifecycle). Pairs: host/JIT packing parity; flooding/scheduled shared lifecycle; single/vmap per-instance isolation; current/legacy N/A. Sparse categorical zeros must remain exact; improper Gaussian messages are unaffected by observation storage.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-wt1c — Validate the PCGym follow-up chain with matched control and runtime comparisons.

## Definitions

DATA port: a named observed variable boundary. Scatter: an indexed array update. Mask: a flag indicating whether an observation is present. Dispatch: submitting an array operation to the execution backend.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

## Notes

**2026-11-25T06:05:43Z**

Implementation gate on feature/hierarchical-runtime. Architecture change path: runtime numerical or status invariant, StateLifecycle and public Engine/Colony binding. The host validates and packs mixed partial updates before compiled scatter. Flooding and schedules share one lifecycle. Single-instance and Colony isolation are checked. Serialization and legacy migration are unchanged. Focused checks cover 99 binary tiles (zeros, ones, missing), mixed cardinalities and vector dimensions, both precision modes, retained masked values, domain errors, and atomic rejection. Synchronized 1/20/200-port old/new measurements follow the change. Parent owns aggregate suite and dashboard.

**2026-11-25T06:13:57Z**

The host precomputes ragged segment addresses and validates submitted ports in their original order. The compiled program applies one scatter per touched state buffer. Dropped padding addresses preserve masked cached values and reuse executables as the input count varies. The lifecycle, Colony, EF, external kernel, Gaussian kernel and tree tests passed: 84 passed, 5 slow cases deselected on 51ae6a7 plus the final binding implementation. New regression cases cover 99 binary integer/None observations, prior NumPy boolean scalar indexing, float32/64, mixed ragged dimensions/EF, partial changes under JIT, no premature device call on error, and Colony instance isolation. Bind integer 0 or 1 for one-hot binary evidence. An explicit Python or NumPy boolean retains the legacy boolean mask indexing behavior. Parent runs the aggregate gate.

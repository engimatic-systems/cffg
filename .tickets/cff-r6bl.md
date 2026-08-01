---
id: cff-r6bl
status: open
deps: [cff-y4nx]
links: [cff-yl0a, cff-wbhu]
created: 2026-08-01T05:05:47Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
tags: [rslds, performance, scheduling]
---
# Reduce canonical rSLDS cold compile and setup cost

## Plain English

Canonical-length rSLDS graphs will compile in a practical fraction of the current time while preserving identical inference and certification. The work will lower repeated hybrid slices through compact staged execution instead of presenting ten thousand nearly identical Python-expanded slices to compilation.

## Evidence and scope

The independent cff-y4nx NASCAR run at CFFG 6c01812 measured 723.14 seconds in compile/setup versus 30.58 seconds in converged inference for T=10,000, with 2.03 GB peak process RSS. That makes cold structure handling the dominant cost and a practical blocker for iterative learned fitting. cff-yl0a owns exact HMM slice fusion and explicitly excludes speculative Gaussian fast paths; this ticket owns the repeated recurrence_link plus gaussian_mixture_linear plus gaussian_linear rSLDS slice. cff-wbhu remains the linked device-loop work.

## Architecture handoff

Applicable change-path row: Runtime numerical or status invariant.

- Authority: compiler hybrid/recurrence plans, schedule mixed-stage construction, and Engine mixed execution/certification.
- Derived surfaces: host/JIT, cold/warm, mixed/flood parity, parameter and observation dynamism, BFE, support checks, status, estimator, and benchmark reporting.
- Serialization and factor equations are unchanged; this is lowering only.
- Single/vmap is N/A until a batched rSLDS contract exists. Float32/x64 and changed parameters/observations are required cache-key controls.
- Focused tests use a short repeated slice and parity against generic lowering. T=10,000 timings stay in an external benchmark.
- Durable docs: ARCHITECTURE schedule/lowering ownership and the measured evaluation envelope.

## Hostile inputs

- Exact categorical zeros and recurrence supports must produce the same messages and feasibility status as generic lowering.
- Improper Gaussian beliefs, non-finite FE, and failed inner recurrence coordinates remain loud divergence.
- Float32 compilation and execution use the same dtype-derived guards and cannot reuse x64 artifacts.
- Scale cases include T=100, T=1,000, and T=10,000 with fixed parameter/state shapes; no hidden unrolling proportional to T inside one slice kernel.
- Dynamic data and learned parameters remain runtime inputs and must not be captured as compilation constants.

## Acceptance Criteria

1. A fresh-process profile separates template construction, grounding/compile, JAX compilation, inference, readout, and artifact work for T=100/1,000/10,000 on the recorded NASCAR environment.
2. T=10,000 compile/setup improves at least 5x from the recorded 723.14-second baseline on the same machine and protocol; converged inference time regresses no more than 5%.
3. Generic and lowered paths match marginals, Gaussian moments, BFE, message/support certificates, inner status, and final RunStatus on a compact oracle case.
4. Changed observations and all learned dynamics/recurrence/emission parameters reuse the compiled executable and change the posterior correctly.
5. Sparse support, improper beliefs, float32, and uncomfortable-scale runs retain truthful status and bounded memory. No test proportional to T=10,000 enters the default suite.
6. Closing evidence records commands, commits, environment, timing distributions, peak RSS, compilation counts, tk dep cycle, and git diff --check.


---
id: cff-csmz
status: closed
deps: [cff-5k0n]
links: []
created: 2026-09-25T03:46:25Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Remove measured redundant work from physical-mean training

## Background in Plain English

Physical-mean training must fit experiment budgets without sacrificing certified inference or correct gradients. The prerequisite profile determines which setup/derivative work is redundant; this ticket optimizes that measured path rather than assuming compilation is the only problem.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. Before implementation, this ticket records the measured bottleneck and a quantitative improvement target from the profiling prerequisite.

2. The optimized path meets that target on the retained fixture with synchronized cold and warm measurements.

3. Reference inference, gradients, parameter updates and objective values match the unoptimized path within declared dtype-aware tolerances.

4. Changing parameters, observations, shapes or registered function identity invalidates any affected reusable artifact correctly.

5. Failed reference/adjoint solves retain truthful statuses, including sparse/vacuous information, float32 and larger plates.

6. Current/legacy checkpoint continuation and existing neural training remain covered; documentation states any remaining cost limitations.

## Methodological Suggestions

Prefer reusing valid executable artifacts or removing duplicate solves only when the mathematical dependency allows it. Architecture: Learnable or checkpoint feature and Runtime numerical or status invariant. Pairs host/JIT required; flooding/scheduled inference certificates preserved; single/vmap as used by training; current/legacy checkpoint required.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-wt1c — Validate the PCGym follow-up chain with matched control and runtime comparisons.

## Definitions

Adjoint: derivative propagation through the inference solution. Artifact invalidation: preventing reuse when inputs affecting an executable change. Quantitative target: a numeric before/after criterion selected before optimization.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

## Notes

**2026-11-28T05:04:04Z**

Baseline handoff from closed cff-5k0n. Freeze two workloads for the optimization comparison: the default synthetic 60-transition CSTR plate in tools/benchmark_physical_mean_training.py (float64 CPU, shared theta[4], 60 rows of Gaussian nonlinear RK4 Taylor transitions and two observed coordinates, warmed after one changed-parameter step, three public training steps) and the five-transition cooling oracle in float32 and float64 (exact marginal value/directions, vacuous observations and bitwise checkpoint continuation). Neither workload uses held-out observations. Evidence c1f9a60, /Users/ravd/agent-share/cffg-physical-mean/retained-c1f9a60.json. Setup 12.854 s, including reference 11.491 s and 8.855 s reference XLA compilation. First step 18.870 s, including 14.004 s fused derivative XLA compile and 1.948 s certificate XLA compile. Warm step 76.250 ms, dominated by forward inference at 56.995 ms. The fused derivative is 11.832 ms warm and isolated exact-input GMRES replay is 6.685 ms. Use the public production steps for timing comparisons; the isolated replay diagnoses the adjoint cost. Distinguish a cold compilation intervention from a warm inference intervention and lock the chosen improvement target before the optimization starts. Preserve all statuses, certificates, objective and gradient parity.

**2026-11-28T07:08:27Z**

Registered before optimization on a0c58a4 (#231 with unchanged #216 tools). Warm public forward inference consumed 56.995 of 76.250 ms per changed-weight step. The host rebuilds Gaussian chart metrics edge by edge for three variables per transition and dispatches one scatter per tied reader in both forward and differentiated paths. Target on the locked 60-transition float64 CPU workload: reduce median public warm forward wall time by at least 15% and median total warm step by at least 10% versus the same unoptimized worktree measured back-to-back on this machine. At archived rates these correspond to 48.445 and 68.625 ms. Report all constructor, first-step and warm components, including misses, without claiming a cold speedup. One process per retained arm, no persistent JAX cache, same 1 changed-weight warmup and 3 warm steps; CPU lock, baseline then candidate. The archived #216 result supplies a separate historical comparison. Profiling the unchanged base is diagnostic only. Compare exact fixture/config and applied/status/termination/iteration counts at every step; objective, gradient, weights and residuals atol=rtol=512 machine epsilon, independent original cooling float32/64 tolerances and bitwise checkpoint checks. No reused numerical results. Architecture Runtime numerical or status invariant + Learnable or checkpoint feature: host chart and tied rows versus JIT coordinates and derivatives, ordinary and adaptive chart consumers, mixed packed row and sparse/improper inputs, repeated weight/data changes, legacy/current trainer and persistence. New solver, schedule, batching of separate rollouts, stochastic decisions and serialization format N/A. Close only on measured acceptance.

**2026-11-28T13:49:12Z**

Implemented 772a167 (equal-dimension host Gaussian metrics) and b5ab3fe (table-wise tied row binding/validation). The unchanged #216 benchmark ran on a0c58a4 then b5ab3fe under the same CPU lock in fresh float64 processes. Three synchronized warm medians: forward 53.751 to 32.268 ms, down 39.97% (target 15%); entire step 71.722 to 48.871 ms, down 31.86% (target 10%). Cold constructor 13.794 to 11.972 s, first step 17.958 to 19.209 s, warm fused adjoint derivative 11.286 to 11.784 ms. Cold compilation remains expensive. All five updates kept identical weights, statuses, inference counts and final inference energies. Maximum absolute differences in differentiated objective and gradient were 1.82e-12 and 2.91e-11, within the registered bounds. The independent float32/float64 cooling oracles passed finite differences, vacuous observations and bitwise within-version continuation. Restoring a baseline version-1 checkpoint in the candidate reproduced step-2 weights/objective exactly in both dtypes; gradient and some Adam moments differed within 512eps. No numerical results are cached between calls. Focused pytest across outer coordinates, fixed-point gradients, neural/physical training and adaptive runtime: 27 passed (11 slow deselected). Final uv run python tools/dashboard.py --run-suite: 2867 passed, 0 failures, 442.426 s. Matched receipts and comparisons: ~/agent-share/cffg-physical-training-costs/{retained-before.json,retained-after.json,comparison.json,legacy.log,suite.log}; immutable driver and inputs are included from #216. End-to-end improvements apply to this deterministic 60-transition CPU fixture. Stacked review on #231; downstream control validation remains cff-wt1c.

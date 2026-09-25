---
id: cff-5k0n
status: open
deps: []
links: []
created: 2026-09-25T03:46:25Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Measure physical-mean training setup and derivative costs on a retained fixture

## Background in Plain English

The agent reports 175 seconds constructing a trainer, 105 seconds for the first step and 5-10 seconds per warm step on 60 RK4 transitions. These are shared-machine observations, not an identified bottleneck. Optimization needs a reproducible breakdown and an independently checked gradient.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. A deterministic 60-transition physical-mean fixture is retained without PCGym or saved pickle dependencies.

2. Measurements separately report constructor/reference inference, derivative compilation, first step, warm step, forward inference and adjoint solve.

3. The report records hardware, versions, dtype, shapes, synchronization, repetitions and convergence statuses; concurrent-load timings are not used as fixed acceptance thresholds.

4. A small identifiable fixture checks directional derivatives independently; reduced objective alone is not claimed to recover non-identifiable constants.

5. The optimization ticket receives measured dominant costs and preregistered target workloads without changing objective or stopping semantics.

6. Results and reproduction command live in this ticket; no new research markdown file is created.

## Methodological Suggestions

Inspect training.py's certified reference solve and implicit.py's fixed-point derivative path. Architecture: Learnable or checkpoint feature; no semantic change intended. Pairs host/JIT derivative check; flooding/scheduled reference identified; single/vmap N/A unless used by fixture; checkpoint continuation control. Hostile zero/vacuous observations, float32 and larger transition count are measured or explicitly rejected.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-csmz — Remove measured redundant work from physical-mean training.

## Definitions

Adjoint solve: the linear solve used to propagate derivatives through a fixed point. Directional derivative: change along one parameter direction. Warm step: a training step after compilation and one-time setup.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

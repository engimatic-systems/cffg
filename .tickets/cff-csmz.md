---
id: cff-csmz
status: open
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

---
id: cff-wt1c
status: open
deps: [cff-toqa, cff-iplr, cff-fl7h, cff-wdhj, cff-csmz, cff-4yr6, cff-qtub, cff-21vm, cff-qu2j, cff-3kdk, cff-wbhu, cff-c5qf, cff-d8jj]
links: []
created: 2026-09-25T03:46:26Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Validate the PCGym follow-up chain with matched control and runtime comparisons

## Background in Plain English

PCGym results mix library behavior with information access, planner choices, model assumptions and concurrent CPU load. After the improvements land, a bounded comparison must show which capabilities and costs actually changed without treating every reward gap as a library defect.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. The comparison records exact revisions, hardware, seeds, model knowledge, setpoint preview, objective, horizon and compute budget.

2. Separate ablations isolate nonlinear planning, fixed/particle parameter uncertainty, bound-aware decisions and facade equivalence.

3. Runtime reporting separates setup, binding, inference, readouts, learning and total wall time with explicit synchronization and concurrency.

4. Numerical success includes status, posterior/energy oracle checks and failure counts, not reward alone.

5. Short deterministic smoke workloads run by default; long multi-trial experiments remain opt-in slow or external artifacts.

6. Results are summarized in this ticket and existing canonical documentation where capability claims change; no new repository session/report markdown is created.

## Methodological Suggestions

Reuse small retained fixtures before optional full PCGym seeds. Architecture: Runtime numerical or status invariant and Learnable or checkpoint feature consumer validation. Pairs host/JIT and flooding/scheduled where comparable; Engine/Colony/facade equivalence; current/resumed checkpoints. Include zero support, float32 and larger workload cases. Do not interpret PPO's end-to-end runtime as a matched inference microbenchmark.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

No additional future work is identified within this ticket; the completed outcome contributes to epic cff-xa6z.

## Definitions

Ablation: a comparison changing one component. Compute budget: allowed time or computational resources. Microbenchmark: timing one defined operation. Oracle: independent mathematical reference.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

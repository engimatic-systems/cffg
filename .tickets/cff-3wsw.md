---
id: cff-3wsw
status: open
deps: []
links: []
created: 2026-09-25T03:46:25Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Expose physical-function point training without neural-specific registration folklore

## Background in Plain English

NeuralTrainer can optimize physical constants when a registered mean function advertises network_parameters, but that undocumented neural-named metadata was discovered by reading code. Model authors need an honest public path for non-neural point training, including its lack of posterior uncertainty.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. A documented public registration contract admits a non-neural physical mean without relying on an unexplained magic metadata key.

2. A standalone physical-mean example demonstrates tied parameter updates, objective evaluation and status handling.

3. Existing neural registrations remain compatible or have an explicit migration path.

4. Checkpoint/resume preserves point parameters and optimizer continuation and never presents them as Bayesian uncertainty.

5. Validation covers malformed parameter shapes, non-finite constants, float32, identifiable/unidentifiable examples and a larger tied plate; invalid inputs fail explicitly.

6. README and LIMITATIONS distinguish expressive nonlinear dynamics, point learning and the separate finite-bank posterior capability.

## Methodological Suggestions

Inspect trainable.py:NeuralParameters.from_engine and existing MeanImplementation metadata. Architecture: Learnable or checkpoint feature; New IR field only if the registration identity changes. Pairs host/JIT parameter/gradient parity; flooding/scheduled inference reference; single/vmap tied member semantics; current/legacy checkpoint required.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

No additional future work is identified within this ticket; the completed outcome contributes to epic cff-xa6z.

## Definitions

Point estimate: one fitted parameter value rather than a distribution. Tied parameters: several factors sharing one value. Identifiability: whether the observations distinguish parameter values. Optimizer: the procedure that changes parameters to reduce an objective.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

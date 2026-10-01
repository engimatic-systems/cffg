---
id: cff-wdhj
status: open
deps: [cff-3nr4, cff-fj4v, cff-v6th, cff-ectw]
links: []
created: 2026-09-25T03:46:26Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Implement the continuous and hybrid rolling-agent facade

## Background in Plain English

A reusable agent loop should let an experiment author specify a model and policy rather than reimplement four lifecycle variants. The boundary design defines what information may be carried and how failures and executed actions are handled.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. The public facade implements the approved continuous/hybrid boundary contract with explicit observation, learning, planning, action and checkpoint steps.

2. A manually driven Engine reference and the facade produce equivalent beliefs, objective values and actions on the linear-Gaussian fixture and supported hybrid case.

3. Learning consumes actual executed transitions once, including externally bounded or overridden actions, and does not treat predicted goals as evidence.

4. Nonconverged plans follow an explicit caller-selected policy; no silent stale action is presented as a successful plan.

5. Checkpoint/resume reproduces uninterrupted operation, including posterior learning and random-key progression.

6. Sparse selectors, missing data, improper boundaries, float32 and larger horizons have tested outcomes; existing categorical agents remain compatible.

7. README provides a concise end-to-end example with public imports and links to the capability/limitation contract.

## Methodological Suggestions

Prefer one lifecycle authority over copied experiment controllers. Architecture: New IR field and Learnable or checkpoint feature; Runtime numerical or status invariant for consumer behavior. Pairs host/JIT boundary transforms; flooding/scheduled status parity; single/vmap per-agent isolation; current/legacy checkpoints required.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-wt1c — Validate the PCGym follow-up chain with matched control and runtime comparisons.

## Definitions

Facade: a small public interface coordinating existing components. Predicted goal: a planning preference, not a newly observed transition. Override: an action executed instead of the agent's proposed action.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

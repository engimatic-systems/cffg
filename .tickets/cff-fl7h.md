---
id: cff-fl7h
status: open
deps: [cff-nd1q]
links: []
created: 2026-09-25T03:46:26Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Implement bound-aware continuous planning from the approved derivation

## Background in Plain English

Once the bounded-decision mathematics is verified, model authors need to express actuator limits inside the model and receive feasible actions without relying on post-hoc clipping. Unsupported multivariate cases must remain explicit.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. The public factor/constraint and decision API implements the exact scope approved by the derivation ticket.

2. Messages, normalizers and free energy agree with the independent mathematical oracles within declared dtype-aware tolerances.

3. Accepted decisions satisfy bounds and the constrained reference case; infeasible bounds cannot return a plausible successful action.

4. Sparse/zero support, vacuous/improper inputs, float32, extreme tails and larger supported dimensions meet the numerical contract.

5. Template validation, serialization, compilation, readouts and status behavior agree across host/JIT, flooding/scheduled and single/vmap execution.

6. A runnable example and matched ablation document where bound-aware planning improves on clipping without claiming every reward gap is solved.

## Methodological Suggestions

Use the New factor or rule family and New IR field change-path rows; review all derived packing, lifecycle, ledger and readout owners. Current/legacy serialization must reject or migrate unsupported identities explicitly. Keep equation derivations separate from execution glue.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-wt1c — Validate the PCGym follow-up chain with matched control and runtime comparisons.

## Definitions

Feasible action: one satisfying the declared limits. Extreme tail: a region with very small probability before truncation. Ledger: the assembled free-energy accounting.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

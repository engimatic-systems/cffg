---
id: cff-nd1q
status: open
deps: []
links: []
created: 2026-09-25T03:46:26Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Derive a bounded continuous-decision contract with an independent planning oracle

## Background in Plain English

The experiment plans with unconstrained Gaussian actions and clips the selected action at actuator limits. Clipping one action is not solving the constrained planning problem. The reward difference motivates a capability, but does not establish that bounds explain the entire gap.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. The design specifies the first supported continuous bound family, correlation assumptions and decision readout semantics.

2. Messages, normalizers, entropy and free-energy terms are derived consistently; any projection or truncation approximation is named.

3. A small box-constrained linear-quadratic planning oracle distinguishes bounded planning from post-hoc clipping.

4. The design defines infeasible/zero-width intervals, tail probabilities, improper/vacuous incoming information, float32 and larger action/horizon dimensions.

5. A matched controller ablation fixes dynamics, observations, objective, horizon and information access so reward effects are attributable to bounds.

6. DERIVATIONS gains a plain-English companion and executable mathematical anchors; the implementation ticket receives explicit supported/rejected cases.

## Methodological Suggestions

Choose the simplest mathematically valid v1 contract; do not assume independent clipping preserves correlated planning beliefs. Architecture: New factor or rule family and Runtime numerical or status invariant. Pairs host/JIT moments/normalizers, flooding/scheduled factor placement, single/vmap local rule, checkpoint N/A until new persisted identity is specified.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-fl7h — Implement bound-aware continuous planning from the approved derivation.

## Definitions

Box bounds: a lower and upper limit on each coordinate. Truncation: restricting a distribution to an allowed region and renormalizing. Normalizer: probability mass used to normalize that restriction. Linear-quadratic: linear dynamics and a quadratic cost. Ablation: a comparison changing one component.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

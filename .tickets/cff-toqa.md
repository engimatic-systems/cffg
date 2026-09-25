---
id: cff-toqa
status: open
deps: [cff-7kpx]
links: []
created: 2026-09-25T03:46:25Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Implement the evidence-backed nonlinear planning admissibility contract

## Background in Plain English

Nonlinear factors exist, but the PCGym planning experiment could only use its physical model after external linearization. The preceding investigation must identify whether the missing capability is initialization, scheduling or a supported local approximation; this ticket implements that specific contract, not an arbitrary noise floor.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. Before code changes, this ticket records the concrete contract and retained failing fixtures produced by the investigation.

2. The supported CSTR planning cases run through gaussian_nonlinear without caller-side linearization and satisfy finite-belief, adaptive-residual and free-energy certificates.

3. Unsupported cavities receive an explicit attributable failure rather than fabricated precision or silent success.

4. Any changed message or energy equation has a derivation and plain-English explanation in DERIVATIONS with executable oracle anchors.

5. Taylor/sigma-point paths, affine limits, precision-ratio grid, float32, zero/vacuous support and uncomfortable scale satisfy the agreed contract.

6. Public examples, architecture and limitations state the supported domain and remaining approximation boundaries.

## Methodological Suggestions

Choose implementation only after the prerequisite establishes the cause. Architecture: New factor or rule family and Runtime numerical or status invariant. Pairs: host/JIT, flooding/scheduled, single/vmap required; current/legacy decoding N/A unless serialized approximation identity changes, in which case test explicit compatibility/rejection.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-wt1c — Validate the PCGym follow-up chain with matched control and runtime comparisons.

## Definitions

Admissibility contract: the inputs on which an update is mathematically defined. Adaptive residual: mismatch measured for a model whose messages depend on current beliefs. Oracle: an independent reference calculation.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

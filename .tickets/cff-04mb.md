---
id: cff-04mb
status: open
deps: []
links: []
created: 2026-09-25T03:46:24Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Report originating factor and reason for inference failures

## Background in Plain English

The nonlinear CSTR experiment had to inspect low-level readouts every sweep to discover an IMPROPER_CAVITY failure. RunResult reports an aggregate status, but not the factor and local reason that caused it. Later non-finite messages can hide the original failure.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. A failed public run exposes a stable structured reason and originating factor name when the execution path can identify one; unavailable attribution is explicit rather than guessed.

2. Diagnostics distinguish the failing phase and logical iteration from downstream non-finite propagation.

3. Adding diagnostics does not change updates, free energy, convergence thresholds, or the four public run statuses.

4. Nonlinear local failures and hybrid non-finite failures have focused regressions; simultaneous failures use a documented deterministic attribution policy.

5. Engine, singleton Colony, and batched Colony expose equivalent per-instance attribution without leaking one member's failure into another.

6. Successful runs do not perform a host readback per factor per sweep; README explains how a caller inspects a failure.

## Methodological Suggestions

Follow local status propagation into execution deltas and RunResult; preserve provenance when it is first observed. Architecture: Runtime numerical or status invariant. Pairs: host/JIT required; flooding/scheduled required; single/vmap required; current/legacy checkpoint N/A unless result persistence is introduced (avoid it). Hostile inputs: sparse support, improper/vacuous Gaussian messages, float32 and large batches must preserve truthful attribution.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-ectw — Stabilize the reproduced hybrid planning failure without changing its model.
- cff-7kpx — Capture and classify the RK4 nonlinear CSTR planning failure.

## Definitions

Cavity: the incoming information used to update a factor after excluding its own outgoing message. IMPROPER_CAVITY: a local update cannot use that incoming distribution. Provenance: where a failure originated. JIT: just-in-time compilation. vmap: JAX batching over independent instances.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

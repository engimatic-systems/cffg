---
id: cff-04mb
status: closed
deps: [cff-gsq5, cff-fbps, cff-lhbm, cff-d57a]
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

## Notes

**2026-09-25T04:23:04Z**

Research question -> answer -> source -> confidence: Can the first failed factor be recovered safely from final-state readouts? No. nonlinear.local_result already supplies per-member failure_code but nonlinear_kernels.configure and execution.registered drop it; synchronous reduction drops member identity and scheduled propagation returns messages only. Carry small integer auxiliary facts on the actual path, decode once at finalization. Source: nonlinear.py:28, nonlinear_kernels.py:75, execution/registered.py:28, execution/synchronous.py:227, schedule_program.py batch/registerized/flat/ExactUnit, iteration.py, outer_strategy.py and outer.py. Confidence: high code-level evidence. VACUOUS_INPUT has status zero and is not a failure. Raw categorical -inf can mean exact zero support. Rejected outer trials may recover and must not poison terminal provenance; fixed_point derivative consumers must exclude auxiliary integers from residual mathematics.

Split before implementation. Legend prerequisite --> dependent:
[R] cff-gsq5 local facts
  --> [B] cff-fbps schedule transport
    --> [B] cff-lhbm ordinary/exact public results
      --> [B] cff-d57a outer terminal provenance
        --> cff-04mb completion --> cff-7kpx / cff-ectw --> cff-xa6z roadmap

Every stage is also an explicit prerequisite of cff-04mb. The split is serial because each stage consumes the previous interface; other PCGym ready roots remain independent. Derived from tk dep tree --full, tk ready and tk blocked; tk dep cycle reports no cycles. No new persisted State/schema is intended.

**2026-09-25T06:52:28Z**

All four prerequisite stages are implemented and validated: cff-gsq5 local facts, cff-fbps execution transport, cff-lhbm public Engine/Colony results, cff-d57a terminal adaptive provenance. Final gate 2486 passed/82 deselected; cff-8jqy documented the permitted marker-only runtime adjustment. Public failure diagnostics retain local names/reasons/phases/attempts when known; global and derivative failures remain explicitly unowned. Success and recovery do not retain failed speculative trials. Current/legacy checkpoints intentionally omit run-local records. Next ready numerical work: cff-ectw and cff-7kpx.

---
id: cff-v6th
status: closed
deps: [cff-rnn1]
links: []
created: 2026-09-25T03:46:24Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Preserve exact Gaussian inference when no delta constraint can harden

## Background in Plain English

PCGym's two-slice Gaussian tree converges through Engine's exact schedule, but an explicit harden_last value makes it exhaust its budget. Colony passes harden_last=n_iters and inherits the failure. This blocks evaluating several Gaussian policies together even though the model has no delta constraint.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. The supplied harden_last_gaussian_chain.py becomes a repository regression with no external experiment dependency.

2. For models with no hardenable delta constraints, an irrelevant hardening request preserves exact-schedule inference, posterior moments, free energy, and termination status.

3. Standalone Engine, singleton Colony, and a same-structure multi-instance Colony produce equivalent results on the regression.

4. Real delta constraints retain soft/hard phase boundaries and hard-assignment behavior.

5. Zero iterations, explicit damping, float32, and a chain at least ten times longer have explicit regression coverage without weakening convergence assertions.

## Methodological Suggestions

Inspect Engine.run's exact dispatch and Colony.run's singleton and batched paths. Do not route every model through a new solver. Architecture: Runtime numerical or status invariant. Pairs: host/JIT status parity; flooding/scheduled dispatch preserved; single/vmap required; current/legacy checkpoint N/A (no persisted fields).

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-qtub — Reuse compiled structure when Colony spawns parameter-only variants.
- cff-wdhj — Implement the continuous and hybrid rolling-agent facade.

## Definitions

Delta constraint: a restriction to one selected value. Hardening: activating that restriction during inference. Exact schedule: a graph-specific finite message pass followed by certification. Colony: the runtime for several compatible graph instances.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

## Notes

**2026-09-25T03:47:40Z**

Research handoff (question -> answer -> source -> confidence): Why does Colony fail on an exact Gaussian graph without delta constraints? Engine.run treats explicit harden_last as a phase even when arena.delta_pairs is empty; multi-instance Colony bypasses the exact strategy altogether. Validate the argument, then normalize irrelevant hardening and reuse the exact Engine policy for each eligible Colony member. Preserve actual delta full-hardening behavior and damping/zero-budget semantics. Sources: engine.py:358-367, colony.py:121-176; supplied harden_last_gaussian_chain.py reproduces plain=converged, explicit harden=budget_exhausted, singleton Colony=budget_exhausted. Confidence: reproduced defect and source-confirmed cause. Focused regressions: tests/test_schedule.py:421 currently codifies the defect; extend test_engine_contracts and test_colony_failures for singleton/multiple distinct parameter and observation cases plus bad-member isolation. A sequential exact-member correctness path is allowed here; cff-wbhu owns compiled controller throughput.

**2026-09-25T03:51:39Z**

Implementation: Engine validates then discards hardening phases when there are no delta constraints. Engine and Colony share exact eligibility; singleton and multi-member exact groups reuse Engine propagation/certification, while ordinary iterative groups remain vmapped and genuine Colony delta constraints remain fully hardened. This is a correctness repair, not an exact-batch throughput claim. Added self-contained Gaussian goal-chain regression (2 and 20 slices; f32/f64), distinct parameters/observations, zero budget/damping, initial failure isolation and real phase flags. Updated the old no-delta HMM assertion that codified the defect. Architecture/README/LIMITATIONS explain per-instance exact execution. Focused command: uv run pytest -q tests/test_exact_colony.py tests/test_engine_contracts.py tests/test_iteration.py tests/test_colony_failures.py tests/test_colony.py tests/test_schedule.py -> 118 passed, 3 deselected in 52.26s. Original external repro now prints converged for plain/harden/Colony. git diff --check and tk dep cycle pass. Final short-suite gate pending.

**2026-09-25T03:52:05Z**

Independent review found no correctness or failure-isolation defect. It identified one coverage gap: irrelevant partial hardening also reset convergence history on ordinary mixed graphs. Added a regression comparing complete default and harden_last=1 results on the existing no-delta mixed fixture; implementation unchanged.

**2026-09-25T04:20:13Z**

Complete: focused runtime/schedule set passed 118 tests (3 deselected), final exact-colony file passed 16 tests after independent review, and dashboard-backed full short suite at 2226bea passed 2,432 tests (67 deselected, no failures). Actual measured wall time was 1,233.520s. cff-rnn1 subsequently moved two already-passing exhaustive gradient cases to opt-in slow; no numerical code/assertions changed, and the estimated remaining suite is 1,194.450s. Reused the passing result per the explicit selection-change exception. Original supplied reproduction now reports all three paths converged. No scientific objective, certificate threshold, serialization or checkpoint change. Next chain ticket is cff-04mb; cff-qtub cached spawning is now unblocked. Exact-member vectorization remains cff-wbhu performance work.

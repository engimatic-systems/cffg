---
id: cff-fbps
status: in_progress
deps: [cff-gsq5, cff-98gj]
links: []
created: 2026-09-25T04:22:00Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-04mb
tags: [pcgym, diagnostics]
---
# Carry first failure facts through synchronous and scheduled execution

## Background in Plain English

Per-member local failure facts currently disappear when sweeps take aggregate maxima and scheduled passes return only messages. Attribution must travel with actual execution rather than being guessed by recomputing the final state.

This is a prerequisite split from cff-04mb before implementation so each change remains reviewable. Evidence: current execution owners and the PCGym audit; source details are in the parent ticket's research note.

## Outcomes / Acceptance Criteria

1. Synchronous sweeps retain the first failed local cause using stable model-local factor ranks; simultaneous updates select the lowest rank deterministically.

2. Batch stages and both registerized and flat scans preserve chronological first-failure facts, including priming, propagation and exact certification.

3. Hybrid and other outgoing non-finite values are attributed before scatter loses factor identity; legitimate sparse categorical support is not called numerical failure.

4. All Sweep, Propagation and ExactUnit consumers unpack auxiliary facts without changing numerical arrays, residuals, support tests, free energy or gradients.

5. RootProgram and other derivative consumers keep the auxiliary integer facts outside residual mathematics.

6. Focused host/JIT, flooding/scheduled, exact/certification and single/vmap tests compare all pre-existing numerical outputs, including float32 and larger scans.

## Methodological Suggestions

Inspect execution/synchronous.py, schedule_program.py, schedule_contracts.py, execution/transport.py and fixed_point.py. Prefer helper owners when the existing module is large. Architecture: Runtime numerical or status invariant. Current/legacy checkpoint N/A; diagnostics stay run-local. Preserve the first local reason when subsequent messages become non-finite.

The implementer may choose another sound method. Keep the four existing run statuses and certificate thresholds unchanged. Follow focused tests with one passing final short suite; continue the PR chain from cff-v6th-exact-colony (PR #163), updating the intended preceding base before branching.

## Future work

- cff-lhbm consumes this contract; cff-04mb closes only after all four prerequisite stages are complete.

## Definitions

- Provenance: where a failure originated.
- ExecutionDelta: one factor group's outgoing messages and local numerical facts.
- JIT/vmap: compiled execution and batching over independent inputs in JAX.
- Sentinel: an explicit numeric value meaning no failure has been recorded.
- Cavity: incoming factor information excluding its own outgoing message.
- Factor rank: deterministic order of complete model-local names, independent of execution order.
- Scatter: writing a local result back into packed message storage.
- Root/adjoint: the inference fixed point and the derivative calculation through it.
- Phase: a distinct initialization, propagation, soft, hard or outer-solver stage.
- N/A: not applicable.

## Notes

**2026-09-25T04:56:47Z**

Pre-implementation handoff: Runtime numerical or status invariant; affected host/JIT, flooding/batch/registerized/flat scheduling, exact propagation/certification, single/vmap and derivative consumers. Checkpoint N/A. schedule_program.py was already 808 lines, so first move its unchanged scan executors to execution/scans.py, retaining existing private import aliases. Focused extraction parity: 35 passed in 17.64s across schedule_program, Gaussian schedule and discrete kernel tests. This commit is a pure ownership refactor; diagnostic transport follows separately.

**2026-09-25T05:06:09Z**

Research question: do provenance checks match every local output and scatter? Answer: constraints omit a member axis; EF storage casts can overflow finite f64 updates in f32 arenas. Fixed singleton handling and destination-dtype checks for categorical/count, Gaussian and EF blocks; mechanism regressions cover both. Sources: execution/contracts.py, compiled.ConstraintPlan, execution/registered.py, execution/{synchronous,scans}.py and schedule_program.py. Confidence: high, read-only reviewer plus focused tests. No status or message-update policy changed.

**2026-09-25T05:07:54Z**

Implementation: immutable model-local factor ranks plus typed outgoing-block checks (including storage casts), synchronous rank reduction, chronological batch/registerized/flat scan transport, exact propagation/certification phases, and numeric-only derivative consumers. Focused validation: 69 schedule/discrete/Gaussian/constraint/shared-goal tests passed; 133 iteration/strategy/gradient/Colony/schedule/execution tests passed (2 slow deselected); final dedicated failure transport mechanisms 20 passed in 10.23s. Reviewer gaps addressed with a dependent lower-rank downstream NaN case and actual hybrid nonfinite-output case. Public exposure remains cff-lhbm; no inference policy changes. Full short-suite gate pending.

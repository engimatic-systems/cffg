---
id: cff-lhbm
status: in_progress
deps: [cff-fbps]
links: []
created: 2026-09-25T04:22:00Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-04mb
tags: [pcgym, diagnostics]
---
# Expose originating failure diagnostics in Engine and Colony results

## Background in Plain English

Once execution retains provenance, callers need a small stable public record on RunResult. Aggregate statuses remain authoritative; the diagnostic explains a failure without inventing factor ownership when only a global check failed.

This is a prerequisite split from cff-04mb before implementation so each change remains reviewable. Evidence: current execution owners and the PCGym audit; source details are in the parent ticket's research note.

## Outcomes / Acceptance Criteria

1. RunResult exposes an immutable optional failure record with stable reason, factor name or explicit unknown, phase and logical iteration.

2. Ordinary iteration, priming and exact propagation/certification decode the retained record once at finalization rather than reading every factor on every host sweep.

3. Engine and singleton/batched Colony agree per instance; failed or stopped members cannot overwrite peers' records.

4. Initial corrupt state, edge-only support failure and scalar-ledger failure have explicit generic reasons and factor=None when no origin is established.

5. Successful results have no failure record, and all pre-existing statuses, states, trace meanings and numerical checks remain unchanged.

6. README and architecture define simultaneous-failure choice and iteration/phase numbering; tests cover sparse/vacuous input, float32, downstream NaNs, exact/mixed paths and a larger healthy/failed batch.

7. Outer results have an honest generic terminal diagnostic until cff-d57a adds retained local origin; rejected recoverable trials are not labeled run failures.

## Methodological Suggestions

Update runtime_types, iteration outcomes/progress, exact strategy, run_results and Colony. Update result-comparison helpers to compare structured optional fields. Architecture: Runtime numerical or status invariant. Host/JIT, flooding/scheduled and single/vmap required. Current/legacy checkpoint schemas stay unchanged, but compatibility checks must distinguish restored persisted fields from omitted run-local diagnostics; cached restored failures legitimately have no diagnostic.

The implementer may choose another sound method. Keep the four existing run statuses and certificate thresholds unchanged. Follow focused tests with one passing final short suite; continue the PR chain from cff-v6th-exact-colony (PR #163), updating the intended preceding base before branching.

## Future work

- cff-d57a consumes this contract; cff-04mb closes only after all four prerequisite stages are complete.

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

**2026-09-25T05:37:46Z**

Pre-implementation handoff: Runtime numerical or status invariant. Affected pairs: host ordinary/finalization and JIT-produced facts; flood/mixed/exact; standalone/singleton/batched Colony. Adaptive outer origin is deferred to cff-d57a, with honest generic diagnostics here. Research corrected the checkpoint N/A assumption: functional RollingAgent persists an explicit whitelist of RunResult fields. Keep that schema unchanged; run-local failure diagnostics are absent after restore, and test cached failed-result parity on persisted fields explicitly. Current/legacy readers require compatibility tests, not a schema migration. Public record will be frozen and optional; successful results always have None. Device failure facts latch first origin plus iteration, then decode factor names once at finalization.

**2026-09-25T05:43:17Z**

Research question: does optional RunResult.failure affect persistence? Answer: only three result finalizers and the explicit functional-agent loader construct production results. The loader intentionally defaults failure=None and its metadata whitelist is unchanged. Fresh Engine/Colony and agent parity helpers now compare the immutable diagnostic; current/legacy restored-result helpers explicitly compare persisted fields and require omitted provenance. Sources: _agent_mu3.py result metadata/loader, _agent_checkpoint.py functional restore, tests/test_agent_mu3.py weight-collapse replay and tests/test_constraint_reconstruction.py prepared cache migration. Confidence: high; direct source review plus regression checks. This corrects the initial checkpoint N/A assumption without broadening persisted schemas.

**2026-09-25T05:44:05Z**

Implementation now exposes frozen FailureDiagnostic(reason, factor, phase, iteration) through RunResult.failure and the cffg facade. First local facts and logical iteration remain on device until finalization; Engine and Colony use the same decoder and cached serialized-name table. Global failures use explicit unknown ownership. Outer controllers provide generic terminal explanations pending cff-d57a, and initial rejections are identified separately. Documentation covers successful/restored None, checkpoint omission, ordering and numbering. Focused checks so far: 62 Engine/Colony parity, 59 strategy/public-API tests, 57 runtime/lifecycle/local-fact/transport tests, and final dedicated public diagnostics 16 tests all passed. Checkpoint regression batch and read-only review are still pending.

**2026-09-25T05:45:03Z**

Read-only review found zero-budget delta-constrained Colony mislabeled its generic diagnostic phase as hard, unlike Engine. Gated hard phase on a positive budget and added two-member full-result parity. Current/legacy functional checkpoints and exact-Colony batch: 56 passed, 23 slow deselected in 140.59s. README facade checks: 5 passed. The checkpoint schema and all status/numerical decisions remain unchanged.

**2026-09-25T05:46:05Z**

Review complete: preserved functional budget provenance in a separate device record even when the existing finalizer reclassifies status as undefined-ledger divergence; added that mechanism and direct generic outer fallback tests. Final focused batch after the last implementation edit: 63 passed in 12.33s (public diagnostics, iteration, run strategies, README). Dedicated public file now has 20 cases. Prior checkpoint/exact batch passed 56 with 23 slow deselected; runtime/lifecycle/transport batch 57 passed. Review found no remaining numerical/status/FE/gradient policy change. Full short-suite validation pending.

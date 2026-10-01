---
id: cff-lhbm
status: open
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

Update runtime_types, iteration outcomes/progress, exact strategy, run_results and Colony. Update result-comparison helpers to compare structured optional fields. Architecture: Runtime numerical or status invariant. Host/JIT, flooding/scheduled and single/vmap required; current/legacy checkpoint N/A because diagnostics are not persisted.

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

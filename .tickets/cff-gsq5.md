---
id: cff-gsq5
status: closed
deps: []
links: []
created: 2026-09-25T04:22:00Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-04mb
tags: [pcgym, diagnostics]
---
# Define local failure facts and retain nonlinear failure reasons

## Background in Plain English

Local nonlinear updates already compute a machine-readable reason, but ExecutionDelta drops it before schedule execution. A small shared numeric contract is needed before changing every schedule/result interface.

This is a prerequisite split from cff-04mb before implementation so each change remains reviewable. Evidence: current execution owners and the PCGym audit; source details are in the parent ticket's research note.

## Outcomes / Acceptance Criteria

1. ExecutionDelta can carry optional per-member structured failure reasons without changing outgoing messages, residuals or existing local status codes.

2. A fixed-shape integer failure record has an explicit empty sentinel and deterministic first-in-time and lowest-factor-rank reduction helpers usable under JIT and vmap.

3. Structured local reasons are eligible only for failed local statuses; a vacuous nonlinear input with status zero is not reported as failure.

4. Non-finite detection distinguishes valid categorical negative infinity from NaN/positive infinity, while Gaussian non-finite outgoing values remain invalid.

5. The configured nonlinear binding and registered Gaussian adapter retain reasons; ordinary uninstrumented bindings keep their existing behavior.

6. Focused tests cover dtype-independent reason codes, sparse support, vacuous/improper input, simultaneous failures and a batch at least ten times a small fixture; no public RunResult behavior changes yet.

## Methodological Suggestions

Use a dedicated small diagnostics owner rather than further enlarging engine/schedule modules. Local facts are auxiliary and must not affect derivatives or decisions. Architecture: Runtime numerical or status invariant, with New factor or rule family local-update transport reviewed. Host/JIT and single/vmap required; flooding/scheduled transport deferred to cff-fbps; current/legacy checkpoint N/A.

The implementer may choose another sound method. Keep the four existing run statuses and certificate thresholds unchanged. Follow focused tests with one passing final short suite; continue the PR chain from cff-v6th-exact-colony (PR #163), updating the intended preceding base before branching.

## Future work

- cff-fbps consumes this contract; cff-04mb closes only after all four prerequisite stages are complete.

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

**2026-09-25T04:27:44Z**

Implemented the bounded local-facts stage: ExecutionDelta gains optional per-member failure_reason; configured nonlinear kernels map only true failure codes to the shared reason vocabulary, and execute_gaussian preserves them. New failure_facts.py defines fixed-shape int32 facts, chronological merge, phase tagging, stable-factor selection and domain-aware nonfinite checks. It does not change numerical status, updates, objective or public RunResult. Current schedule consumers still ignore the auxiliary field. Focused tests: uv run pytest -q tests/test_failure_facts.py tests/test_execution.py tests/test_nonlinear_engine.py -> 43 passed, 1 deselected in 86.54s. Independent read-only review found no actionable issue; float32/64, both approximation methods, sparse/vacuous input and 30-40 member JIT/vmap checks are covered. git diff --check passes. Final short-suite gate pending.

**2026-09-25T04:54:10Z**

Completed final validation at e344b11: uv run python tools/dashboard.py --run-suite -> 2,441 passed, 69 deselected, no failures. Measured wall time 1,193.265s (pytest 1,179.22s), within the 1,200s short-suite limit. No code or selection edits followed this run. Runtime history and dashboard refreshed. Focused checks and independent review were already green. Next child cff-fbps carries these local facts through schedules; public RunResult attribution is not claimed by this stage.

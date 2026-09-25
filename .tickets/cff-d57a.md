---
id: cff-d57a
status: in_progress
deps: [cff-lhbm]
links: []
created: 2026-09-25T04:22:00Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-04mb
tags: [pcgym, diagnostics]
---
# Preserve terminal failure provenance through adaptive outer solves

## Background in Plain English

Adaptive/GFE solvers can reject a bad trial and still converge. A global latch around every map evaluation would misreport those recoverable trials and may also run during differentiation. The diagnostic must describe failure on the returned trajectory or terminal evaluation.

This is a prerequisite split from cff-04mb before implementation so each change remains reviewable. Evidence: current execution owners and the PCGym audit; source details are in the parent ticket's research note.

## Outcomes / Acceptance Criteria

1. Initialization, reconstruction and outer-map outputs carry auxiliary provenance without changing root, objective, derivative or solver decisions.

2. The documented policy retains the originating cause for a terminal failed solve; rejected trial/shadow failures do not contaminate a later successful result.

3. Outer iteration and phase names come from controller state, not map-evaluation count or trace length.

4. A nonlinear improper-cavity mechanism test exposes the responsible factor and stable local reason even when downstream messages become NaN.

5. A recovering rejected-trial regression returns success without a terminal failure record; exhaustion with no numerical failure is distinguished from divergence.

6. Host/JIT, flooding/scheduled reconstruction and supported single/vmap paths preserve prior arrays/statuses/gradients in float32 and at larger scale; generic unknown ownership remains explicit.

## Methodological Suggestions

Inspect outer_strategy.py, outer.py, fixed_point.py and nonlinear failure facts. Architecture: Runtime numerical or status invariant and derivative consumers in Learnable or checkpoint feature. Current/legacy checkpoint N/A. Do not use Python side effects inside traced numerical functions or recompute the final state to invent the first cause.

The implementer may choose another sound method. Keep the four existing run statuses and certificate thresholds unchanged. Follow focused tests with one passing final short suite; continue the PR chain from cff-v6th-exact-colony (PR #163), updating the intended preceding base before branching.

## Future work

- cff-04mb consumes this contract; cff-04mb closes only after all four prerequisite stages are complete.
- cff-7kpx consumes this contract; cff-04mb closes only after all four prerequisite stages are complete.
- cff-ectw consumes this contract; cff-04mb closes only after all four prerequisite stages are complete.

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

**2026-09-25T06:15:40Z**

Pre-implementation handoff: Runtime numerical/status invariant plus derivative consumers. Affected host/JIT, flood/mixed reconstruction, exact initialization within outer control, Engine/per-instance Colony; outer-controller vmap is unsupported (generic map/kernel batching remains relevant). Current/legacy checkpoint schemas remain unchanged and omission compatibility is already covered by cff-lhbm. outer.py is 1401 lines, so first extract unchanged chart geometry and compiled trust kernels into outer_chart.py and outer_trust.py, retaining existing outer import aliases; then carry auxiliary provenance without altering numerical decisions. Rejected trial/shadow facts must never be latched into a recovered successful result.

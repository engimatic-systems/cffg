---
id: cdh-urn0
status: open
deps: []
links: [cdc-ztuf]
created: 2026-09-14T22:56:19Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-zov5
tags: [hybrid, numerical, inference]
---
# Diagnose native weighted covariance projection discrepancy before moderate qualification

## Background in Plain English — WHY

The cdc-ztuf report repairs exposed a native-float32 projection discrepancy in unchanged hybrid_natural.marginal code. A single T2-shaped local pair belief with K=4 and d=2, nonuniform masses proportional to 1..16, means spaced by 1/16, covariance 100000*I and retained low words eps*high/4 produces source-label-0 off-diagonal covariance 0.05000000074505806. Replaying the prescribed mixture operations from the saved belief gives about 0.0625. The difference exceeds the frozen local allowance 512*4*eps32 (absolute and relative). A uniform rare source at mass scale 1e-12 also produced 0.076171875 versus 0.078125. Both native status flags were zero. This is local arithmetic evidence, not a moderate solve or qualification.

The repaired report validator rejects this discrepancy; weakening it would conceal an unresolved numerical problem. The implementation author must finish ordinary report repairs and pause, so diagnosis and any core numerical repair belong to this separate bounded ticket. No underlying numerical code was changed during discovery.

## Outcomes / Acceptance Criteria — WHAT

1. Identify whether compiler transformation, native local arithmetic, or the independent replay causes the covariance discrepancy, with a bounded reproducer and operation-level evidence.
2. Preserve the frozen D12 mathematical definition, approximation, dtype policy and allowances; do not make a discrepant result pass by increasing tolerances or choosing only favorable masses.
3. Make the preserved native projection regression pass normally, including nonuniform and rare-source controls; remove its explicit expected-failure marker only with supported repair evidence.
4. Preserve both retained covariance words for residuals and the once-rounded physical covariance consumed by learning. No universal exactness claim is required.
5. Before implementation, record the architecture change path and affected host/JIT/single/vmap/derivative surfaces. Sparse support, vacuous inputs, float32 and 10–100x domain size need explicit planning answers. Bound any larger audit prospectively in a dependent ticket.

## Methodological Suggestions — HOW

Optionally start with tools/d12_moderate_checks.py --projection-probe and the strict expected-failure regression in tests/test_d12_moderate_integrity.py. Compare the native marginal against saved-input replay and exposed intermediate local operations. During discovery, a separate compilation of mixture intermediates produced approximately 0.0625 while the marginal returned 0.05 from the same saved region; this suggests execution-context sensitivity but does not establish its cause. /tmp/cdc-ztuf-local-debug.py and its output contain supplementary diagnostics; the committed probe is the durable reproducer.

## Future work

This ticket is an explicit prerequisite of cff-a3wz moderate qualification. cdc-ztuf owns report validation and setup review; it does not own a new solver-arithmetic change. Root must assess this blocker before authorizing subsequent immutable source requalification. No moderate solve, storage compilation, recovery fit, historical-evidence rewrite, or extra timing solve is authorized by this ticket.

## Definitions

D12 is the complete hybrid message/auxiliary solve. T2 describes two time points; K is label count and d is continuous dimension. A covariance low word retains the rounding remainder alongside its primary float32 word. eps32 is float32 machine epsilon. A source-conditioned marginal is the projected Gaussian mixture for one source label. JIT means just-in-time compilation; vmap is JAX batching. An expected-failure regression keeps a known defect visible without claiming it passed.



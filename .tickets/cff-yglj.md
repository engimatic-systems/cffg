---
id: cff-yglj
status: open
deps: [cff-sf29]
links: [cff-sf29, cff-wbhu]
created: 2026-09-10T00:07:26Z
type: bug
priority: 2
assignee: coopwilliams
tags: [engine, colony, status]
---
# Align Colony initial-failure results with standalone Engine

## Background in Plain English — WHY

A graph with a non-finite initial state must report a consistent failure whether it runs alone or beside another graph. At engine baseline 76251b44fc467a976aa056556f8fb371fe5de7f4, Engine.run reports inner divergence with an infinite inner residual. Colony's actual batched path reports inner convergence with a zero inner residual for the same input, even though both public statuses are diverged at zero iterations.

This is a result-reporting defect, not evidence that the poisoned graph converged. Fix it before cff-ii9t unifies the two result policies so the extraction does not preserve conflicting contracts.

## Outcomes / Acceptance Criteria — WHAT

1. The same initially non-finite ordinary state produces matching public status, inner status, inner residual and zero logical iterations in standalone Engine and a real multi-instance Colony batch.
2. The failed instance retains its input state without priming or updating it.
3. A healthy same-structure peer still produces the same state, traces and result fields as its standalone run.
4. The regression exercises both zero and positive iteration budgets and both float32 and float64.
5. Exact-zero infeasibility remains distinct from non-finite divergence. Existing vacuous/improper-state tests retain their classifications.
6. The test uses at least two same-structure instances and demonstrates batch execution for the healthy peer when a positive budget permits an update.
7. The Runtime numerical or status invariant change-path row covers standalone/batched and host/JIT classification. Flooding/scheduled priming is checked where applicable; model serialization and stochastic checkpoint decoding are N/A.
8. Focused runtime/Colony/characterization tests and one final dashboard-backed short suite pass after the last code change.
9. The repair is a separate behavior-change commit, not part of the cff-sf29 characterization or a new batching optimization.

## Methodological Suggestions — HOW

These suggestions are optional; another sound method is acceptable.

- Reuse the ordinary VMP multi-instance fixture from tests/test_engine_contracts.py.
- Classify original states before priming. Freeze invalid members at that boundary and use Engine's existing initial-failure fields as the authority.
- Hostile-input review: NaN and infinity are the triggering inputs; literal log-zeros are not NaNs and must retain support semantics. Vacuous/improper continuous states keep existing policy. Test both dtypes. A tenfold model size is N/A to this host-side per-instance result initialization; batching isolation, not a new numerical rule, is changed.

## Future work

cff-ii9t removes duplicated ordinary termination and result policy. cff-wbhu compiles the ordinary convergence loop. cff-rqqc owns stochastic Colony key/state batching. This repair implements none of those larger changes.

## Definitions

- **Colony:** The service that groups graph instances by structure and executes compatible numerical updates together.
- **Inner status:** The public result field describing the numerical update solver, distinct from the overall run status.
- **Priming:** A schedule's initial propagation pass before its repeated updates.
- **Logical iteration:** An update counted in the public result; a rejected initial state has zero such updates.
- **VMP:** Variational message passing.
- **JIT:** Just-in-time compilation of array computations.



## Notes

**2026-09-10T00:09:36Z**

Confirmed at unchanged production source 707c31b, identical to engine baseline 76251b44fc467a976aa056556f8fb371fe5de7f4. A read-only probe imports spawn_cases from tests/test_engine_contracts.py, sets instance zero's messages[0] to NaN with _replace, and compares cached_engine.run(state, 5, tol=1e-12, harden_last=5) with colony.run(5, tol=1e-12). Standalone: status=diverged, iterations=0, inner_status=diverged, inner_residual=inf. Actual batch: status=diverged, iterations=0, inner_status=converged, inner_residual=0.0. Both returned messages remain non-finite. Colony also invokes its prime function before initial-state classification; state preservation across that boundary requires an explicit regression rather than assuming the prime is inert. Sources: engine.py:1022-1048; colony.py:129-146,159-184,260-271. Confidence: high, direct executable reproduction; this is not a report of false overall convergence.

cff-sf29 supplies the fixture. Fix this before cff-ii9t's shared controller/result-policy extraction, in a separate behavior-change PR. cff-wbhu already requires truthful initial-failure classification as part of its broader compiled-loop work; this ticket isolates the present result-field defect without waiting for that optimization.

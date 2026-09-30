---
id: cff-pl0v
status: closed
deps: []
links: []
created: 2026-09-30T18:57:13Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-toqa
tags: [pcgym, numerics]
---
# Return the state actually certified by the outer solver

## Background in Plain English

An inference result must return the state whose convergence it proved. The outer solver checks the current state against its next update, but its three successful paths return that next update instead. A nonlinear map can amplify a small residual, so the returned state need not satisfy the reported certificate.

The new short reactor integration check exposed this in float32 sigma-point inference. With a frozen solver chart, the reported residual was 1.9500526e-6, but one update from the returned state had residual 6.6173605e-5. The effective tolerance was 64*eps=7.6293945e-6. Rebuilding the chart gave nearly the same mismatch, but the frozen-chart check is the authoritative comparison. The physical cavity point clouds remained finite and agreed with independent integration.

## Outcomes / Acceptance Criteria

1. Each successful candidate, main trust-solver and auxiliary-sequence path returns the point whose full residual and finite free-energy change were checked, in the same frozen chart.
2. Independent deterministic maps reproduce the mechanism: a small update at the certified point followed by an update above tolerance at its successor. Cover float32 and float64 without relying only on reactor goldens.
3. Successful results retain their budget accounting, inner status, failure facts and free-energy meaning. The returned state and final recorded energy refer to the same certified point after declared eliminated-slot reconstruction.
4. Failed and budget-exhausted paths remain explicit. Sparse support, vacuous inputs and improper-reference rejection are unchanged and covered by their existing tests; this fix does not introduce an approximation point, ridge or extra uncounted update.
5. Host control uses the same success handoff for compiled and uncompiled map callbacks, flooding and mixed reconstruction. Single/vmap map arithmetic remains unchanged. Current/legacy decoding is N/A because no persistent schema changes.
6. The float32 nonlinear reactor reproducer satisfies an actual next-map certificate at the returned point in the solver chart after the repair. Keep this expensive regression opt-in; a final default suite passes once after relevant changes.

## Methodological Suggestions

Another sound method is acceptable. Start with the Runtime numerical or status invariant architecture row, outer.py success returns and outer_strategy.py reconstruction. Returning the already-certified point avoids an extra evaluation or a hidden convergence check. Use an independently controlled scalar map to distinguish certificate handoff from chart metric dependence, plus the retained reactor integration inputs from cff-qw7x. Update ARCHITECTURE and LIMITATIONS in place if the public numerical contract needs clarification.

## Future work

cff-qw7x publishes the complete declared reactor planning envelope and standalone recipe after this prerequisite. cff-toqa closes the nonlinear startup integration contract. cff-wt1c performs matched closed-loop comparisons; no performance claim belongs to this repair.

## Definitions

A residual is the difference between a point and its update. A frozen chart fixes the coordinates and whitening used to measure that difference throughout a solve. Whitening scales Gaussian information using a proper reference covariance. A successor is the result of one update. An auxiliary sequence proposes additional candidate points. Eliminated slots are message values reconstructed from the active coordinates rather than independently solved. Eps is machine precision for the numeric dtype. A convergence certificate is a tested residual and objective-change bound, not a posterior-accuracy guarantee.

## Notes

**2026-09-30T19:02:00Z**

Pre-implementation change-path handoff: Runtime numerical or status invariant. Change only the3 successful host outer-controller handoffs; registered Gaussian/discrete/EF equations, JIT/vmap map arithmetic, startup, sparse/vacuous classification and trust steps unchanged. Flood and mixed/eliminated reconstruction consumers apply; current/legacy checkpoint serialization N/A. Independent scalar expanding map covers candidate/trust/auxiliary returns in f32/f64 with host/JIT callbacks, final energy and counters. Fixed numerical metric must be captured; rebuilding whitening is a different assertion. New reactor source work remains separately preserved for cff-qw7x.

**2026-09-30T19:05:42Z**

Dependency refinement, built from tk dep tree --full cff-toqa, tk ready and tk blocked; tk dep cycle clean. Legend prerequisite --> dependent. [x]7kpx --> [x]fc2y --> [x]36x7 --> [in_progress,ready]pl0v --> [in_progress,blocked]qw7x --> [open,blocked]toqa --> [open,blocked]wt1c --> [open]xa6z. Existing readiness/startup roots are complete; fixed-bank, bounds design and other independent ready roots can proceed without this handoff. The tk records are authoritative; pl0v now directly gates qw7x and toqa. Retained recipe work is in the task-named git stash until its prerequisite lands.

**2026-09-30T19:06:11Z**

Focused validation after the final runtime edit: uv run pytest -q tests/test_outer_failures.py tests/test_run_strategies.py tests/test_nonlinear_engine.py tests/test_adaptive_startup.py tests/test_runtime_status.py —97 passed,13 deselected in94.57s. Relevant opt-in uv run pytest -q -m slow tests/test_outer_failures.py -k float32_reactor —1 passed,30 deselected in28.21s; actual returned-point residual and FE certificate pass in the frozen chart, without changing tau. All12 independent scalar combinations (3 success paths x2dtypes xhost/JIT) verify residual/energy identity, counters and a successor above tolerance. Full gate pending; no claim yet that reactor envelope is complete.

**2026-09-30T19:06:51Z**

Independent bounded review question—does returning the certified point preserve reconstruction/counters/energy across all3 paths? Answer—yes: outer strategy reconstructs solved.arrays and certificate FE uses that same reconstruction; evaluation counts still include the successor used for certification and no extra accepted update is implied. Source outer.py and outer_strategy.py; confidence high. Remaining targeted coverage suggestion is a successful mixed-schedule return with eliminated slots; no defect identified. Scalar/JIT and real-flood regressions remain independent authorities.

**2026-09-30T19:07:53Z**

Review follow-up addresses the mixed/eliminated-slot gap in the existing curved-sensor oracle tests: both Taylor/sigma auto cases assert a real Gaussian elimination set, returned state unchanged by exact reconstruction, no reconstruction failure and final energy equality. No new expensive default replay; strengthen existing tests in place. Runtime equations and counters unchanged.

**2026-09-30T19:09:15Z**

Mixed/eliminated follow-up validation: uv run pytest -q tests/test_nonlinear_engine.py -k curved_sensor —2 passed,18 deselected in11.17s after final test edits. Full gate will run once at proposed PR head; latest measured1182.688s, prescribed wait1590s. No repeat is planned for later ticket/runtime-history bookkeeping or marker-only budget adjustments.

**2026-09-30T19:37:01Z**

Final gate at1695f20: uv run python tools/dashboard.py --run-suite —2631 passed,157 deselected,4 existing Gym deprecation warnings,pytest1168.19s,wall1182.696s (within1200s budget). Dashboard refreshed at /Users/ravd/agent-share/cffg-tests.html. Complete: all3 success points and final solve-energy handoffs corrected; independent host/JIT/dtype maps, real f32 reactor and mixed reconstruction verified. No new message/energy equation or schema. Close with validation commit; next cff-qw7x restores retained recipe integration and assesses the remaining declared grid. Full gate reused for bookkeeping; no second unchanged run.

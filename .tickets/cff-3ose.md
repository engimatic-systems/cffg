---
id: cff-3ose
status: closed
deps: [cff-5xdy]
links: [cff-yglj]
created: 2026-09-10T05:14:03Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-ws3c
---
# Align late-failure diagnostics between Engine and Colony

## Background in Plain English — WHY

Preparing the shared iterative-controller extraction (cff-ii9t) exposed a remaining Engine/Colony disagreement at source cf5cdb6. If a batched instance fails after earlier finite free-energy measurements, Colony changes its status but can retain the previous finite residual. Engine reports an infinite residual for the same failure. The overall failure status is already correct; this is not evidence that a failed run is reported as converged.

A bounded controlled-step probe reproduced the mismatch in float32 and float64: after FE values 1 and 2, an inner-solver failure, non-finite FE, non-finite State, or empty support on iteration 3 yielded residual=1 in Colony and infinity in Engine. A healthy peer kept running. The upcoming shared result owner needs one repaired contract rather than preserving contradictory public facts.

Comparing every result field also exposed a related trace defect: Engine can record a zero message delta when the changed message block contains NaN. Its Python maximum starts with a finite functional-update delta, and Python comparison can silently discard a later NaN. Colony starts with the first nonempty message block and can instead report NaN; it can also discard NaN from a later block. Both runs still report divergence because the State finiteness check catches the failure. The repair makes these diagnostics truthful before the shared controller moves them.

## Outcomes / Acceptance Criteria — WHAT

1. For ordinary batched Colony runs, every terminal divergence or infeasibility reports residual=infinity, matching Engine. Cover later numerical/inner failure, empty support, non-finite FE, non-finite decoded max-product objective, and the final undefined-continuous-ledger classification.
2. A non-finite change in any message representation produces a non-finite message-delta trace entry in both Engine and Colony, never a smaller finite value because of Python maximum ordering. Preserve finite delta values, existing overall/inner statuses, inner residual, final State, iteration count and FE trace contents. Budget exhaustion with a defined ledger retains its existing finite residual; this ticket changes no convergence threshold or numerical update.
3. A rejected instance freezes after its failure while healthy peers continue with independent results. Compare all public RunResult fields to the corresponding standalone Engine result, using a real batch rather than the singleton fallback.
4. The applicable change-path row is Runtime numerical or status invariant. Engine is the existing reference; Colony is the repaired duplicate. Host/JIT and single-instance/vmap are checked pairs. Flooding and available primed schedules retain ordering. New IR fields, factor formulas, learning cadence and current/legacy checkpoint decoding are N/A.
5. Hostile inputs: exact zero support becomes infeasible; non-finite state/FE or an undefined continuous ledger becomes diverged; vacuous continuous state is not prematurely rejected before the existing propagation opportunity. Exercise float32 and float64, NaN changes in the first and a later nonempty representation, a late failure after a finite residual, and a larger batch (at least 20 instances) with independent stopping. No new numeric floor or solver is needed.
6. Keep the fix as a behavior-repair commit before cff-ii9t. Existing runtime/Colony contract tests and one final default-suite gate pass. Update ARCHITECTURE's Colony failure contract.

## Methodological Suggestions — HOW

Inject controlled fixed-shape step/energy facts in focused tests to reproduce each classification branch without inventing a failing factor formula. Use the existing Engine contract/result comparators plus an explicit non-finite-delta assertion, since baseline parity alone would preserve the maximum-ordering defect. Keep residual repair and non-finite-delta repair in separate atomic commits if useful; the shared policy/result extraction remains cff-ii9t.

## Future work

cff-ii9t removes the duplicated termination policy after this repaired baseline. cff-wbhu owns compiling the convergence loop; stochastic Colony PRNG and mu3 support remain cff-rqqc.

## Definitions

- Residual: the reported numerical distance from a stopping condition. Infinity records that a failed run has no usable convergence certificate.
- Late failure: a failure after earlier successful updates, potentially after a finite residual was already measured.
- Undefined ledger: the existing continuous/EF run path never produced a finite free-energy entry before termination.


## Notes

**2026-09-10T05:19:13Z**

Pre-implementation evidence at cf5cdb6: /tmp/cffg-colony-late-failure.py checks four injected failure paths in both precisions and a healthy peer. All eight cases differ in residual (Colony 1, Engine infinity); the NaN-message cases also differ in the final delta (Colony NaN, Engine 0). Every other non-State result field agrees. This is a controller-fact probe, not a claim about a naturally failing factor rule. Confidence: high; source owners are Engine.run phase and Colony.run's batched classification. The source also has residual omissions in max-objective failure and final undefined-ledger classification; these are included in acceptance before implementation.

**2026-09-10T05:20:10Z**

The second controlled probe /tmp/cffg-colony-terminal-failure.py confirms both remaining residual branches: non-finite max objective reports Colony residual 0 versus Engine infinity, and final undefined continuous ledger after an inner budget stop reports Colony 0.125 versus Engine infinity. Overall divergence agrees. These are now reproduced, not just inferred from source.

**2026-09-10T05:41:56Z**

Residual repair checkpoint: a new f32 primed-schedule regression failed on the old Colony residual=1, as intended. Setting the final public residual to infinity for divergence/infeasibility repairs all terminal branches without changing their statuses or inner facts. The complete Colony-failure and engine-contract selection then passed 50 tests in 9.67 seconds. The 20 new cases cover later inner failure/budget, FE failure, empty support, max-objective failure and undefined continuous ledger, both precisions and available primed/flood schedules. Healthy neighbors and every other result field match standalone execution. Non-finite delta aggregation is the next separate commit; final default gate follows the final code/test change.

**2026-09-10T05:44:45Z**

Non-finite delta repair: the new f32 Gaussian precision-block case failed on the old finite delta=1 even though that later block contained NaN. Engine and Colony now use NumPy maximum reductions that propagate NaN from any representation, preserving all finite delta values and the existing divergence status. Added first/later representation cases and genuine 20-instance batches in both precisions. The complete focused failure/engine-contract/runtime-status/Colony selection passed 72 tests, 3 slow deselected, in 14.65 seconds. There are 32 new cases in the existing dashboard-enrolled failure file (314 lines). Small-file lint/format checks and diff check pass; Engine retains 19 and Colony five pre-existing Ruff findings, with none introduced. Source changes are complete; the single final default gate is next. No numerical factor update, objective, schema, tolerance or public status changed.

**2026-09-10T06:05:58Z**

Completed in separate source commits: 8bb584a repairs stale Colony failure residuals; a2c55e6 preserves non-finite delta traces. The final focused selection passed 72 tests with three slow deselections. The single final default gate passed 2,043 tests, zero failures, 41 slow deselections and four existing Gym warnings. Pytest took 1,132.26 seconds; measured wall time was 1,145.615 seconds. It completed 2026-09-10T06:03:51.571355+00:00 at ticket-only head 25f537c. Code/test source a2c55e6 is authoritative. Runtime history and the shared dashboard are refreshed. Only one suite-output check was made after the planned first deadline.

Every reproduced failure still has its original overall status; residuals and non-finite deltas now report the failure consistently. Healthy batch peers, finite budget results, inner facts, traces apart from previously masked NaN deltas, and numerical states retain their reference behavior. No factor formula, solver, objective, schema or tolerance changed. Publication reuses this gate. Next is cff-ii9t, whose handoff records the ordering and trace-precision constraints. Independent agent review remains unavailable and is not claimed.

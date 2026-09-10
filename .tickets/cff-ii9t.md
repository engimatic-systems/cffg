---
id: cff-ii9t
status: closed
deps: [cff-5xdy, cff-yglj, cff-3ose]
links: [cff-jrbw, cff-sf29, cff-jby3]
created: 2026-09-02T04:17:42Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-ws3c
tags: [engine, architecture, runtime, refactor]
---
# Extract iterative run control beneath Engine.run

## Background in Plain English — WHY

At authoring snapshot e488df3, `src/cffg/engine.py:916-1190` and lines
1657-1752 are scope anchors for ordinary iteration plus finite-state and zero-
support checks; they are not promised future line maps. The implementation
starts from the exact `cff-sf29` engine baseline. This responsibility selects
propagation, tracks free energy and deltas, applies damping/hardening, and
distinguishes convergence, exhaustion, divergence, and infeasibility.

This ticket moves only ordinary iterative control behind a typed outcome.
Specialized deterministic outer trust-region and exact-run strategies remain in
`cff-jrbw`. Engine and Colony share one termination policy and result finalizer;
each public entry point delegates its result construction to that owner.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-5xdy` is closed with lifecycle services preserving repaired state.
2. One iterative controller accepts prepared `State`, `ScheduleProgram`,
   objective evaluator, and explicit tolerance/budget/damping controls and
   returns an immutable `IterationOutcome`.
3. `IterationOutcome` contains final internal state, iteration count, delta/FE
   trace facts, finite/support signals, and one typed termination reason. It is
   not a public `RunResult` and does not choose a public status enum.
4. One lower-level result owner interprets termination facts, validates the
   outcome, maps it to public `RunStatus`, and constructs `RunResult`.
   Engine.run and Colony.run both delegate to it. Their ordinary iterations
   share incremental termination assessment instead of retaining separate
   finite/support/FE/status policies. Public results and enum identity remain
   compatible; runtime value definitions live below the facade and are
   re-exported through the existing imports.
5. Non-finite delta/FE is divergence; exhausted budget is not convergence;
   literal zero support is infeasible; existing convergence tolerances and
   logical iteration meaning remain unchanged.
6. The affected row is **Runtime numerical or status invariant**. **New factor
   or rule family** applies only through the shared schedule/objective inputs;
   **New IR field** and **Learnable or checkpoint feature** are N/A because no
   schema or restored-state decoding changes.
7. Flooding/scheduled controllers consume the same executor/schedule owners.
   The cff-sf29 multi-instance/multi-structure Colony fixture and host/JIT
   termination facts retain their recorded contract. Restored states enter
   through the same prepared-state boundary. Colony keeps batching and active
   instance freezing; it is not replaced with a sequential Engine loop.
   7.1. A bounded batch case includes one initially invalid or infeasible
        instance beside a healthy instance, and instances that stop at
        different iterations. Results retain independent status and state.
   7.2. Numerical step/termination facts remain fixed-shape arrays usable by
        JAX transformations. Converting traces to public Python tuples occurs
        at the result boundary. Compiling the convergence loop is cff-wbhu.
8. Literal zero support, vacuous/improper state, non-finite update or FE,
   float32, and uncomfortable-scale cases retain their exact status reason and
   trace behavior.
9. Existing runtime-status and family oracle tests remain authoritative.
   Old/new outcome characterization protects the refactor; any newly discovered
   scientific evidence gap becomes a separate ticket unless it proves the move
   unsafe.
10. No solver/globalization rule, objective formula, factor behavior, public
    status, diagnostic field, checkpoint schema, or performance target changes.
11. The iterative-controller module is below 1,000 physical lines and the
    ticket owns ordinary-loop and support/finite classification plus the
    corresponding duplicated Colony policy. It adds no new run strategy.
12. Focused runtime-status/schedule tests and the final short suite pass.
13. `docs/ARCHITECTURE.md` names `IterationOutcome`, the controller, and the
    final public status boundary in the extraction commit.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Use typed numerical reason codes beneath one shared public-result finalizer;
  avoid parallel enum-to-status mappings in Engine and Colony.
- Pass the already-built schedule and objective callables; do not make the
  controller inspect factor types.
- Characterize traces and status facts field by field rather than adding a
  long-running benchmark.

## Future work

`cff-jrbw` extracts explicit outer trust-region and exact strategies under the
same shared result boundary. cff-wbhu compiles the ordinary convergence loop;
cff-rqqc owns stochastic Colony state/PRNG isolation.

## Definitions

- **IterationOutcome:** Internal state and termination facts returned to the
  public status authority.
- **Ordinary iterative control:** Repeated scheduled/flooding updates governed
  by existing delta and free-energy convergence rules.
- **Termination reason:** Internal converged, exhausted, diverged, infeasible,
  or unsupported fact before public status construction.
- **Result finalizer:** The shared operation that validates termination facts
  and constructs the public result for Engine or Colony.

## Notes

**2026-09-09T03:29:39Z**

PR #108 research — question: does one-instance vmap test Colony result handling? Answer: no; Colony bypasses its batched path for one instance and constructs RunResult itself for actual batches. Engine and Colony need a shared termination/result owner, with real mixed-input batches and independent stopping checks. Sources at 7885bf5: colony.py:111-118, 149-280. Confidence: high. This moves existing status semantics; cff-wbhu still owns compiling the host convergence loop.

**2026-09-10T00:09:36Z**

The cff-sf29 inventory found and directly reproduced the initial-state result discrepancy now isolated in cff-yglj. That ticket is an additional prerequisite here: it establishes one corrected initial-failure contract before this behavior-preserving extraction. Do not preserve Colony's old inner_status=converged/inner_residual=0 for an initially non-finite state merely because the pre-repair characterization baseline recorded it. The overall status was already diverged; the repair concerns consistent inner facts and retaining the rejected state. Ordinary valid-state baseline fixtures remain unchanged.

**2026-09-10T00:49:06Z**

cff-yglj is now closed. Source 1308688 repairs initial Colony result facts and freezes rejected inputs across priming; its final gate passed 1,843 tests with 41 slow deselections and four existing warnings, recorded in e30b330. tests/test_colony_failures.py contains 25 reference cases for the controller extraction. Preserve this corrected behavior, not the pre-repair discrepancy recorded at 76251b4. The original valid-state characterization remains unchanged.

**2026-09-10T05:14:04Z**

Pre-controller probe at cf5cdb6 confirmed a second duplicated-result mismatch: after FE=(1,2), later inner/FE/State/support failure yields Colony residual=1 while Engine reports infinity, in both f32 and f64. Overall failure statuses already agree, and healthy peers continue. Repair is isolated in cff-3ose, now an explicit prerequisite. Do not normalize this discrepancy inside the extraction commit.

Legend: prerequisite --> dependent
[x] cff-7kfk
  --> [~] cff-5xdy (final gate running)
        --> [ ] cff-3ose (late Colony residual repair)
              --> [ ] cff-ii9t (shared iterative policy/result owner)
                    --> [ ] cff-jrbw --> cff-gl89 --> cff-3ola --> cff-ws3c
The existing direct lifecycle prerequisite remains; tk dependency records are authoritative.

**2026-09-10T05:17:25Z**

The full late-failure result comparison also found Engine masking a NaN message delta as 0 because Python max begins with the finite mu3 delta. Colony may instead preserve that NaN, or mask one from a later representation. Both already report overall divergence. cff-3ose was refined before implementation to repair non-finite delta traces as well as stale residuals; preserve its corrected diagnostic baseline during extraction.

**2026-09-10T05:49:59Z**

Implementation handoff prepared while cff-3ose's gate runs. The intended baseline is its final published head; current repair source is a2c55e6, not the earlier discrepant snapshot. Applicable row: Runtime numerical or status invariant; New factor/rule applies only through supplied schedule/objective callbacks. New IR and learning/checkpoint decoding are N/A. Host/JIT, flooding/scheduled and standalone/real vmap remain checked pairs. Current/legacy restored arrays enter the same State boundary; decoding itself does not move.

Concrete ownership constraints:
- The ordinary controller takes State, ScheduleProgram, explicit controls and narrow objective/check/functional-update callbacks, never Engine or factor-type dispatch. A shared incremental assessment owner is used by the standalone loop and Colony's existing batched transport. The finalizer alone maps the internal typed termination reason to public RunStatus and constructs ordinary RunResult.
- Retain per-instance active masks and freezing across both priming and later updates. Do not replace Colony with sequential Engine calls. Initial invalid states are checked before priming. Support checks keep reading current State.params, including constraint thresholds/tables repaired in cff-jby3.
- Preserve evaluation order: functional update failure stops before a counted scheduled step; finite/state/inner checks precede support, then inner budget, then objective evaluation. Max objective is evaluated only when its delta threshold is met. Initial undefined continuous FE may propagate until a finite entry appears; a nonempty run that never records FE is still divergence at termination.
- FE comparison history resets at each soft/hard phase. Soft convergence starts the requested hardening phase instead of ending the whole run. Initial rejection keeps zero iterations; functional-update rejection does not invent a delta entry.
- Numerical facts must stay fixed-shape array values suitable for JAX transformation; outcome/status policy remains host-side until cff-wbhu. Trace conversion must preserve the current Python-float precision: do not turn host-double FE differences into default-f32 JAX arrays just to store the internal trace. Convert traces to public Python tuples at the finalizer.
- Engine's current private step/prime aliases are instrumented by the contract tests and Colony. If the controller consumes ScheduleProgram directly, either overlay the current aliases at call time or consistently move instrumentation to the program boundary without changing test oracles. Do not accidentally make failure tests patch a handle the controller no longer uses.
- Reuse runtime_types as the pure value leaf. Extract host finite/support checks without importing Engine back into the new owners. Exact and trust-region strategy implementations, their caching and solver diagnostics remain cff-jrbw; FE formulas remain cff-gl89.

Authorities: existing test_engine_contracts, test_runtime_status and test_colony_failures. cff-3ose adds late failures, NaN in first/later representations, finite budget residuals and real 20-instance batches; its final gate will establish the repaired baseline. This is a local source/fixture handoff, not independent agent review.

**2026-09-10T06:10:20Z**

Prerequisite cff-3ose is closed and published as PR #129. Its repaired code/test source is a2c55e6; final 2,043-test gate (41 slow deselections, four existing warnings) is recorded in 29a4c9f, measured wall 1,145.615 seconds. Use the final published agent/cff-3ose-failure-diagnostics head as this ticket's branch base. The 32 new failure cases preserve truthful non-finite delta traces and infinite failed-run residuals; do not restore the old discrepancies.

**2026-09-10T06:20:05Z**

First extraction checkpoint: runtime_checks.StateChecks now owns host exact-support checks and immutable edge-cardinality buckets; state_is_finite is a standalone check. Engine's existing private check methods remain delegates for ordinary, exact and outer callers. Current constraint values still come from State.params. The immediate alldifferent search binds its per-constraint support list explicitly; the search algorithm and evaluation order are unchanged. The existing failure/runtime-status/constraint-data selection passed 120 tests in 29.20 seconds. New check-owner lint/format and diff checks pass; architecture and the change-path row name the owner. Iterative progress/outcomes and the shared public result finalizer are next; no final default gate has been run for this ticket yet.

**2026-09-10T06:25:43Z**

Additional result-field trap identified before wiring the controller: in explicit legacy-Picard GFE runs, initial non-finite/empty-support rejection returns termination_reason=NOT_RUN, while a valid zero-budget run and later failures reach the common final result and report LEGACY_PICARD. Internal outcomes distinguish initial rejection so the shared finalizer can preserve this field. A zero iteration count alone cannot distinguish these cases.

**2026-09-10T06:54:39Z**

Ordinary controller implementation checkpoint: iteration.py owns the soft/hard controller and shared IterationProgress assessment; Engine and real Colony batches both use run_results.finalize_iteration. Colony retains vmapped transport and active-member freezing. iteration_types.py provides immutable host-double traces, a typed reason and fixed-shape numeric facts; unchecked predicates remain distinct from false predicates. Current step/prime aliases are overlaid on ScheduleProgram at entry, and per-group callbacks explicitly bind their program. Exact/trust-region implementations remain in Engine.

Evidence: the trusted pre-extraction Engine and Colony at a2c55e6 matched all public fields in 118 complete results (100 standalone configurations across ten model families and both precisions, plus 18 Colony results), with 3,818 byte-identical scalar/array comparisons. After the final per-group callback binding, the 18 Colony results were rechecked with 486 byte-identical comparisons. The existing mixed-precision mu3 scatter warning also appears in this bounded comparison; no new warning is claimed. This is baseline compatibility evidence, not an independent mathematical oracle.

Existing failure/engine-contract/runtime/Colony selection: 72 passed, 3 slow deselected, 14.71 seconds. New boundary file: 16 passed in 0.96 seconds. Final boundary/schedule/CSP/GFE/mu3 selection: 82 passed, 3 slow deselected, 58.01 seconds. New modules/tests pass Ruff and formatting; Engine findings decrease 19 to 16, Colony five to three, with no new findings. New owner sizes: iteration 282, types 98, result finalizer 77, runtime checks 147; boundary tests 202. The new test file is dashboard-enrolled. Architecture/module maps and change paths describe the actual owners. The final default-suite gate is next; no current-head default result is yet claimed.

**2026-09-10T07:16:58Z**

Completed at code/test source 20c0367, with host checks separately extracted in a746778. The single final default suite passed 2,059 tests, zero failures, 41 slow deselections and four existing Gym warnings. Pytest took 1,142.43 seconds; measured wall time was 1,155.815 seconds. It completed 2026-09-10T07:13:55.778316+00:00 at ticket-only head ef94242. Runtime history and shared dashboard are refreshed; suggested allowance is now 1,470 seconds. The suite output was read once after the planned first deadline, and no later code/test edits supersede the result.

All ordinary run-control, shared assessment, immutable outcome/fact and shared public-result responsibilities are complete. Existing state/result schemas, enum identity, PRNG use, factor/FE mathematics and successful-run behavior are preserved against the repaired baseline. Batching/freezing remain with Colony; exact/outer strategy implementation and public result classification for those paths remain cff-jrbw. Its handoff now records lazy kernel-plan selection and strategy-specific count/tag/certificate rules. PR publication reuses this gate. No significant user decision is pending. Independent agent review is unavailable and is not claimed.

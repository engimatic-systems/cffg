---
id: cff-nyad
status: open
deps: [cff-ny7f, cff-jby3]
links: [cff-5xdy]
created: 2026-09-02T04:17:42Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-ws3c
tags: [engine, architecture, schedule, refactor]
---
# Extract schedule and propagation construction

## Background in Plain English — WHY

At authoring snapshot e488df3, `src/cffg/engine.py:3041-3669` is the scope
anchor for batch stages, registerized/flat scans, support checks, exact units,
and propagation; it is not a promised future line map. The implementation
starts from the exact `cff-sf29` engine baseline. Those builders sit beside the
family sweeps they invoke, allowing a scheduled path to reinterpret a plan
instead of reusing one rule-call owner.

After `cff-ny7f` owns plan execution, this ticket moves only schedule
classification and propagation construction.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-ny7f` is closed with all plan updates using typed execution owners.
2. One schedule owner accepts immutable `Arena` schedule plans and the typed
   execution-owner collection and returns a `ScheduleProgram` for flooding,
   batch stages, scan stages, exact units, support checks, and propagation.
3. Schedule builders decide ordering, repetition, and array/register transport
   from typed execution plans. They do not select family parameter meanings,
   call a family rule directly, compute an objective, or classify a public
   `RunStatus`.
4. Flooding, batch, registerized scan, and flat scan use the cff-ny7f
   rule-call owner for each equivalent update.
   4.1. Registerized scans retain prefetched inputs, local-block carry, and one
        final scatter through the shared local-update interface. They do not
        gain a full-arena carry or gather on each chain step.
   4.2. Existing discrete and Gaussian chain checks retain schedule kind,
        shapes/dtypes, flat persistent bytes, and trace reuse when only runtime
        parameter or observation values change on one Engine.
   4.3. Local solver status/residual facts flow through scheduled and flooding
        updates with the same aggregation rules and fixed-shape array types.
5. Schedule selection, fingerprints, forest priming, exact two-pass behavior,
   damping restrictions, delta hardening, and logical iteration meaning remain
   unchanged.
6. The affected rows are **Runtime numerical or status invariant** and **New
   factor or rule family**. **New IR field** is N/A because schedule plan fields
   do not change. **Learnable or checkpoint feature** applies only to restored
   states consuming the same program.
7. Flooding/scheduled parity is the direct duplicated-surface obligation.
   Direct/one-instance `vmap` uses the same schedule program where supported.
   Host/JIT consumes one static program. Checkpoint decoding remains outside.
8. Literal zero support, vacuous/improper inputs, non-finite updates, float32,
   and uncomfortable-scale schedules retain their baseline values, support
   signal, or loud internal failure.
9. Existing schedule/family oracle tests remain authoritative. Old/new
   schedule fingerprints, messages, and state characterization protect the
   refactor. No new scientific recovery or performance benchmark is added.
10. No schedule kind, factor rule, solver, public status, checkpoint schema,
    diagnostic field, or speed target changes.
11. The schedule module is below 1,000 physical lines and the ticket fits one
    context because its source responsibility is anchored by about 629 lines at
    e488df3 and has one `ScheduleProgram` output.
12. Focused schedule/runtime tests and the final short suite pass.
13. `docs/ARCHITECTURE.md` names the schedule owner and its dependency on the
    execution owner in the extraction commit.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Move `_batch_stage_fn` through `_make_propagate` behind the typed program.
- Make source checks fail if schedule code selects a family rule directly.
- Keep JAX-trace-time specialization static; do not add runtime type strings.

## Future work

`cff-5xdy` moves state/data/mu3 lifecycle services while consuming this one
schedule program.

## Definitions

- **Propagation:** Applying an ordered collection of schedule stages without
  deciding the public termination status.
- **ScheduleProgram:** Static callable stages built from compiled plans and
  typed execution owners.
- **Support signal:** Internal evidence that a state has literal zero support;
  public status classification uses the shared runtime result owner.

## Notes

**2026-09-10T03:17:51Z**

Read-only preparation at cff-jby3 source 84aa318; that prerequisite is still awaiting its final gate. Current extraction anchors are engine.py:2816-3439 (batch, scan classification, registerized and flat scans, support, exact unit, propagation) plus constructor schedule selection/program assembly at :359-442. Synchronous _make_step and the free-energy ledger are separate inputs, not family math to duplicate inside schedule builders.

Proposed boundary: one immutable ScheduleProgram from the validated Schedule, a typed collection of registered execution/parameter-selection owners, the existing synchronous sweep factory, and the existing raw free-energy callback. Schedule construction owns kind selection, priming, mixed-phase ordering, certification orchestration and JIT wrapping. Keep public status decisions, outer elimination policy and objective formulas outside. Move shared segmented transport to a leaf module to avoid importing Engine from the schedule owner. The current EF scan prefetch reads kernel param_keys itself; route it through the existing ef_parameters owner. Move the equality/potential-specific support calculation to the execution support owner; the schedule layer combines its results with edge support. These are ownership moves, not new numerical behavior.

Change paths: Runtime numerical or status invariant and New factor or rule family. Flood/batch/registerized/flat updates, host/JIT support, mixed priming/reconstruction, exact certification, solo/vmap and restored-State consumers are checked pairs. IR, schema, checkpoint decoding, readout math, objective formulas and public statuses are N/A to modification. Existing test_schedule.py has empty-support, poisoned input, doctored schedule, float32 missing-phase and T=256 mixed-scale certificates. The discrete/Gaussian kernel schedule tests cover dynamic parameters/observations, trace reuse and local-register carry without full-arena storage. Preserve these assertions; follow moved private instrumentation to its new owner as needed. Plan additional boundary tests for program composition and no direct family-rule imports, plus bounded old/new schedule/result parity. No implementation started; update the published prerequisite base before branching.

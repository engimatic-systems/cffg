---
id: cff-nyad
status: open
deps: [cff-ny7f]
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
3. Schedule builders decide ordering and repetition only. They do not gather
   family parameters, call a family rule directly, compute an objective, or
   classify a public `RunStatus`.
4. Flooding, batch, registerized scan, and flat scan use the cff-ny7f
   rule-call owner for each equivalent update.
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
  public status classification remains with `Engine.run`.

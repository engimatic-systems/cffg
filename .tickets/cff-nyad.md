---
id: cff-nyad
status: closed
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

**2026-09-10T03:24:40Z**

Starting after cff-jby3 closed and PR #125 was verified at 9cd52b9c4e20ef0ccc1bef74da6fd55dd3eeb1b3. The intended base was pulled with --ff-only before branching. Its final gate passed 1,974 tests at source 84aa318. The preparation note above is the implementation map: use the existing sweep factory and ledger as explicit inputs, extract schedule composition and transport without changing their math, and preserve the listed duplicated surfaces and hostile-input certificates. New schedule code will consume a typed execution-owner collection; it will not import Engine or family numerical modules. Final verification will use existing schedule/family oracles, boundary checks, bounded old/new parity, and one measured default gate.

**2026-09-10T03:37:12Z**

Extraction implemented. schedule_program.py owns validated schedule selection, stage/registerized/flat construction, priming, mixed-phase composition, support combination and exact certification. It returns a frozen ScheduleProgram retained by Engine; existing private callable aliases point into that program. ScheduledExecutors supplies typed registered local updates and parameter/support capabilities. Shared segmented gathers/scatters moved to execution/transport.py. The schedule owner does not import Engine, kernel registries or factor rule modules, and does not inspect family types, bindings or parameter addresses. Synchronous sweep construction, ledger formulas, public run/status control and outer elimination policy remain explicit external responsibilities.

Current evidence: 45 existing schedule/custom-kernel/engine-contract cases passed in 29.32 seconds. Seventeen new boundary cases passed in 2.33 seconds, including float32/float64 program composition, exact sum/max ledger callback ownership, fixed-shape mixed inner-failure transport, forbidden dependency checks and 32-observation EF registerized-versus-flat parity with changed data and trace reuse. Existing private test instrumentation was redirected to the extracted builders without weakening scientific assertions. A bounded comparison script at /tmp/cffg-schedule-parity.py loads trusted baseline Engine source 84aa318 in a separate Python module: 17 models x automatic/flooding x float32/float64, with schedule kind/fingerprint/description/registerization, initial state, three damped updates, numerical status/residual facts, FE, priming, forest reconstruction and exact units where available. All 2,622 array comparisons were bitwise identical, including matched NaNs; one pre-existing scatter dtype warning appears in the functional case. No speedup or new algorithm is claimed.

Ruff/format pass on new modules and tests. Engine lint findings decrease from 23 to 20 with no introduced code/message findings; existing schedule-test findings stay at three. Architecture/change-path owners and dashboard enrollment are updated. The schedule module is 808 physical lines, below the ticket limit. Final source adds only retention of the already-built program after that parity check; the final focused selection and default gate will validate that complete head. No full-suite result yet. Run one final default suite after the source commit, using the measured first-check deadline; publication alone does not require another run.

**2026-09-10T03:39:00Z**

Final combined focused selection passed after retention of the ScheduleProgram: 62 tests in 31.70 seconds across test_schedule_program.py, test_schedule.py, test_gaussian_kernel_schedules.py, test_discrete_kernels.py and test_engine_contracts.py. No assertion weakening, formula change or schema change was needed. Commit this final source, then run the dashboard-backed default gate once. Cached timing: latest wall 1,132.293 seconds; suggested wait 1,440 seconds. First output check is planned at 20 minutes, with a second only at the 24-minute allowance if still running.

**2026-09-10T03:59:27Z**

Final default gate passed: 1,991 tests, zero failures, 41 opt-in slow deselections and four existing Gym deprecation warnings; pytest reported 1,121.93 seconds and measured wall time was 1,135.251 seconds. Completion: 2026-09-10T03:57:56.251866+00:00, recorded against f48def6. The dashboard includes all 17 new boundary cases. Tested code source is 47701a6; the subsequent f48def6 change is ticket-only. The suite was checked once at the scheduled 20-minute deadline, not repeatedly polled. No later source/test changes supersede the gate.

All acceptance criteria are met: schedule construction and certification composition have one typed owner, registerized transport and numeric/status facts match the baseline, the schedule file is below 1,000 lines, existing oracles remain intact and the architecture index is updated. Close this ticket and publish above PR #125. The next ready ticket is cff-7kfk, moving the remaining synchronous transport through the already-defined sweep-factory interface; its source/caller map and downstream lifecycle handoff are recorded. No new user decision is required.

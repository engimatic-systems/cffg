---
id: cdc-ztuf
status: open
deps: [cff-y15d]
links: []
created: 2026-09-14T18:09:57Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-zov5
tags: [rslds, inference, hybrid, numerical]
---
# Prepare explicit starts and the moderate hybrid qualification harness

## Background in Plain English — WHY

The tiny joint solver in cff-y15d provides the numerical iteration, but cff-a3wz must exercise genuine zero-message, feasible, and perturbed starts on larger native-float32 chains. The current public solver entry always builds a feasible initializer, so it cannot represent all three required starts. Preparing an explicit-coordinate entry, deterministic fixtures, and an inspectable execution/storage harness is a bounded implementation task that should be reviewed before expensive qualification runs. This ticket splits that setup out of cff-a3wz; it does not establish numerical feasibility by itself.

## Outcomes / Acceptance Criteria — WHAT

1. A shared solver entry accepts an explicit initial message/auxiliary coordinate while preserving the existing feasible-start entry and its status/certificate semantics. Both use one iteration and terminal-certification implementation. Raw-zero mode really starts from zero message coefficients and the named xi value; it is not a renamed feasible initialization. Metadata, invalid coordinates and unsupported support changes retain the existing truthful failure behavior.
2. A committed machine-readable pre-run specification builds exactly the fixed cff-a3wz formulas at T=32 and T=100, K=4, d=2. Freeze the twelve combinations T x xi in {0,1} x {raw zero, feasible, fixed non-oracle perturbation}. Record the perturbation, strategy, outer/trial/linear budgets, no-restart policy, dtype-scaled thresholds, initialization checks, comparison fields, resource diagnostics and first-failure preservation policy before executing a moderate solve. Reuse qualified Y15D defaults unless a concrete inventory justifies a prospectively documented ordinary choice; no silent tuning or numerical threshold changes.
3. A standalone runner starts native-float32 subprocesses with jax_enable_x64 false before numerical imports and asserts the actual configuration and floating leaf dtypes. Preserve source/spec/git/version identifiers and synchronized solve timings. Saved records include all terminal certificates, properness/finite free energy, natural message and auxiliary coordinates, beliefs, all source-conditioned learning groups, counters and initialization failures. Compare starts by returned outputs and disclose meaningful differences without requiring universal uniqueness.
4. The storage harness inventories parameters, fixed metadata, coordinate/evaluation/derivative/linear/trial/outer/terminal leaves including low words and integer limbs. Measure actual enclosing iteration and complete-solve compilation with dynamic arguments at both horizons and independently varied outer/linear caps; outer-cap comparisons must include the complete solve rather than a step whose body cannot depend on the outer cap; keep XLA argument/output/temporary/alias bytes separate from logical leaves and compiler/process memory. Include K=20,d=10 shape inventory. No dense global Jacobian, trajectory covariance, retained optimization history or path enumeration. The prerequisite implements and focused-tests the measurement path; cff-a3wz owns execution and interpretation of the moderate measurements.
5. Meaningful focused checks verify formulas/shapes, raw-zero versus feasible initialization, shared host/JIT and single/vmap status behavior, native-float32 process setup, exact run inventory and rejection of incomplete/malformed reports. Default tests do not execute the twelve moderate optimizers; expose those as opt-in slow validation. Register new short test files in dashboard EPIC_FILES.
6. Before expensive checkpoint execution, commit the implementation and specification, complete focused checks and pause for supervisor review. After fixes, preserve existing immutable tiny reports and produce any newly required source-matched tiny evidence; do not rewrite old hashes or present an earlier run as a new one. Complete one final short-suite gate after the final relevant edit and close this prerequisite with reproducible commands/results. cff-a3wz remains open until actual moderate numerical and storage qualification passes.

## Methodological Suggestions — HOW

Use cff-a3wz's committed preflight mapping and formula rather than adapting the tiny reference factory: L=T-1 transitions, source i / destination j / stick l. Keep the moderate fixture independent of the candidate's solver implementation. Reuse hybrid_joint_storage.programs/arrays where their semantics match, and extend the actual shared solver entry instead of copying solve logic into a test harness. Report unavailable physical memory analysis explicitly with the named substitute. Keep report helpers bounded rather than growing existing large test modules.

## Future work

cff-a3wz executes the twelve frozen cases and the actual moderate compiled/iteration storage measurements, diagnoses any failure, and owns the decision to proceed. cff-k6g5 and cff-7sw5 cover declarations and compiler plans; production integration and the registered three recovery fits remain behind TK dependencies. This ticket does not authorize new recovery fits, supported-scope changes, approximation changes or weaker gates.

## Definitions

- Explicit coordinate: the gauge-reduced message coefficients and active squared logistic auxiliary values supplied to the solver.
- Native float32: JAX's consumer precision with 64-bit mode disabled, not float32 arrays inside an x64-enabled process.
- Feasible start: the declared initializer's proper initial beliefs; raw zero is a separate start.
- Logical bytes: the sum of described array leaves, which may share physical storage.
- Enclosing iteration: the complete accepted/trial/linear numerical step compiled together, rather than a sum of isolated kernel peaks.

## Change paths and hostile inputs

Applicable architecture rows: Derived component coordinate and Runtime numerical or status invariant. Affected pairs: unbound host/JIT, single/vmap, explicit/feasible initialization, iteration/terminal certification, logical/compiled storage, runner/saved-report validation. Production compiler, schedule, lifecycle, model selection, serialization, public learning, persistence and registered recovery are N/A. Preserve exact sparse log-zero support and inactive xi=0, reject empty/changed support and nonfinite active coordinates, distinguish tiny underflow from support removal, permit signed/vacuous messages only when the resulting active beliefs are proper, and retain dtype-scaled guards. Test native float32 mechanisms and inventory T32/T100 plus K20/d10; moderate convergence itself belongs to cff-a3wz.



## Notes

**2026-09-14T18:09:57Z**

Setup is split before implementation into cdc-ztuf: a fresh worker owns the shared explicit-coordinate entry, twelve-case native-float32 fixture/specification, and numerical/storage harness. cff-a3wz retains all actual moderate convergence, initialization sensitivity and iteration-memory execution/interpretation, plus the integration decision. Y15D tiny evidence and final short-suite publication finish first. No fixture formula, approximation, supported scope, tolerance, storage requirement or recovery gate changes.

Legend: prerequisite --> dependent. TK records remain authoritative.

[~] cdc-bwta (ready; root final suite gate)
  --> [~] cff-y15d (author fixes committed; re-review/evidence/gate pending)
        --> [ ] cdc-ztuf (setup; immediately blocked by Y15D)
              --> [ ] cff-a3wz (twelve moderate starts and actual storage)
                    --> [ ] cff-7sw5 (compiler plans)
                              ^
[ ] cff-k6g5 (ready; independent declarations) --+

cff-7sw5 and cff-a3wz feed the remaining integration/state/ledger chain before cff-f3jz, learning/composition, cff-x0if's three frozen recovery fits and cff-zov5. This companion comes from tk dep tree --full cff-a3wz, tk ready and tk blocked; no registered recovery fit has run.

**2026-09-14T20:01:41Z**

Independent instrumentation preflight (read-only Darwin arm64, CPython3.12.13, JAX/jaxlib0.10.2). Question: how should native-float32 timing/storage evidence be measured without extra optimizer executions? Answer: separately time jax.jit(function).lower(dynamic ShapeDtypeStruct arguments), lowered.compile(), and the first authorized Compiled call through jax.block_until_ready. Retain the executable for all remaining authorized starts with the same shapes/dtypes/static configuration; do not add a warm-up or timing repeat. Synchronize prepared inputs before an execution-only timer and serialize results afterward. time.perf_counter_ns uses a monotonic non-adjustable mach_absolute_time clock here. With a persistent cache, compile() time is executable acquisition unless a miss/cold compilation is established; a fresh Python process alone does not establish cold compilation.

Native-float32 caveat: set JAX_ENABLE_X64=false for the fresh child, immediately assert/configure x64 false after importing JAX and before workload imports, and check actual floating leaves throughout inputs/metadata/coordinates/evaluations/derivative buffers/results. Installed jax_explicit_x64_dtypes='allow' can permit explicit f64 despite x64=false; record and constrain that policy so it cannot undermine the claim. Integer/status/limb leaves remain separate. Root additional source check: tests/run_hybrid_joint_candidate.py explicitly enables x64 on import, so the moderate runner must not import that tiny runner for convenience helpers. Record cache policy/location and hit/miss evidence.

Keep logical arrays, XLA estimates and process peaks separate. storage.arrays/buffers cover leaf shapes/dtypes/bytes and derivative/linear/trial/outer groups; aliases can double-count logical totals. memory_analysis supplies argument/output/temp/alias byte fields; None is unavailable, not zero. Installed CompiledMemoryStats also exposes generated-code, host_* and peak_memory_in_bytes fields, but Python binding semantics for backend-specific extras are limited; label them as XLA estimates, never process RSS. Compile complete solve variants at independently declared outer caps: an isolated iteration cannot reveal outer-loop history. Cap-only memory variants need no additional optimizer execution.

On this macOS, resource.getrusage(RUSAGE_SELF).ru_maxrss is BYTES (divide by2**30 for GiB), unlike Linux KiB. It is cumulative high-water memory, so differences between checkpoints do not yield independent phase peaks. Current ps snapshots are not peaks. /usr/bin/time -l can independently wrap the actual Python workload inside the uv environment through termination; do not accidentally measure only a parent. Reusing executables across starts also means child peak RSS covers accumulated activity, not isolated per-start peaks.

Sources: hybrid_joint_storage.py:16,58,80; tests/run_hybrid_joint_candidate.py:16; installed jax/_src/stages.py:586,651,711,881; api.py:2318,2479; config.py:1535,1613,1688; dtypes.py:987; compiler.py:429; xla_bridge.py:364; CompiledMemoryStats binding and time API metadata; /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/share/man/man2/getrusage.2:94 and /usr/share/man/man1/time.1:74. Confidence high for API/control-flow/unit facts; backend-specific extra-memory-field interpretation remains explicitly limited. No numerical arrays, compilation, tests or solves ran during research. These are implementation cautions, not new thresholds, schema or numerical qualification evidence.

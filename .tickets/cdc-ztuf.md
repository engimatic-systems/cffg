---
id: cdc-ztuf
status: in_progress
deps: [cff-y15d, cdc-ogj9, cdc-c8kc]
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

**2026-09-14T21:44:41Z**

Implementation plan recorded before code. Requested branch cdc-ztuf-moderate-harness is clean at published PR148 base d5e009876e3ab8ad3db0a7138437f429dc61128e; root already bootstrapped it, so no branch/base switch. Reuse the exact independent tools/d12_compile_fixture.py formula without changing historical compile evidence. Add explicit initial-coordinate state/solve entries feeding the same iteration and terminal certification as the feasible API. Freeze twelve T32/T100 x xi0/1 x raw-zero/feasible/perturbed cases in a new machine-readable specification, inheriting Y15D budgets and 0.01*sin(index+1) message-only perturbation. Add bounded native-float32 process runner, strict saved-output/provenance/inventory validation, pairwise start comparisons, dynamic complete-solve and enclosing-iteration memory programs, and K20/d10 abstract shape inventory. Use existing compiler/process instrumentation only where semantics match; no moderate solve, memory compile, recovery fit or tiny checkpoint execution before review. Focused default tests will use tiny held coordinates, abstract large shapes, scalar instrumentation, malformed synthetic reports and native subprocess setup; opt-in slow evidence validation does not silently start optimizers. Register new files in EPIC_FILES. Applicable architecture rows: Derived component coordinate and Runtime numerical or status invariant. Affected pairs: host/JIT, single/vmap, feasible/explicit initialization, iteration/certification, logical/compiled storage and runner/report validation. Checked N/A: production compiler/IR, scheduling, lifecycle, model selection, serialization/persistence, public learning and recovery. Hostile cases retain exact support/inactive xi=0; changed/empty support, negative or nonfinite active xi and malformed coordinates fail truthfully; no clamping of signed/vacuous messages; properness and finite FE remain required. Native float32 policy is x64=false plus explicit-x64=error before workload imports and checked leaves/StableHLO. No thresholds or supported scope change. Solver-source changes invalidate the source-match check for immutable attempt003; a new immutable tiny attempt is required after review, without rewriting old reports/hashes. Commit implementation/spec/focused evidence and pause in_progress; root owns review sequencing, full short-suite gate, publication and A3WZ numerical qualification.

**2026-09-14T22:04:20Z**

Focused-check diagnosis before review: the first native T2 status check used an outer cap of zero but JIT still traced/compiled the complete unused iteration body. The check was stopped by this author after 225.81 seconds (pytest result failed, child SIGTERM -15) during compilation; no moderate optimizer or source-matched tiny checkpoint ran. Host zero-budget feasible/explicit comparisons had completed before the blocked compilation. Do not treat this as passing evidence. The bounded replacement checks the real explicit initialization and terminal certification under host/JIT/vmap, with host zero-budget and abstract complete-wrapper checks; full compiled solve evidence remains after independent review. No solver performance workaround, budget increase, threshold change or runtime-history edit was made.

**2026-09-14T22:19:25Z**

Pre-review implementation and focused evidence complete; pause in_progress for root to start independent review. The shared explicit initial_state_from_coordinate/solve_from_coordinate entries preserve raw message coefficients, original invalid coordinates and fixed-context checks. Feasible and explicit starts feed one _initial_state, one _solve loop, the existing iteration and the existing returned-point certification. No numerical thresholds or iteration formulas changed.

The prospective tools/d12_moderate_spec.json fixes exactly twelve T32/T100 x xi0/1 x raw-zero/feasible/perturbed starts and unchanged Y15D numerical budgets (outer200, trials24, linear128, backtrack0.5, Armijo0.0001, no restarts). It reuses the independent PR147 fixture without altering that fixture or historical compilation evidence. Memory-only variants independently double outer to400 or linear to256; outer comparisons compile complete solves. Ten compilation children make zero executable calls; a separate shape-only child inventories T32/K4/d2, T100/K4/d2 and T100/K20/d10 without contaminating compilation-child process peaks. All three abstract inventories passed, with N/M=1798/3286,5742/10494,298782/560142 and773 derivative capture leaves per shape. These are logical shapes, not actual moderate storage qualification.

The runner configures fresh CPU children with x64=false and explicit-x64=error before workload imports. It checks native floating leaves and StableHLO, separately measures lower/compile/synchronized authorized calls/export, retains one solve executable across the six starts at each horizon, and never runs memory-variant optimizers. Lossless immutable NPZ snapshots retain every initial/terminal State leaf, natural coordinates, beliefs and centered/physical learning readouts; hashes, source/spec/git/runtime identities, compilation logs and real process high-water remain separate from logical/XLA bytes. Incomplete runs and malformed records cannot qualify. All30 within-horizon start pairs disclose material differences using the frozen threshold. First failures preserve their attempted prefix and stop without retry.

Final focused commands/results:
- uv run pytest -q tests/test_d12_moderate.py:25 passed,2 deselected,21.54s after final harness edits. Default tests execute no moderate optimizer; they cover exact formulas/inventory, lossless and malformed reports, full dynamic program descriptors, native process setup, actual abstract buffer contracts and scalar timing instrumentation. Registered in dashboard EPIC_FILES.
- uv run pytest -q -m slow tests/test_d12_moderate.py -k explicit_host_jit_single_vmap_status:1 passed,26 deselected,203.28s. This opts into the bounded native T2 check:2 host complete zero-budget calls,9 real initial/terminal checks under JIT/single/vmap, four status variants, malformed coordinates, and abstract complete-wrapper tracing. It does not compile or execute complete iterative optimizers. Subsequent changes to this helper were formatting/import cleanup and the independent default shape-check function; no status-check or solver behavior changed. The earlier stopped full-body compilation remains recorded above as a failed check, not passing evidence.
- uv run pytest -q tests/test_hybrid_joint_coordinate.py tests/test_hybrid_joint_solver.py -k 'full_support_shape_formulas or packing_rejects_malformed or invalid_message_metadata or stale_support_and_origins or complete_loop_has_fixed_int32_status or compiled_memory_interface':7 passed,88 deselected,89.46s.
- uv run python tools/d12_compile_checks.py:7 passed,0.231s; includes independent fixture formulas, real resource-stop preservation, Darwin/Linux memory units and scalar compilation evidence.
- uv run --with ruff ruff format --check tools/d12_moderate*.py tests/test_d12_moderate.py and uv run --with ruff ruff check tools/d12_moderate*.py tests/test_d12_moderate.py --select F:passed. Ruff was an ephemeral uv tool; no dependency/lockfile change.
- uv run python tools/d12_moderate_runner.py plan > /tmp/cdc-ztuf-plan.json:passed. A separate runtime_identity(native_jax()) JSON serialization probe confirmed x64=False and explicit policy ERROR.
- git diff --check and tk dep cycle:passed, no cycles.

Required after review: fix reviewed issues in this same author/branch, then generate a NEW immutable source-matched tiny attempt and select it in the source-binding test. Exactly hybrid_joint_solver.py and hybrid_joint_storage.py differ from attempt003's source manifest. Attempt003 is untouched, still SHA2563cfa37ede07bba2a872c7938f66c208207a2f9a7059d758e92685ecb9bc75bee; never rewrite its report or hashes. The current frozen-source test intentionally cannot match the new source until fresh evidence exists. Root owns the final short-suite gate, publication and ticket closure. A3WZ owns actual moderate solve/storage execution, initialization-sensitivity diagnosis and the qualification decision. No full suite, runtime-history edits, moderate solves, moderate memory compilation, recovery fits, expensive new tiny checkpoint, push or PR ran here. The implementation/specification/focused evidence are ready to commit and pause; no reviewer or implementation worker was spawned by this author.

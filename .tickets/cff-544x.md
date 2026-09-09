---
id: cff-544x
status: closed
deps: [cff-ecw0]
links: [cff-8t4d]
created: 2026-09-02T04:02:00Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-edoz
tags: [compiler, architecture, refactor]
---
# Move compiled plan types behind a one-way boundary

## Background in Plain English — WHY

At authoring snapshot e488df3, `src/cffg/compiler.py:77-602` is the source-
region anchor for compiled plan records and `Arena`. The exact `cff-ecw0`
baseline is the completed `cff-209l` head, where those definitions may have
moved. Runtime, schedule, estimator, learning, and agent code import the values
from the compiler implementation module. That makes the compiler both a data-
contract owner and an orchestration owner.

`cff-ecw0` first records the observable contract. This ticket moves compiled
values to a focused lower-level owner and makes their published data read-only.
Cooper approved combining these changes on 2026-09-09. This is an intentional
mutability change, not a promise to preserve the old mutable containers.

Today the engine captures arena metadata in compiled functions, while schedules
and readouts also consult it. Changing that shared metadata after construction
can make those consumers disagree. Compiler builders may remain mutable until
publication. Per-instance messages, observations, and learned parameters remain
runtime State values; making the arena immutable must not prevent their updates.
This ticket does not extract lowering logic or change field names or values.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-ecw0` is closed with its characterization evidence passing.
2. Every compiled plan record and `Arena` used outside compiler orchestration
   has one focused internal owner.
3. The compiled-value owner imports neither `compile_template` nor public
   `Engine`, schedule execution, estimator, agent, or model helpers.
4. Published arena and plan data is immutable through ordinary Python/NumPy
   operations, including nested mappings, sequences, and numeric arrays.
   4.1. Retained caller-owned containers or writable array views cannot change
        published values. Construction does not make caller-owned inputs
        read-only as a side effect.
   4.2. Reusing an already immutable value during arena assembly or
        `dataclasses.replace` does not copy its numeric buffer again.
   4.3. Existing field names, numeric dtypes, shapes, element order, and values
        are preserved. Container representation may change to enforce the
        approved read-only contract; those changes are documented and tested.
        Numeric fields use an owned array value whose NumPy/JAX exports detach
        storage. Requesting a zero-copy NumPy export fails explicitly; external
        callers must not be able to acquire a writable alias to stored data.
   4.4. Mutable compiler builders stay private. Consumers receive complete
        values and update inference/learning data only through runtime State.
   4.5. Captured kernel handles keep their registered identity. This contract
        protects owned plan data, not arbitrary state inside trusted Python
        callbacks, and is not a security sandbox.
5. Runtime, schedule, estimator, learning, agent, and model consumers import
   compiled values from the focused owner rather than from compiler
   implementation helpers.
6. Existing documented imports of `compile_template`, `CompileError`, and
   supported compiled values continue to work through their canonical owner or
   a compatibility re-export.
7. Every `cff-ecw0` characterization numeric value, address, order, identity,
   and existing failure diagnostic remains unchanged. Tests explicitly record
   the intentional immutable-container representation where relevant.
8. JSON and CFFGZ reconstruction, flooding and scheduled execution, direct and
   `vmap` consumers, and latest-schema checkpoint reconstruction use the same
   moved values.
9. The move changes no array dtype, support mask, numerical guard, message,
   objective value, runtime status, structure hash, or serialized field.
   Sparse-support, improper-input, float32, and scale fixtures remain
   integration regressions rather than new plan-type behavior.
10. Import-cycle checks show one-way dependency from core declarations and
    kernels to compiled values, then to runtime consumers.
11. The focused compiler and affected consumer tests pass. The
    dashboard-backed short suite passes after the final code change.
12. `docs/ARCHITECTURE.md` names the compiled-value owner in the same commit
    as the move.
13. The work stays bounded to value ownership, immutable publication, and
    direct consumers. Publication costs are measured on a small arena and the
    32-group fixture; freezing is not added to inference iterations or performed
    once per instance for shared structural metadata. Report allocation and
    construction-time tradeoffs without promising an inference speedup.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Move values without renaming fields in one commit, then enforce immutability
  in a separate atomic commit so the behavioral change is reviewable.
- Preserve compatibility re-exports until all internal consumers use the new
  owner.
- Use the characterization tests as value-level evidence; do not rely only on
  unchanged high-level model output.
- Use all four architecture change-path rows as a consumer checklist. This
  ticket changes transport ownership, not their semantics.
- Do not rely on NumPy's writeable flag or bytes-backed ndarray views: installed
  NumPy 2.5.1 allows ufunc.at to modify those buffers, including a NumPy view of
  a CPU JAX array. Own storage behind an explicit export boundary, and reuse
  published values rather than copying at each containing record.

## Future work

`cff-8t4d` uses the moved values and the publication policy for typed compiler
outputs. `cff-ddsq` applies the same policy at final assembly. There is no
separate immutability ticket under Cooper's revised decision.

## Definitions

- **Compatibility re-export:** An old import path that delegates to the new
  canonical definition during migration.
- **Compiled value:** Immutable metadata or arrays derived from serialized
  model structure for runtime use.
- **Publication:** Returning a completed arena or plan for consumers to share.
- **Alias:** Another reference or array view pointing to the same underlying data.
- **Runtime State:** Per-instance messages, observations, parameters, and solver
  data that change during inference without changing the compiled blueprint.
- **Partly initialized:** Missing required fields that another function is
  expected to fill later.

## Notes

**2026-09-09T16:43:59Z**

Pre-implementation research from cff-ecw0 (no extraction started). Question: can the planned pure move also deliver enforced immutability? Answer: no. At baseline bf6bbe9, compiler.py:450 uses an unfrozen Arena dataclass; frozen plan dataclasses still contain writable NumPy arrays, dictionaries and lists (including LearnPlan.shapes). Read-only inspection found no production consumer mutation of compiled metadata, which establishes an ownership convention, not enforcement. Engine.init_state creates separate runtime parameters; learning and named writes update runtime State rather than compiled addresses. Minimal type consumers are engine.py, outer.py, schedule.py (Arena), plus TYPE_CHECKING imports in learn.py (LearnPlan) and params.py (MemberAddress/ParameterAddress). Source: compiler.py:54-611, engine.py:41/460, schedule.py:58/292-304, learn.py:65/524, params.py:14/200. Confidence: high, direct source inspection by the modularity research agent. Existing AC4 conflicts with pure-move/no-field-change scope if immutable means enforced. Asked Cooper whether to split enforced immutability into its own behavior-change ticket and preserve current representation during extraction; decision pending. Any enforcement ticket must specify container types, NumPy writeability, alias/copy ownership and dataclasses.replace compatibility. Do not silently claim that moving classes or setting frozen=True provides deep immutability.

**2026-09-09T17:06:35Z**

Revised decision: Cooper explicitly approved enforcing immutability within this extraction. No separate ticket was created. Branch agent/cff-544x-immutable-values starts from updated agent/cff-ecw0-compiler-contracts at 1c246f3 (PR #113). All four ARCHITECTURE change-path rows apply to transport ownership and consumers; there are no model/schema/rule changes. Pairs: flooding/scheduled, direct/vmap and current/supported-legacy reconstruction consume the same immutable arena; host/JIT receive one host-built value, with no second compiler. Sparse/zero-support and improper/vacuous runtime values retain existing numerical behavior; float32/float64 arrays preserve their current compiler dtypes and values. The 32-group fixture and publication measurements cover scale. Implementation will separate the mechanical ownership move from the explicit immutability behavior change in atomic commits, preserve compatibility re-exports, and leave runtime parameter/observation updates functional.

**2026-09-09T17:32:55Z**

Implementation/research handoff: moved all 19 compiled value records plus NEG/PARAMETER_TRANSFORMS to cffg.compiled in 0dab458, preserving compiler re-exports and redirecting direct consumers. Immutability uses frozen records, tuples, FrozenMap, and private-storage FrozenArray; constructors isolate caller aliases and replacement/assembly reuse published payloads. Question: do bytes-backed NumPy arrays or CPU JAX arrays provide the required public-alias protection? Answer: no on NumPy 2.5.1/JAX 0.10.2: np.add.at(np.asarray(value), ...) mutates nominally read-only backing storage. Source: local performance-agent probes; confidence high, reproduced for integer/f32/f64. Chosen explicit detached NumPy/JAX exports; JAX index consumers convert at their boundary. Follow-up review found no remaining public alias leak in 54 JAX export, 13 ufunc, and six view probes, and no replacement/assembly payload recopy. Host scalar/address operations stay NumPy; JAX remains the numerical inference backend. Focused checks so far: 96 compiled/Gaussian/EF/parameter/schedule tests plus 124 characterization/discrete/precision/hybrid-learning/CSP/dashboard tests passed. Final short-suite gate and construction-cost evidence pending. No serialized callback or schema changes.

**2026-09-09T17:34:48Z**

Construction evidence at code head 315b082 vs mechanical-move baseline 0dab458: sequential warmed CPU measurements, median of five batches of ten compilations, exclude JAX tracing/inference. HMM: 14.471 -> 14.626 ms; Gaussian: 71.001 -> 70.698 ms (timing noise, no speedup claim); EF: 0.071 -> 0.167 ms; 32 distinct groups: 93.980 -> 95.861 ms. Retained numeric payload unchanged: 512/672/208/6048 bytes respectively. One traced allocation sample: EF peak 12017 -> 19713 bytes; 32-group peak 175123 -> 198422 bytes. These are Python/NumPy tracemalloc samples, not total process/device memory. Initial publication snapshots mutable inputs; detached exports may allocate; no per-iteration or per-shared-instance structural republication. Reproduction script: ~/agent-share/CFFG architecture overhaul/measure-immutable-values.py; run using uv with PYTHONPATH=<chosen worktree>/src:<repo>/tests. Exact baseline worktree: /tmp/cffg-immutable-measure.xSoMyd/baseline. Final focused ownership file: 41 passed, including caller aliases, NumPy ufunc.at, JAX exports, dtype/bit preservation, copy/asdict, import direction, and shared-arena runtime/vmap isolation. Final dashboard gate running at 315b082.

**2026-09-09T17:36:58Z**

Dependency handoff (records checked with tk dep tree --full, tk ready, tk blocked; tk dep cycle is clear). Legend: prerequisite --> dependent. [x] cff-209l --> [x] cff-ecw0 --> [in_progress] cff-544x --> [blocked by cff-544x] cff-8t4d --> cff-kfo7 --> cff-ejib --> remaining compiler owners --> cff-edoz --> engine owners --> cff-cfqg. cff-544x is the only ready root in this serial extraction chain; unrelated ready roadmap tickets are not pulled into its scope. Final validation is running at committed code head 315b082; subsequent edits so far are ticket/share-report evidence only.

**2026-09-09T17:50:57Z**

Final gate passed at code head 315b082: 1663 passed, 41 opt-in slow tests deselected, four existing Gym deprecation warnings; pytest 980.69 seconds, measured wall 992.697 seconds. Dashboard refreshed with 41 new tests, zero failures, 30 diagrams and 23 galleries. The code has not changed since this gate; closure/publication edits only tickets and runtime evidence, so no second unchanged suite run. All acceptance criteria are met: canonical one-way value owner and compatibility imports; explicit alias-isolated publication and copy policy; preserved numeric/address/order/hash/diagnostic contracts; existing JSON/bundle/checkpoint, flooding/scheduled and direct/vmap consumers pass. Next work is cff-8t4d topology extraction, whose preparation note preserves observation-row ownership and error reporting order. No remaining user decision blocks that ticket.

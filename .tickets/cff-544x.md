---
id: cff-544x
status: open
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

`cff-ecw0` first records the observable contract. This ticket moves only the
immutable compiled values to a focused lower-level owner. It does not split
lowering logic or change any field.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-ecw0` is closed with its characterization evidence passing.
2. Every compiled plan record and `Arena` used outside compiler orchestration
   has one focused internal owner.
3. The compiled-value owner imports neither `compile_template` nor public
   `Engine`, schedule execution, estimator, agent, or model helpers.
4. Compiler stages construct complete immutable values. No consumer receives a
   partly initialized arena or mutates plan metadata after construction.
5. Runtime, schedule, estimator, learning, agent, and model consumers import
   compiled values from the focused owner rather than from compiler
   implementation helpers.
6. Existing documented imports of `compile_template`, `CompileError`, and
   supported compiled values continue to work through their canonical owner or
   a compatibility re-export.
7. Every `cff-ecw0` characterization value and failure diagnostic remains
   unchanged.
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
13. The ticket fits one implementation context because it moves the bounded
    value-definition region without changing fields or lowering behavior.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Move values without renaming fields in the first commit. Change imports in a
  separate atomic commit if that makes review clearer.
- Preserve compatibility re-exports until all internal consumers use the new
  owner.
- Use the characterization tests as value-level evidence; do not rely only on
  unchanged high-level model output.
- Use all four architecture change-path rows as a consumer checklist. This
  ticket changes transport ownership, not their semantics.

## Future work

`cff-8t4d` uses the moved values as typed outputs from compiler stages. No
other future work is identified.

## Definitions

- **Compatibility re-export:** An old import path that delegates to the new
  canonical definition during migration.
- **Compiled value:** Immutable metadata or arrays derived from serialized
  model structure for runtime use.
- **Partly initialized:** Missing required fields that another function is
  expected to fill later.

## Notes

**2026-09-09T16:43:59Z**

Pre-implementation research from cff-ecw0 (no extraction started). Question: can the planned pure move also deliver enforced immutability? Answer: no. At baseline bf6bbe9, compiler.py:450 uses an unfrozen Arena dataclass; frozen plan dataclasses still contain writable NumPy arrays, dictionaries and lists (including LearnPlan.shapes). Read-only inspection found no production consumer mutation of compiled metadata, which establishes an ownership convention, not enforcement. Engine.init_state creates separate runtime parameters; learning and named writes update runtime State rather than compiled addresses. Minimal type consumers are engine.py, outer.py, schedule.py (Arena), plus TYPE_CHECKING imports in learn.py (LearnPlan) and params.py (MemberAddress/ParameterAddress). Source: compiler.py:54-611, engine.py:41/460, schedule.py:58/292-304, learn.py:65/524, params.py:14/200. Confidence: high, direct source inspection by the modularity research agent. Existing AC4 conflicts with pure-move/no-field-change scope if immutable means enforced. Asked Cooper whether to split enforced immutability into its own behavior-change ticket and preserve current representation during extraction; decision pending. Any enforcement ticket must specify container types, NumPy writeability, alias/copy ownership and dataclasses.replace compatibility. Do not silently claim that moving classes or setting frozen=True provides deep immutability.

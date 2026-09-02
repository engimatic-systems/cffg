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

The compiler currently defines every compiled plan record and the full
`Arena` beside the code that constructs them. Runtime, schedule, estimator,
learning, and agent code import those values from the compiler implementation
module. That makes the compiler both a data-contract owner and an
orchestration owner.

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
   `vmap` consumers, and current checkpoint reconstruction use the same moved
   values.
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

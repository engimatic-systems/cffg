---
id: cff-ddsq
status: open
deps: [cff-bfr9]
links: [cff-edoz, cff-ws3c]
created: 2026-09-02T04:17:42Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-edoz
tags: [compiler, architecture, refactor]
---
# Assemble immutable arenas through slim compiler orchestration

## Background in Plain English — WHY

At e488df3, `compile_template` computes final support groups, slot prefix
offsets, delta/interior edge tables, semiring exclusions, and the `Arena`
constructor across `src/cffg/compiler.py:1819-1936`. Earlier parts of the same
function also leave intermediate state for that constructor.

The previous compiler tickets replace those intermediates with complete typed
values. This ticket gives final invariant checking and immutable assembly one
owner, leaving `compile_template` as readable sequencing only.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-bfr9` is closed with `ResolvedCompileData` passing.
2. One assembly owner accepts `TopologyLayout`, `RegisteredPlanDrafts`,
   `SpecializedPlanDrafts`, `ResolvedCompileData`, and semiring; it returns one
   complete immutable `Arena` or a structured `CompileError`.
3. The assembler computes slot prefix arrays, interior-edge ledger groups,
   delta pairs, final plan collections, structure/fingerprint inputs, and
   semiring compatibility without reclassifying a factor type.
4. `compile_template` performs only plate expansion/validation and sequential
   calls to topology, registered lowering, specialized lowering, data
   resolution, and assembly.
5. All four architecture change-path rows are checked because the final value
   serves compiler, runtime, serialized-model, and learning consumers. The
   assembler introduces no new semantic interpretation for any row.
6. Flooding/scheduled, direct/`vmap`, and latest-schema/supported-legacy paths
   consume the same `Arena`. Host/JIT execution receives this one host-built
   value.
7. Sparse/vacuous/improper messages are runtime N/A. Exact support masks,
   float32 dtypes, and the 32-group fixture preserve the baseline offsets and
   plan order needed downstream.
8. Every `cff-ecw0` compiler characterization remains unchanged. Existing
   family oracle tests remain authoritative, while old/new compiler
   characterization protects this structural refactor. A newly noticed
   scientific evidence gap becomes a separate ticket and does not enlarge this
   ticket unless it proves assembly unsafe.
9. No factor, rule, approximation, objective term, schema, checkpoint field,
   status, or public compiler call form changes.
10. `compile_template` and its public compatibility surface are below 250
    physical lines; the whole public compiler orchestration module is below
    1,000 physical lines; no extracted file exceeds 1,000 lines.
11. This ticket fits one implementation context because all earlier stages are
    already complete and it owns only invariant validation plus construction of
    one immutable output.
12. Focused compiler/integration tests and the final short suite pass.
13. `docs/ARCHITECTURE.md` records the final sequence, inputs, outputs, and
    one-way imports in the assembly commit.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Make the assembler pure over complete typed values.
- Keep compatibility exports separate from assembly logic.
- Use field-by-field compiler characterization instead of a new opaque golden
  arena blob.

## Future work

Closing this ticket closes compiler track `cff-edoz`. Engine characterization
in `cff-sf29` then records the runtime baseline against these compiled owners.

## Definitions

- **Assembler:** The owner that checks cross-stage invariants and constructs
  the complete immutable arena.
- **Compiler orchestration:** Ordered calls to focused compiler stages without
  duplicating their interpretations.
- **Prefix array:** Cumulative offsets that map logical slots into flat arrays.

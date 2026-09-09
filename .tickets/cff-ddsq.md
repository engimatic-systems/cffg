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

At authoring snapshot e488df3, `compile_template` computes final support groups,
slot prefix offsets, delta/interior edge tables, semiring exclusions, and the
`Arena` constructor across `src/cffg/compiler.py:1819-1936`. That range is a
scope anchor, not a promised future line map. Earlier parts of the same
function also leave intermediate state for that constructor. The semiring
exclusions move to public option checking and the registered/specialized
compatibility owners; they are not an assembly responsibility.

The previous compiler tickets replace those intermediates with complete typed
values. This ticket gives final invariant checking and immutable assembly one
owner, leaving `compile_template` as readable sequencing only.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-bfr9` is closed with `ResolvedCompileData` passing.
2. One assembly owner accepts `TopologyLayout`, `RegisteredPlanDrafts`,
   `SpecializedPlanDrafts`, `ResolvedCompileData`, and semiring; it returns one
   complete immutable `Arena` or a structured `CompileError`.
3. The assembler consumes resolved slot prefix arrays and specialized delta
   pairs, and constructs interior-edge ledger groups, final plan collections,
   and structure/fingerprint inputs without reclassifying a factor or
   reevaluating family-specific semiring compatibility. Prefix arrays are
   resolved by `cff-bfr9` before learning validation to preserve failure order.
   3.1. It mechanically joins raw topology, specialized semantic edge tags,
        drafts, and validated numeric payloads. No plan requires a missing
        blob-derived field before resolution.
   3.2. It verifies exactly one execution owner for each factor or absorbed
        member and the declared objective ownership/exclusions. A shared
        parameter contribution is accounted once, independent of its users.
4. `compile_template` performs only accepted-semiring option checking, plate
   expansion, the existing core `validate` call, and sequential calls to
   topology, specialized discovery, registered lowering, specialized
   completion, data resolution, and assembly. Discovery and completion use
   one specialized owner. Family-specific semiring compatibility stays with
   its registered or specialized owner.
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

## Notes

**2026-09-09T21:53:23Z**

cff-bfr9 now publishes immutable slot/observation prefixes at the original final resolution checkpoint before learning validation. This small ownership correction preserves Python constructor-argument evaluation order from compiler.py:606-640 at701c10f; otherwise a later invalid learnable could hide an earlier prefix-overflow error. Assembly consumes these arrays without republishing numeric storage. Delta pairs already belong to specialized drafts. Acceptance criterion3 updated; no extra scientific or public contract change. Prerequisite --> dependent: cff-ejib (final gate running) --> cff-bfr9 (immediate blocker) --> cff-ddsq --> cff-edoz --> engine track --> cff-cfqg. No parallel prerequisite within this chain; tk records are authoritative.

**2026-09-09T22:57:05Z**

Preparation during bfr9 final gate at339c46a, not an implementation start. Remaining assembly is compiler.py:129-134 edge tagging plus142-199 Gaussian/EF interior tables and Arena construction. Inputs now include ResolvedCompileData with immutable prefixes, final ordinary plans, validated constraints, learning/member/observation data, and unchanged mu3/global-goal plans; do not load arrays, repack, recompute prefixes, or repeat learning. schedule._arena_signature already consumes Arena, so fingerprint inputs means preserve its fields/order, not add a second fingerprint schema. CompositeOwnership distinguishes executed and absorbed factors from parameter-owner references. Equality plans appear in resolved ordinary collections but belong to specialized ownership; count their execution once. mu3 and goal-parameter plans are overlays, not additional executed factors. Validate declared ownership/exclusion/reference consistency using one-pass indexes; keep family-specific interpretation in specialized owners. Negative tests should remove or duplicate an execution owner, mismatch an exclusion/parameter reference, and duplicate a shared-parameter definition, while positive mixed-goal and mu3 fixtures protect valid shared references. Reuse all published maps/arrays/plans; assemble only new edge tags and interior tables. Source: bounded local inspection of compiler.py, specialized/contracts.py and __init__.py, resolution/contracts.py, compiled.py and schedule.py at339c46a; no new tests or benchmarks. Extra preparation agents were unavailable due to usage limits; do not treat them as reviewers of this handoff.

**2026-09-09T22:57:05Z**

Dependency companion from tk dep tree --full cff-ddsq, tk ready, tk blocked; prerequisite --> dependent:
[x] cff-ejib (#119)
  --> [in_progress: final gate] cff-bfr9
    --> [blocked: next] cff-ddsq
      --> [blocked] cff-edoz
        --> [blocked] cff-sf29 --> engine track cff-ws3c --> cff-ofv1 --> cff-cfqg
No parallel prerequisite within this compiler chain; unrelated ready work remains outside this epic. Start from the published bfr9 branch only after its gate and closure, following the normal base update.

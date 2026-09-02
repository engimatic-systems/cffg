---
id: cff-edoz
status: open
deps: [cff-ddsq]
links: [cff-209l, cff-ws3c]
created: 2026-08-13T04:26:16Z
type: epic
priority: 1
assignee: coopwilliams
parent: cff-cfqg
tags: [architecture, compiler, extensibility]
---
# Compiler track: Give planning and lowering typed owners

## Background in Plain English — WHY

At e488df3, `src/cffg/compiler.py` has 2,368 lines and `compile_template`
occupies lines 632-1936. It validates topology, assigns slots, classifies
registered and specialized factors, resolves arrays and learnables, and
constructs the final arena. Those responsibilities are too large for one
implementation context and make a local factor change cross the whole compiler.

This track first records the compiler contract and moves compiled values. It
then extracts a sequential chain with one named boundary per context: topology,
registered lowering, specialized lowering, data/learning resolution, and final
assembly. `cff-209l` is the cross-track same-arena foundation parented by the
program epic rather than this compiler-only track.

## Outcomes / Acceptance Criteria — WHAT

1. Direct dependency `cff-ddsq` is closed. Its chain closes `cff-ecw0`,
   `cff-544x`, `cff-8t4d`, `cff-ejib`, `cff-kfo7`, `cff-bfr9`, and the
   cross-track foundation `cff-209l` first.
2. Characterization pins the e488df3 compiled plans, addresses, hashes,
   schedules, and diagnostics before production code moves.
3. Immutable compiled values have one lower-level owner.
4. The sequential stage boundaries are:
   4.1. expanded validated template to `TopologyLayout`;
   4.2. topology to `RegisteredPlanDrafts`;
   4.3. topology plus exclusions to `SpecializedPlanDrafts`;
   4.4. plan drafts plus array sources to `ResolvedCompileData`; and
   4.5. complete stage values to immutable `Arena`.
5. Registered same-arena and explicit specialized plans remain distinct. No
   universal callback is introduced.
6. `compile_template` stays the public entry point, is below 250 physical
   lines, and sequences named stages. The public compiler orchestration module
   is below 1,000 lines and no extracted file exceeds 1,000 lines.
7. Dependency direction is one way from IR/kernel declarations to compiled
   values and compiler stages, then to runtime consumers.
8. The track preserves HMM, Gaussian, GFE, hybrid, recurrence, functional mu3,
   hard-constraint, declared-learning, AgentSpec, plate, external-kernel, and
   malformed-input compiler characterization.
9. Existing mathematical oracle tests stay with their factor families.
   Old/new observable characterization is permitted for refactor-only compiler
   surfaces. A newly discovered scientific gap becomes another ticket unless
   it demonstrates that the structural move is unsafe.
10. Compile-time hostile declarations stay loud. Sparse/vacuous/improper
    runtime values remain N/A; float32 and a 32-group fixture preserve layout.
11. No serialized meaning, factor behavior, objective, runtime status,
    learnable cadence, checkpoint schema, performance target, or public call
    form changes.
12. `docs/ARCHITECTURE.md` names every final compiler owner and typed boundary.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Follow the child order so each commit has one complete input/output seam.
- Use **New factor or rule family**, **New IR field**, and **Learnable or
  checkpoint feature** as the primary rows; each child states any runtime-row
  applicability.
- Preserve flooding/scheduled, direct/`vmap`, and checkpoint consumers on one
  compiled contract; compiler host/JIT duplication remains N/A.

## Future work

`cff-sf29` records the engine baseline after this track. Performance, resource
estimation, source recovery, and new-family work retain their own tickets.

## Definitions

- **Compiler stage:** One host-side transformation with named complete input
  and output values.
- **Plan draft:** Structural runtime plan information before arrays and final
  arena assembly.
- **One-way dependency:** Lower-level declarations and values never import
  compiler orchestration or runtime owners.

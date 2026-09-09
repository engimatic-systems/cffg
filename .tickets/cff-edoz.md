---
id: cff-edoz
status: closed
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

At the e488df3 authoring snapshot, `src/cffg/compiler.py` has 2,368 lines and
`compile_template` occupies lines 632-1936. Those locations are scope anchors,
not promised future line numbers. The function validates topology, assigns
slots, classifies registered and specialized factors, resolves arrays and
learnables, and constructs the final arena. Those responsibilities are too
large for one implementation context and make a local factor change cross the
whole compiler.

This track first records the compiler contract and moves compiled values. It
then extracts a sequence of owners: topology, specialized discovery/completion,
registered lowering, data/learning resolution, and final
assembly. `cff-209l` is the cross-track same-arena foundation parented by the
program epic rather than this compiler-only track.

## Outcomes / Acceptance Criteria — WHAT

1. Direct dependency `cff-ddsq` is closed. Its chain closes `cff-ecw0`,
   `cff-544x`, `cff-8t4d`, `cff-kfo7`, `cff-ejib`, `cff-bfr9`, and the
   cross-track foundation `cff-209l` first.
2. `cff-ecw0` pins compiled plans, addresses, hashes, schedules, and diagnostics
   at the exact commit that closes `cff-209l`. e488df3 remains the external-
   parity reference only for behavior that `cff-209l` promises not to change.
3. Immutable compiled values have one lower-level owner.
4. The sequential stage boundaries are:
   4.1. expanded core-validated template to `TopologyLayout`;
   4.2. topology to `SpecializedClaims`, including absorbed goals/priors and
        inferred parameter-edge semantic tags;
   4.3. topology plus claims to `RegisteredPlanDrafts` for unclaimed factors;
   4.4. claims plus registered membership to completed `SpecializedPlanDrafts`;
   4.5. plan requirements plus sources to `ResolvedCompileData`, including
        validated constraint payloads; and
   4.6. stage values to immutable `Arena`, with exactly-once execution and
        objective ownership checks.
5. Registered same-arena and explicit specialized plans remain distinct. No
   universal callback is introduced. Public orchestration alone owns accepted-
   semiring option checking, plate expansion, and the existing core `validate`
   call; topology carries declared facts without classifying plan meaning.
   Registered/specialized owners enforce their own semiring compatibility.
   Specialized discovery and completion belong to one owner. Extraction order
   is a landing sequence, not a requirement that the owner be called once;
   compatibility delegates supply ordinary membership until cff-ejib lands.
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

## Notes

**2026-09-09T23:42:34Z**

Compiler track complete. The dependency chain has closed compiler contracts (cff-ecw0), immutable compiled values (cff-544x), declared topology (cff-8t4d), specialized discovery/completion (cff-kfo7), registered lowering (cff-ejib), array/learning resolution (cff-bfr9), and final assembly (cff-ddsq), on the completed registration foundation. Shared learned-goal and declaration-conflict corrections were delivered separately before extraction. The public compiler is 138 lines; all extracted modules are below 1,000 lines. It sequences expanded/core-validated Template -> TopologyLayout -> SpecializedClaims -> RegisteredPlanDrafts -> completed SpecializedPlanDrafts -> ResolvedCompileData -> checked immutable Arena. No stage reads back through the public compiler facade. Final source 707c31b passed 1,813 default tests, 41 slow deselections and four existing warnings, in 1,016.282 seconds measured wall; runtime record 6364053. Fourteen complete arenas and immutable-array publication counts match the pre-assembly compiler. Existing model identities, addresses, schedules, family oracles, external kernels, learning and checkpoint consumers remain protected. Architecture records the delivered owners and change paths. The exact commit closing this ticket becomes the engine characterization baseline; cff-sf29 is next. Independent P2 cff-34gi records an inherited invalid-probability input gap without expanding this refactor.

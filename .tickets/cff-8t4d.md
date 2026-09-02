---
id: cff-8t4d
status: open
deps: [cff-544x]
links: [cff-edoz, cff-ws3c]
created: 2026-09-02T04:02:00Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-edoz
tags: [compiler, architecture, refactor]
---
# Extract typed compiler lowering stages

## Background in Plain English — WHY

After `cff-544x`, compiled values have a lower-level owner but one large
compiler function still validates topology, classifies factors, resolves
addresses and arrays, recognizes specialized plans, resolves learning, and
assembles the final arena. These responsibilities need explicit inputs and
outputs so a future family changes the smallest relevant stage.

This ticket extracts those stages without changing the model language or
runtime result. Registered same-arena factors and explicit specialized plans
remain different compiler paths.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-544x` is closed with compiled values behind a one-way boundary.
2. `compile_template` remains the public orchestration entry point and
   delegates to typed stages with these responsibilities:
   2.1. topology validation and message-slot layout;
   2.2. registered same-arena classification and lowering;
   2.3. explicit multi-arena, GFE, constraint, and evaluator lowering;
   2.4. learnable source and target resolution;
   2.5. parameter, observation, and derived-array resolution; and
   2.6. final immutable arena assembly.
3. Each stage accepts and returns a named typed value. Stages do not communicate
   through module globals, hidden mutation, or a partly initialized arena.
4. Factor and constraint meaning has one classification owner.
   4.1. Registered same-arena factors dispatch from kernel metadata.
   4.2. Each specialized plan class has one explicit lowering owner.
   4.3. Parameter and observation roles are interpreted once and passed by
        typed address records.
5. Existing compile diagnostics continue to name the model item and incompatible
   declaration that caused failure.
6. Every `cff-ecw0` characterization value remains unchanged, including
   ordering, addresses, schedule fingerprints, hashes, learnable plans, and
   evaluator identity.
7. The resulting compiler orchestration module has fewer than 1,000 physical
   lines. No new or extracted file exceeds 1,000 physical lines, and each file
   has one stated responsibility.
8. JSON and CFFGZ reconstruction use the same stage sequence. Plates and
   composition supply ordinary validated templates rather than separate
   lowerers.
9. Flooding and scheduled runtimes, direct and `vmap` execution, and
   checkpoint consumers receive the same final compiled values. No execution
   pair gains a second compiler path.
10. Exact-zero support, vacuous messages, improper beliefs, and runtime
    convergence are not compiler decisions. Float32 and the 32-group
    characterization fixture retain the values required by `cff-ecw0`.
11. No factor behavior, objective term, approximation, public API, serialized
    field, checkpoint schema, or performance claim changes.
12. Focused compiler and all affected integration tests pass. The
    dashboard-backed short suite passes after the final code change.
13. `docs/ARCHITECTURE.md` names the stage inputs, outputs, and dependency
    direction after the extraction lands.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Extract one responsibility per atomic commit and keep the suite focused after
  each move.
- Prefer descriptive stage modules over a general compiler-utilities module.
- Keep registered lowering data-driven. Keep specialized plans explicit
  instead of widening one callback with unused fields.
- Use the **New factor or rule family**, **New IR field**, and **Learnable or
  checkpoint feature** change-path rows as the preservation checklist.

## Future work

Closing this ticket closes the compiler track `cff-edoz`.
`cff-sf29` then characterizes the engine against the new compiled-value
boundary.

## Definitions

- **Assembly:** Construction of the complete immutable arena after every stage
  succeeds.
- **Classification:** Selecting a registered arena binding or an explicit
  specialized plan from validated declarations.
- **Stage value:** A named immutable input or output shared between compiler
  stages.

---
id: cff-ws3c
status: open
deps: [cff-3ola]
links: [cff-edoz, cff-3kdk, cff-cj5h, cff-wbhu]
created: 2026-08-13T04:26:16Z
type: epic
priority: 1
assignee: coopwilliams
parent: cff-cfqg
tags: [architecture, engine, extensibility]
---
# Engine track: Give execution, schedules, ledger, and readouts focused owners

## Background in Plain English — WHY

At commit e488df3, `src/cffg/engine.py` has 3,973 lines. It defines public
state and status types, data binding, ordinary and GFE execution, support
checks, schedule builders, objective accounting, and public readouts. It also
contains 44 references to factor or group type selection. Several paths gather
the same arena inputs and parameters separately. That duplication makes a local
factor change hard to review and can let execution, diagnostics, and the
free-energy ledger disagree.

This track preserves the engine behavior recorded at commit e488df3 while
assigning those responsibilities to focused modules. It starts after the
compiler track supplies typed compiled values. Four context-sized child tickets
record the post-compiler contract, extract execution and schedules, extract the
ledger, and finally extract readouts while leaving the public `Engine` as
orchestration.

## Outcomes / Acceptance Criteria — WHAT

1. The direct dependency `cff-3ola` is closed because its acceptance criteria
   are satisfied. Its chain closes `cff-gl89`, `cff-ny7f`, `cff-sf29`,
   and `cff-edoz` first.
2. Characterization evidence pins messages, beliefs, statuses, schedules,
   diagnostics, objective values, and readouts at its recorded baseline before
   engine code moves.
3. Per-arena and specialized execution modules own input gathering, parameter
   selection, rule calls, and message writes for their named plans.
4. One schedule-execution owner builds flooding, staged, scan, exact-pass,
   support-check, and propagation functions from typed compiled values.
5. One objective owner assembles factor terms, parameter terms, constraint
   terms, GFE substitutions, and edge-entropy corrections.
6. Focused readout owners consume typed arena views without reimplementing
   message rules or objective terms.
7. Public `Engine`, `State`, `RunResult`, `RunStatus`, and
   `InferenceError` retain their documented imports, call forms, and meanings.
8. The public engine module becomes readable orchestration below 1,000 physical
   lines. No extracted source file exceeds the repository's 1,000-line limit.
9. Imports remain acyclic. Compiled values and mathematical rules sit below
   execution, schedule, objective, and readout owners; public orchestration
   sits above them.
10. The track preserves template and structure identity, parameter addresses,
    state shape, checkpoints, messages, beliefs, BFE and GFE meaning,
    convergence statuses, and each approximation boundary recorded by the
    baseline documentation and tests.
11. Literal zero support, vacuous messages, improper beliefs, non-finite
    updates, float32, and uncomfortable-scale fixtures keep their documented
    values or loud statuses across flooding and scheduled paths.
12. This track introduces no runtime-speed threshold, new diagnostic, new
    factor family, scientific claim, or public-release promise.
13. `docs/ARCHITECTURE.md` names the final engine owners and their dependency
    direction after the extraction lands.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Follow the child order and land one owner per atomic code commit.
- Use the **Runtime numerical or status invariant** row as the primary
  preservation checklist. Also use **New factor or rule family** and
  **Learnable or checkpoint feature** for affected consumers.
- Keep flooding and scheduled builders on the same family execution
  primitives. Keep direct and one-instance batched behavior on the same typed
  values.
- Preserve compatibility imports until direct consumers migrate. Avoid broad
  helper modules that have no single responsibility.

## Future work

`cff-ofv1` verifies the bounded extension paths after this track closes.
`cff-3kdk`, `cff-cj5h`, and `cff-wbhu` remain independent consumers of
the final readout, objective, and schedule seams. They are not prerequisites
and this track does not claim their performance or diagnostic outcomes.

## Definitions

- **Execution owner:** The module that gathers inputs, calls one rule family,
  and writes its messages.
- **Objective owner:** The module that computes and labels BFE or GFE
  contributions.
- **Orchestration:** Selection and sequencing of existing contracts without
  owning their local mathematics.
- **Readout owner:** The module that converts a typed arena view into a
  documented posterior or diagnostic value.
- **Schedule owner:** The module that builds the ordered execution functions
  selected by a compiled schedule.
- **Typed arena view:** The focused immutable subset of compiled metadata that
  one runtime owner needs.

---
id: cff-ws3c
status: open
deps: [cff-edoz, cff-wbhu, cff-cj5h, cff-3kdk]
links: []
created: 2026-08-13T04:26:16Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-cfqg
tags: [trustworthiness, engine, maintainability]
---
# Extract engine execution and ledger owners after characterization

## Background in Plain English

`src/cffg/engine.py` is 3,816 lines. It combines public state and status types,
agent-facing orchestration, data binding, readouts, synchronous family sweeps,
scheduled execution builders, support checks, exact certification, and the
free-energy ledger. Several of those paths independently gather the same arena
inputs and parameters. That duplication makes central execution changes hard
to review and makes a new family touch more code than its mathematics requires.

Three tickets must land first. `cff-wbhu` establishes the compiled ordinary
loop and status carry, `cff-cj5h` establishes the public objective breakdown,
and `cff-3kdk` establishes batched readouts. `cff-edoz` supplies the typed
compiled-plan boundary. This ticket then extracts stable owners without
changing inference or objective meaning. Characterization evidence lands
before extraction.

## Outcomes / Acceptance Criteria

1. The four dependency tickets are closed with their public execution,
   diagnostic, readout, and compiled-plan contracts satisfied.
2. Before extraction, characterization tests pin representative discrete,
   Gaussian, EF, hybrid-mixture, recurrence, precision, hard-constraint, GFE,
   transition-GFE, and `mu3` behavior.
   2.1. The artifacts pin messages, beliefs, statuses, iteration counts,
   residuals, schedule fingerprints, objective totals and breakdowns, and
   public readouts where defined.
   2.2. The artifacts cover host and JIT, flooding and scheduled, direct and
   one-instance `vmap`, and current and legacy checkpoint consumers wherever
   both paths apply.
3. Public `Engine`, `State`, `RunResult`, `RunStatus`, and `InferenceError`
   retain their documented import paths and behavior.
4. Engine orchestration owns state initialization, data binding, selection of
   the compiled execution plan, status-bearing public calls, and conversion of
   internal results to public readouts.
5. Per-arena execution modules own message gathering, parameter selection,
   factor-rule calls, and message writes for their named arena families.
   5.1. Flooding and scheduled builders call the same per-arena execution
   contract.
   5.2. A factor's parameter or observation interpretation has one owner.
   5.3. An execution module does not import the public `Engine` class.
6. One focused schedule-execution owner builds flooding, batch-stage,
   scan-stage, exact-pass, support-check, and compiled-loop functions from the
   typed arena contract.
7. One focused objective owner assembles node terms, shared parameter terms,
   constraint terms, and edge-entropy corrections.
   7.1. `Engine.bfe` and `Engine.fe_breakdown` use the same local-term
   implementations.
   7.2. The objective owner does not perform message updates or public action
   selection.
8. Batched readout owners consume typed arena views and do not reimplement
   message rules or objective terms.
9. Imports remain acyclic. Compiled-plan values and mathematical rule modules
   sit below execution and objective owners; the public engine orchestration
   sits above them.
10. Every artifact from criterion 2 retains its pre-extraction value or
    dtype-derived tolerance, including literal zero support, vacuous messages,
    improper-belief failure, float32, and uncomfortable-scale fixtures.
11. Template JSON, structure hashes, bundles, parameter addresses, checkpoint
    schemas, and public method signatures are unchanged.
12. `engine.py` becomes a readable orchestration module below 1,000 physical
    nonblank, noncomment lines. No extracted source file exceeds the project's
    1,000-line limit.
13. Each extraction is a separate atomic commit after the characterization
    commit and leaves the focused suite passing.
14. `docs/ARCHITECTURE.md` updates the engine module map, responsibility
    boundaries, and duplicated execution pairs after the move.

## Methodological Suggestions

These suggestions are optional. An implementer may choose another sound method
that satisfies the numbered outcomes with equivalent evidence.

- Extract stable data and gather/write primitives before moving builders that
  close over an `Arena`.
- Move one responsibility at a time: family execution, schedule execution,
  objective assembly, then readouts. Keep compatibility imports until direct
  consumers migrate.
- Prefer modules named for their mathematical or runtime owner over a general
  helpers module.
- Use the **Runtime numerical or status invariant** change path as the primary
  preservation checklist. Also walk the factor-family and checkpoint rows
  because this refactor touches their derived execution surfaces.

## Future work

No cosmetic file-splitting campaign follows this ticket. A later extraction
requires a new repeated-change-path problem and a one-way contract. Nonlinear
and neural factors remain blocked on `cff-cfqg`; they do not land during this
refactor.

## Definitions

- **Execution owner:** The module that gathers one arena family's inputs, calls
  its registered rule, and writes the resulting messages.
- **Objective owner:** The module that computes and labels the node, shared,
  constraint, and edge terms in BFE or GFE.
- **Orchestration:** Selection and sequencing of existing execution contracts,
  including status handling, without owning their local mathematics.
- **Typed arena view:** A focused immutable subset of compiled arena metadata
  needed by one execution, objective, or readout owner.

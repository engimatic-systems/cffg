---
id: cff-edoz
status: open
deps: [cff-209l]
links: []
created: 2026-08-13T04:26:16Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-cfqg
tags: [trustworthiness, compiler, maintainability]
---
# Extract compiler planning and lowering behind typed internal contracts

## Background in Plain English

`src/cffg/compiler.py` is 2,316 lines. It defines every compiled-plan type and
the `Arena`, then performs topology layout, factor classification, per-arena
lowering, composite recognition, learnable resolution, parameter packing, and
final assembly in one module. This makes a change to one factor family look
like a change to the whole compiler.

Ticket `cff-209l` first moves ordinary discrete and Gaussian execution onto the
registered kernel contract. This ticket then extracts compiler responsibilities
along the resulting typed boundaries. It is a behavior-preserving
maintainability change. Characterization evidence lands before code moves.

## Outcomes / Acceptance Criteria

1. `cff-209l` is closed with the final discrete, Gaussian, and EF kernel
   metadata available as the classification authority.
2. Before extraction, characterization tests pin the compiled HMM, Kalman,
   T-maze, hybrid-mixture, recurrence, hard-constraint, and one external-kernel
   fixtures.
   2.1. The tests pin plan kinds, group membership, member order, role order,
   slot offsets, parameter addresses, schedule fingerprint, structure hash,
   and learnable source and target addresses where applicable.
   2.2. The tests pin successful and failing compile diagnostics for one
   unsupported factor, constraint, role, parameter, and external-kernel case.
3. Immutable compiled-plan and `Arena` types live in a focused internal module
   that imports neither the compiler orchestration function nor the inference
   engine.
4. `compile_template` remains the public orchestration entry point and delegates
   to typed stages with these single responsibilities:
   4.1. topology validation and slot layout;
   4.2. registered factor classification and per-arena lowering;
   4.3. specialized constraint and GFE composite lowering;
   4.4. learnable resolution;
   4.5. parameter and observation-array resolution; and
   4.6. final immutable arena assembly.
5. Each stage accepts and returns a named typed value. No stage communicates
   through hidden module globals or a partly initialized `Arena`.
6. Dependency direction is one way: core IR and kernels feed compiler stages;
   compiler stages feed compiled-plan values; engine and schedule consumers
   depend on compiled-plan values and do not import compiler implementation
   helpers.
7. Existing public imports of `compile_template`, `CompileError`, and supported
   compiled values continue to work or receive a documented compatibility
   re-export.
8. The extraction preserves every characterization value from criterion 2.
9. HMM, Kalman, T-maze, hybrid, recurrence, constraint, plate, learning,
   bundle, external-kernel, estimator, and schedule tests retain their
   pre-extraction numerical and serialization results.
10. Host and JIT, flooding and scheduled, single-instance and `vmap`, and
    current and legacy checkpoint paths receive no new compiler interpretation.
    Each pair is checked against the characterization artifacts or marked
    inapplicable with a concrete reason.
11. `compiler.py` becomes a readable orchestration module below 1,000 physical
    nonblank, noncomment lines. No extracted source file exceeds the project's
    1,000-line limit.
12. `docs/ARCHITECTURE.md` updates the compiler module map and names the owner of
    each lowering stage.

## Methodological Suggestions

These suggestions are optional. An implementer may choose another sound method
that satisfies the numbered outcomes with equivalent evidence.

- Make the characterization commit stand alone and pass before the first move.
- Extract plan dataclasses and `Arena` first because that creates the clearest
  one-way dependency. Move lowering stages only after their inputs and outputs
  are explicit.
- Prefer small internal modules named for responsibilities, not a generic
  `compiler_utils.py` collection.
- Use the **New factor or rule family**, **New IR field**, and **Learnable or
  checkpoint feature** change paths only to prove preservation. This ticket
  adds no new family, field, or learning behavior.

## Future work

`cff-ws3c` consumes the extracted compiled-plan contract when it separates
engine execution, ledger, and readout owners. New factor families remain
blocked on `cff-cfqg` until both modularization tickets and the other trust
dependencies close.

## Definitions

- **Characterization test:** A test that records existing observable behavior
  before a refactor so the move cannot silently change it.
- **Compiled plan:** Immutable metadata that tells the runtime where and how to
  execute one class of factors or constraints.
- **Lowering:** Conversion from validated serialized model declarations into
  executable arena plans and addresses.
- **One-way dependency:** A module relationship with no reverse import or
  circular ownership.

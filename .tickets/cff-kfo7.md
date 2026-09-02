---
id: cff-kfo7
status: open
deps: [cff-ejib]
links: [cff-bfr9]
created: 2026-09-02T04:17:42Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-edoz
tags: [compiler, architecture, refactor]
---
# Extract explicit specialized compiler lowering

## Background in Plain English — WHY

At e488df3, `src/cffg/compiler.py:898-1479` recognizes parameter-edge VMP,
P-substitution/GFE composites, hybrid mixtures, functional evaluators,
recurrence, and Gaussian-precision relationships; lines 1743-1818 lower hard
constraints. These constructs cross arenas or change structural/objective
meaning and cannot honestly use one ordinary same-arena callback.

This ticket gives those plan classes explicit lowering owners behind one typed
specialized-stage result. It does not change their mathematics or invent a
universal solver interface.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-ejib` is closed with registered same-arena lowering passing.
2. A specialized-lowering stage accepts `Template`, `TopologyLayout`, and
   registered-stage exclusions and returns immutable `SpecializedPlanDrafts`
   plus typed `ArrayRequirement` values.
3. Named lowerers cover the e488df3 plan classes: parameter-edge Dirichlet VMP,
   mixture/hybrid, recurrence, Gaussian precision, P-substitution and GFE,
   hard constraints, and functional evaluator metadata.
4. Each plan class has one classification/lowering owner. Shared orchestration
   invokes those named lowerers without passing unrelated arena state through a
   universal callback.
5. The affected rows are **New factor or rule family**, **New IR field**, and
   **Learnable or checkpoint feature**. **Runtime numerical or status
   invariant** is N/A except that emitted plan/status metadata must stay
   identical for runtime consumers.
6. Flooding/scheduled, direct/`vmap`, and latest-schema/supported-legacy
   consumers receive the same specialized drafts. Host/JIT is N/A because the
   stage is host-side.
7. Sparse/vacuous/improper messages are runtime N/A. Float32 preserves plan
   identity and address order. The uncomfortable compiler fixture exercises at
   least one specialized plan in a 32-group template.
8. Existing hybrid, GFE, recurrence, mu3, constraint, learning, evaluator, and
   compile-failure tests protect old/new observable behavior. Existing
   mathematical authorities remain in their family tests; no new recovery or
   scientific benchmark is required for this move.
9. Unsupported substituted roles, constraints, evaluator metadata, shapes,
   and plan combinations retain a loud factor-specific `CompileError`.
10. The specialized stage changes no approximation label, factor behavior,
    objective term, schema, public API, or supported combination.
11. Each named lowerer and its stage module is below 1,000 physical lines. The
    ticket fits one context because it moves structural classification only;
    arrays and final assembly remain separate.
12. Focused specialized-family/compiler tests and the final short suite pass.
13. `docs/ARCHITECTURE.md` names every specialized plan owner in the same
    commit as the extraction.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Return a tagged collection of explicit draft types rather than one object
  with irrelevant optional fields.
- Keep source-specific evaluator identity and GFE/P-substitution details on
  their plan drafts.
- Emit typed array requirements for the next stage instead of reading blobs.

## Future work

`cff-bfr9` resolves arrays and learning addresses for both registered and
specialized drafts.

## Definitions

- **Specialized plan:** A plan whose multi-arena, structural, evaluator, or
  GFE behavior cannot use the ordinary same-arena contract.
- **SpecializedPlanDrafts:** The typed collection of explicit specialized plan
  values before array resolution.
- **Universal callback:** One callable forced to accept unrelated state for
  every plan class.

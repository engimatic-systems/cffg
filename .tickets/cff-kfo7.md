---
id: cff-kfo7
status: open
deps: [cff-8t4d]
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

At authoring snapshot e488df3, `src/cffg/compiler.py:898-1479` is the source-
region anchor for parameter-edge VMP, P-substitution/GFE composites, hybrid
mixtures, functional evaluators, recurrence, and Gaussian-precision
relationships; lines 1743-1818 anchor hard constraints. The implementation
starts from the exact `cff-ecw0` baseline, and those old line ranges are not
promised future maps. These constructs cross arenas or change structural/
objective meaning and cannot use one ordinary same-arena callback.

This ticket gives those plan classes explicit lowering owners behind one typed
specialized-stage result. It does not change their mathematics or invent a
universal solver interface.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-8t4d` is closed with raw topology and logical slots characterized.
2. The specialized owner exposes two ordered operations:
   2.1. Discovery accepts `Template`, `TopologyLayout`, and semiring and returns
        immutable `SpecializedClaims`: plan kind, owned factors, absorbed
        goals/priors, and inferred parameter-edge shape/axis tags.
   2.2. Completion accepts those claims plus ordinary registered membership
        and returns `SpecializedPlanDrafts` and typed `ArrayRequirement`
        values. It can construct recurrence components without classifying
        factors again. Until cff-ejib lands, a compatibility delegate supplies
        the ordinary membership from the existing compiler path.
3. Named lowerers cover the plan classes present at the `cff-ecw0` baseline:
   parameter-edge Dirichlet VMP,
   mixture/hybrid, recurrence, Gaussian precision, P-substitution and GFE,
   hard constraints, and functional evaluator metadata.
   Structural equality is also owned here: discovery chooses its representation
   from declared edge domains, and lowering may retain the existing runtime
   group shape. Edge DELTA applicability and semantic validation belong here;
   assembly receives validated delta-pair facts without choosing their meaning.
4. Each plan class has one classification/lowering owner. Shared orchestration
   invokes those named lowerers without passing unrelated arena state through a
   universal callback.
   4.1. Neither topology, registered lowering, data resolution, nor assembly
        chooses specialized plan meaning; this ticket owns those choices.
   4.2. Claims partition factor ownership before ordinary grouping. Conflicting
        claims fail with the affected factor or absorbed member. Shared
        parameter terms have one declared owner rather than one owner per user.
   4.3. Drafts describe blob-derived hard-constraint k/table payloads through
        declared requirements and family-owned validators/transforms. They do
        not pretend the numeric payload exists before data resolution.
   4.4. A compact mapping connects each composite's original factors and
        constraints to its runtime plan, local objective terms, absorbed
        factor/edge exclusions, and shared-parameter owner. Existing GFE,
        hybrid, and multi-slice mu3 authority tests check the mapping.
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

`cff-ejib` extracts ordinary registered lowering between discovery and
completion. `cff-bfr9` then resolves arrays, constraint payloads, and learning
addresses from both sets of declared requirements.

## Definitions

- **Specialized plan:** A plan whose multi-arena, structural, evaluator, or
  GFE behavior cannot use the ordinary same-arena contract.
- **SpecializedPlanDrafts:** The typed collection of explicit specialized plan
  values before array resolution.
- **SpecializedClaims:** Discovered ownership and semantic edge tags used to
  exclude composite members from ordinary grouping.
- **Absorbed member:** A goal or prior represented inside a composite rather
  than executed and charged to the objective again as an ordinary factor.
- **Universal callback:** One callable forced to accept unrelated state for
  every plan class.

## Notes

**2026-09-09T03:29:39Z**

PR #108 research — question: can ordinary lowering precede specialized ownership? Answer: no; GFE absorbs ordinary-looking goals/priors before ordinary grouping, and recurrence completion later needs ordinary BP membership. One specialized owner now discovers claims first and completes drafts after registered membership. Raw slots remain topology-owned; semantic parameter-edge tags and blob-derived constraint payloads have explicit later owners. Sources at 7885bf5: compiler.py:760-823, 877-947, 1389-1392, 1547-1569, 1776-1815, 2080-2130. Confidence: high from direct code inspection. Assembly checks execution ownership and objective exclusions without rediscovering semantics.

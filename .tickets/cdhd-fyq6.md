---
id: cdhd-fyq6
status: in_progress
deps: [cdd-ziiw]
links: []
created: 2026-09-15T07:10:33Z
type: bug
priority: 2
assignee: coopwilliams
parent: cdpf-gzix
tags: [hybrid, compiler]
---
# Preserve admitted one-label unary hybrid plan metadata

## Background in Plain English — WHY

The declaration layer already permits a unary hybrid component with one categorical label. A reproduced template with z cardinality 1, x dimension 1, a gaussian_mixture unary region, support (0,) and categorical/Gaussian GLUE adapters passes core.validate. The new compiled plan rejects the same component because it requires cardinality at least 2. This restriction comes from the separate recurrent solver's layout and does not belong in a record that promises to preserve admitted declaration metadata.

This bounded repair restores consistency with existing declaration admission. It does not expand public runtime support or change the numerical solver, its qualified layouts, mathematics or thresholds. It blocks closure of cdpf-gzix, where the mismatch was reproduced before final review.

## Outcomes / Acceptance Criteria — WHAT

1. HybridComponentPlan accepts positive separator cardinality, including a valid unary one-label component, while preserving nonempty in-range unique support and positive state-dimension checks. Zero, negative and nonintegral cardinalities remain named errors.
2. A regression reproduces the actual core-to-draft boundary: the existing admitted unary K=1 declaration validates and lowers through the cdpf-gzix stage APIs to a consistent immutable claim/draft, retaining support (0,), adapter slots and parameter requirements. This is metadata-only evidence, with no array loading or inference.
3. Existing sparse K>=2 plans and ordinary models retain their behavior. Compiler/bundle rejection and shared D12 execution guards remain active. Numerical Coordinate.from_sites and every frozen solver/source/specification/evidence artifact remain unchanged; accepting metadata does not claim that a numerical layout is qualified.
4. Before implementation, apply the Derived component coordinate local plan-identity boundary and its declaration-to-compiled/draft pair. Serialization schema, actual array/support resolution, State/ledger/readouts, host/JIT/single-vmap numerical arithmetic and learned persistence are N/A. The existing serialized scope is authoritative; do not tighten core validation to hide the mismatch.
5. A fresh implementation worker makes the small repair and focused checks, commits and pauses. Root obtains independent review after the pause and runs one final short suite after all relevant cdpf-gzix/repair edits, under the unchanged 1200-second cap. Close this ticket and cdpf-gzix only after that combined final gate passes.

## Methodological Suggestions — HOW

Change only the unintended minimum in compiled_hybrid.py and add a meaningful direct-plan/real-declaration regression using existing metadata fixtures. Verify zero/invalid cardinality remains rejected. Keep numerical source files untouched. Another sound method preserving all outcomes is welcome.

Hostile inputs: K=1 with exact singleton support is the mechanism; empty/invalid/out-of-range support stays invalid. Improper cavities, convergence and FE arithmetic are N/A because no solver runs. Existing explicit float32/float64 inventory and larger shape-only/structural tests remain applicable and unchanged.

## Future work

cdpf-gzix retains graph discovery review, closure and publication. cff-7sw5/IDHK/F3JZ must keep admitted graph metadata distinct from the qualified chain solver's runtime layout; any necessary numerical adapter or additional qualification requires its own bounded work. No numerical execution is enabled by this repair.

## Definitions

- Unary region: one declared factor acting on a single hybrid separator.
- Cardinality K: total categorical label count; K=1 describes a deterministic label.
- Support: the exact declared set of labels that can carry probability.
- Claim/draft: immutable graph ownership and parameter-requirement intentions before actual data resolution.
- D12: the accepted hybrid-regime-ec-v1 approximation; its runtime implementation remains guarded.



## Notes

**2026-09-15T07:27:12Z**

Implementation handoff: applicable ARCHITECTURE change-path row is Derived component coordinate, limited to the local plan-identity boundary. Affected pair: admitted core declaration -> compiled plan and specialized claim/draft. Serialization schema, array/support resolution, assembly, State/ledger/readouts, host/JIT, synchronous/scheduled, single/vmap numerical execution, current/legacy checkpoints, and learned persistence are N/A. Preserve existing core admission and all execution guards. Hostile-input coverage is unary K=1/support (0,), plus invalid cardinality and support rejection; cavity/FE/convergence arithmetic is N/A. Focused metadata checks only in this worker; root owns independent review, the combined final short-suite gate and closure.

**2026-09-15T07:29:46Z**

Implemented the bounded metadata repair: HybridComponentPlan now accepts positive separator cardinality, including K=1. No core admission, compiler/bundle/runtime guard, numerical Coordinate.from_sites, numerical source, specification, qualification or evidence artifact changed. This restores the existing documented declaration-to-plan contract; it makes no runtime qualification claim.

Regression evidence: an actual gaussian_mixture unary template with z cardinality 1, x dimension 1, separator support (0,), and categorical/Gaussian GLUE adapters passes core.validate and now produces consistent immutable specialized claims/drafts. Assertions cover region ownership, ordinary gather/scatter slots, unresolved mean/covariance source shapes, destination keys, member index and transforms, disabled writes/statistics, and unavailable coordinate/scratch requirements. The new regression failed before the repair specifically at the compiled minimum-cardinality check. Existing no-resolution/no-execution and immutable-snapshot checks now also cover this unary fixture. Direct plan checks retain named errors for zero/negative/nonintegral/bool cardinality or dimension and empty/duplicate/out-of-range/negative/nonintegral/bool support.

Final focused validation after the source edit:
- uv run pytest -q tests/test_hybrid_discovery.py tests/test_hybrid_plans.py -k 'not ordinary_models_keep_execution_and_inventory_paths' -> 77 passed, 2 deselected in 0.31s. The two excluded tests execute ordinary numerical models, outside this worker's metadata-only authorization.
- uv run pytest -q tests/test_hybrid_declarations.py -k 'compile_and_bundle_guards_precede_lowering_and_destination_changes or bundle_load_rejects_a_declared_plan_before_array_resolution' -> 2 passed, 33 deselected in 0.02s.
- git diff --check passed; tk dep cycle reports no dependency cycles.

Handoff: implementation is ready for an atomic cdhd-fyq6 commit and a clean pause. Root owns independent review, one combined final short-suite run after all relevant edits under the unchanged 1200-second cap, ticket closure and publication. This ticket remains in_progress pending that gate. No full suite, array loading, numerical program, qualification program or recovery fit was run by this worker; all three recovery fits remain unused. Downstream numerical source/layout generalization remains separate work.

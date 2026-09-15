---
id: cdhd-fyq6
status: open
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



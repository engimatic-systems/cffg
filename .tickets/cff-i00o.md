---
id: cff-i00o
status: in_progress
deps: [cff-8xve]
links: []
created: 2026-08-17T20:46:52Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, statistics, preregistration]
---
# Specify dual-family protocols and source-recovery claims

## Background in Plain English

The final evidence has two jobs. Exact source-homogeneous NASCAR tests the faithful classic benchmark. The cff-8xve extension tests source-dependent switching that the general rSLDS formulation permits but the original demonstration does not exercise. The protocol and statistics must keep these claims separate. Agreement between two learners is not enough to show that either learner recovered the generating behavior.

This ticket defines the family-aware schema, stochastic coordinates, estimands, absolute adequacy gates, multiplicity rule, and claim language before the dual runner or final freeze is built. It extends the calibrated two-tier CFFG-versus-paper-adapter design without generating final data.

## Outcomes / Acceptance Criteria

1. The protocol schema uses exactly `canonical_homogeneous` and `source_heterogeneous` as explicit family coordinates in generator descriptors, scales, splits, seeds, data identities, stochastic streams, margins, and claims.
2. Family-qualified stochastic stream coordinates cannot collide across dataset family, data seed, tier, replicate, chain, control, initialization, evaluation, or rollout purpose.
3. The classic family retains the six paired non-inferiority effects from three primary metrics in two tiers and adds absolute CFFG adequacy gates for functional transition error and bounded periodic rollout. Both paired and absolute requirements must pass before the report calls the classic benchmark successful.
4. The heterogeneous family retains the same six paired effects and adds absolute functional recovery requirements in both tiers: a simultaneous bound on the worst source's transition-probability error over one frozen balanced source-by-state grid and a simultaneous bound on source-contrast error against truth.
5. The balanced evaluation grid exercises every source at the same states near every gate. Its identity and construction are frozen before final execution, and label/affine alignment occurs before functional transition scoring.
6. Raw recurrence coefficient error remains secondary unless a separately reviewed canonical mapping removes label, decision-list-order, and affine latent freedoms. Two equally wrong learners cannot pass an absolute recovery claim through paired agreement.
7. Fixed-sequence gatekeeping tests the classic family first at family alpha 0.05 with Bonferroni coverage across its registered effects. Only a passed classic family unlocks the heterogeneous claim family, which uses its own Bonferroni coverage across every paired and absolute primary effect.
8. A classic failure does not erase heterogeneous descriptive results, and a heterogeneous failure does not erase a passed classic result. The combined “classic plus broader capability” statement requires both families to pass in sequence.
9. Wrong generator provenance, incomplete coverage eligibility, missing coordinates, non-ready arms, invalid metrics, convergence failure, mixing failure, or incomplete raw evidence produces an explicit family-level not-ready result. A complete interval that misses its margin produces a failed claim rather than not-ready.
10. Every numerical margin has a calibration rule and domain cap that can be fixed by cff-s6rs without inspecting final seeds. Schema and oracle tests cover family identity, stream uniqueness, multiplicity, claim gatekeeping, source contrasts, and each failure state.
11. No registered final seed is generated or fit by this ticket.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound method that satisfies the outcomes.

- Nest family-specific protocol sections under one versioned schema instead of encoding the family into integer seeds.
- Prefer functional transition surfaces to raw gate coefficients because the latent space and regime labels require alignment.
- Extend the existing simultaneous interval helper by supplying the exact effect dictionary for each fixed-sequence family.
- Keep protocol construction pure so cff-s6rs can derive and freeze it from calibration artifacts.

## Future work

cff-9o39 applies this schema to family-aware runners, controls, aggregation, and checkpoints. cff-s6rs performs calibration-only measurement. cff-huxt writes the immutable replacement freeze. cff-oyie owns all final seeds.

## Definitions

- **Absolute adequacy gate:** A truth-based ceiling or floor that both implementations could fail even when they agree with each other.
- **Balanced source-by-state grid:** One frozen set of continuous states evaluated under every source regime so source effects cannot be confused with different visited states.
- **Fixed-sequence gatekeeping:** A multiplicity rule in which the second claim family is tested only after the first family passes.
- **Source contrast:** The difference between transition probabilities under two current source regimes at the same continuous state.

## Notes

### 2026-08-17 — implementation handoff

`docs/ARCHITECTURE.md` change-path row: N/A. This ticket changes the sibling validation experiment's protocol schema and statistical evaluation contract, not a CFFG engine owner. CFFG's consumer-facing evidence documentation is deferred to the final evidence ticket after results exist.

Duplicated execution surfaces checked before implementation:

- family descriptors, scales, seed sets, data identities, and stochastic stream preimages;
- classic paired effects and absolute adequacy outcomes;
- heterogeneous paired effects, shared-grid worst-source error, and source-contrast recovery;
- calibration validation and final preregistration validation;
- fixed-sequence family decisions, partial-family failures, and descriptive-result retention;
- runner, checkpoints, raw rows, and reports: explicitly delegated to cff-9o39 after the pure protocol contract lands.

Hostile-input decisions:

- Missing, non-finite, wrongly shaped, unnormalized, or source-collapsed transition tensors are invalid metrics and cannot enter a claim interval.
- Sparse, one-sided, or rank-deficient heterogeneous generator coverage makes that family ineligible before fitting.
- Float32 functional scoring uses dtype-scaled normalization checks and must remain finite; final execution remains registered as float64.
- The balanced grid evaluates every source at every frozen state, so a naturally unoccupied source/state pair cannot disappear from the worst-source outcome.
- Empty effect families, incomplete seeds, failed arms, and absent absolute outcomes produce explicit not-ready decisions rather than vacuous success.

The statistical design is fixed from cff-0l31 calibration rules, cff-3vhn reporting contracts, cff-ah4t coverage findings, and cff-8xve's calibration-only generator artifact. No final seed is generated or fit here.

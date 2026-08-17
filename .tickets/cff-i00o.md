---
id: cff-i00o
status: open
deps: [cff-8xve]
links: []
created: 2026-08-17T20:46:52Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, statistics, preregistration]
---
# Integrate and freeze the dual-family rSLDS comparison

## Background in Plain English

The final evidence now has two different jobs. Exact source-homogeneous NASCAR tests faithful reproduction of the classic benchmark. The cff-8xve extension tests source-dependent switching that the general rSLDS formulation permits but the original demonstration does not exercise. The runner, statistics, checkpoints, and report must keep those families separate so success in one cannot conceal failure in the other.

This ticket integrates both families, runs calibration-only comparisons, fixes each claim and failure rule, and writes one replacement preregistration before cff-oyie sees a final seed. It preserves the calibrated two-tier CFFG-versus-paper-adapter design, learned SLDS controls, secondary outcomes, and strict diagnostics already built by cff-0l31 and cff-3vhn.

## Outcomes / Acceptance Criteria

1. Protocols, stochastic streams, data identities, raw rows, checkpoints, manifests, and aggregates use an explicit dataset-family coordinate with exactly `canonical_homogeneous` and `source_heterogeneous` values.
2. The canonical family uses the unchanged classic generator; the heterogeneous family uses the single calibration-frozen cff-8xve generator. Neither fitter receives generating states, regimes, or parameters before fitting finishes.
3. Calibration uses only the two registered calibration seeds, keeps final seeds untouched, and measures both families, both primary tiers, four independent reference chains, and both learned SLDS controls.
4. The preregistration states separate claim text for classic benchmark empirical non-inferiority and broader source-specific capability evidence. It explicitly excludes authors-code, inference-algorithm, message-equation, free-energy, and universal model-family parity.
5. Every primary outcome, recovery requirement, multiplicity rule, margin, diagnostic, budget, stopping rule, and failure policy is fixed per family before final execution.
6. A failed, missing, non-converged, budget-exhausted, diagnostic-failed, or unscorable arm remains in its family and prevents that family's conclusion. One family's result cannot substitute for the other.
7. Both families retain paired held-out predictive likelihood, affine-aligned latent RMSE, label-aligned regime accuracy, all registered secondary outcomes, the same-`K` SLDS control, and the closest-free-scalar-budget SLDS control.
8. The heterogeneous family registers direct source-conditioned transition recovery on realized support and a coverage-aware gate recovery outcome that cannot pass from one-sided or rank-deficient source/gate data.
9. Final execution checkpoints one family/seed coordinate at a time, hashes each raw attempt, preserves interrupted attempts, and resumes only missing coordinates from the unchanged preregistration.
10. Direct calibration-only measurements select and record one horizon and train/held-out split for each family with a complete-run projection and restart plan.
11. A new immutable preregistration and manifest preserve the cff-0l31 freeze as superseded provenance, name cff-oyie as the only final-execution owner, and verify every calibration, generator, mapping, adapter, environment, and source artifact hash.
12. README states the claim ladder and limits in plain language. No final seed is generated or fit by this ticket.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound method that satisfies the outcomes.

- Represent family and seed as separate coordinates rather than encoding the family into an integer seed.
- Calibrate family-specific margins when repeatability differs, but control multiplicity across every primary claim used in the same public conclusion.
- Recompute each family aggregate from raw rows and compute a joint claim only from the two complete family results.
- Use a new preregistration output directory so the first freeze remains byte-for-byte available.

## Future work

cff-oyie executes the immutable final protocol and reports every family/seed result. cff-nkbj audits the resulting claim ladder and closes only if the supported language matches the artifacts. Failed layers receive new repair tickets instead of threshold changes.

## Definitions

- **Claim ladder:** Separate statements for classic reproduction, registered outcome non-inferiority, and source-specific extension evidence, each with its own required result.
- **Dataset family:** One named generator and parameter set applied across its registered seeds.
- **Dual-family:** The exact canonical source-homogeneous benchmark plus the separately frozen source-heterogeneous extension.
- **Family coordinate:** The explicit artifact and execution key that prevents two datasets with the same integer seed from colliding.

---
id: cff-s6rs
status: in_progress
deps: [cff-9o39]
links: []
created: 2026-08-17T20:52:04Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, statistics, performance]
---
# Measure and calibrate the dual-family comparison

## Background in Plain English

cff-8xve supplies two fixed generators, cff-i00o supplies the claim schema, and cff-9o39 supplies restartable execution. Before final data can exist, the project must measure both workflows, choose family scales, calibrate sampling and optimization budgets, and fix numerical margins from calibration-only evidence.

This ticket performs that measured calibration on seeds 3101 and 3102 only. It produces manifested artifacts that cff-huxt can freeze without rerunning or inspecting any final seed.

## Outcomes / Acceptance Criteria

1. Calibration runs exactly the two registered calibration seeds for both families, both primary tiers, four independent reference chains, and both learned SLDS controls with family-qualified streams and append-only checkpoints.
2. At least three candidate horizons per family have complete-chain, CFFG, control, wall-time, peak-memory, coverage, and whole-run projection evidence. Failed candidate coordinates remain in the artifact.
3. The selected homogeneous scale is justified as a faithful classic benchmark scale; the selected heterogeneous scale satisfies every cff-8xve exposure, outcome, rank, distinctness, and long-rollout eligibility gate.
4. Calibration fixes family-specific or shared optimization budgets, Gibbs warmup/draws/thinning, ESS and split-R-hat thresholds, particles, tolerances, and stopping rules from an explicit rule with domain caps.
5. Calibration fixes paired non-inferiority margins and absolute adequacy/recovery margins for every cff-i00o primary effect without using a final seed.
6. The balanced source-by-state evaluation grid, generator descriptors, scales, splits, data identities, and stochastic coordinates are serialized and hashed in the artifact.
7. Raw family/seed/replicate rows recompute every calibration summary, projection, margin input, and eligibility decision. A complete manifest covers all attempts, retained reference samples, and environment provenance.
8. No registered final seed is generated or fit by this ticket.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound method that satisfies the outcomes.

- Reuse completed coordinates when comparing horizons and record the extrapolation rule separately from measured work.
- Prefer common worst-case budgets across families when that does not weaken diagnostics; otherwise state family-specific budgets explicitly.
- Treat calibration failures as evidence for scale or budget selection, not as rows to omit.

## Future work

cff-huxt converts the complete calibration artifact into the immutable preregistration. cff-oyie owns final execution and must not begin earlier.

## Definitions

- **Calibration-only:** Uses only the registered non-final seeds and may affect protocol choices.
- **Domain cap:** A substantive maximum acceptable loss that calibration variability cannot relax.
- **Whole-run projection:** A declared calculation from measured work to every registered final family/seed/arm coordinate plus headroom.

## Notes

### 2026-08-17 — implementation handoff

`docs/ARCHITECTURE.md` change-path row: N/A. This ticket measures the sibling validation experiment. It does not change CFFG compiler, engine, learning, factor ownership, or public API behavior.

Duplicated execution surfaces checked before calibration:

- `canonical_homogeneous` and `source_heterogeneous` generator dispatch, descriptors, design gates, and data identities;
- both calibration seeds, both inference tiers, four independent reference chains, and both protocol-declared learned controls;
- candidate-horizon execution, coverage and occupancy audits, diagnostic summaries, resource measurements, and whole-run projections;
- paired non-inferiority outcomes, absolute family-specific outcomes, threshold inputs, and fixed-sequence claim records;
- raw attempt payloads, retained reference samples, coordinate checkpoints, aggregate recomputation, manifests, commands, environment provenance, and source revisions;
- the immutable family registry, balanced evaluation grid, primary-effect definitions, claim ordering, and final-seed map, which this ticket must not alter.

Hostile-input decisions:

- Sparse source/gate exposure, one-sided binary outcomes, rank deficiency, or insufficient source-row contrast makes a heterogeneous candidate ineligible before fitted scores are considered.
- Any non-finite metric, incomplete chain, diagnostic failure, convergence failure, budget exhaustion, missing arm, or corrupted checkpoint remains in the raw artifact and cannot support scale or margin selection.
- Calibration runs in explicit float64. The existing float32 generator and functional-score regressions remain the consumer-survival evidence; this ticket does not weaken them.
- Candidate horizons span at least three declared domain sizes per family. Scale selection must use measured complete coordinates rather than extrapolation alone.
- Margin rules are bounded by the cff-i00o domain caps. Calibration variability cannot relax those caps, and final-seed data cannot enter the rule.
- cff-s6rs may increase calibration budgets and rerun failed calibration coordinates under append-only attempts, but it cannot generate or fit a registered final seed.

cff-huxt owns the immutable replacement freeze after calibration is complete. cff-oyie remains the only final-execution owner.

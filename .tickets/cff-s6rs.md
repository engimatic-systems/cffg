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

### 2026-08-17 — classic-scale research

Question → What scale did Linderman et al. use, and what does that permit this project to claim?

Answer → Section 5.1 and the pinned Figure-3 example use one 10,000-step trajectory with four regimes, a two-dimensional latent state, ten-dimensional observations, and 1,000 Gibbs iterations. They fit the complete trajectory; neither source declares a NASCAR train/test split. Therefore 10,000 total steps with an 8,000/2,000 split would not reproduce the paper's fitting scale. cff-s6rs will instead calibrate a 12,500-step canonical trajectory with 10,000 training steps and a registered 2,500-step held-out predictive extension. Passing that protocol may support empirical non-inferiority on the exact canonical generator at the paper's training scale plus the named extension. It may not be called an exact Figure-3 run because the held-out extension and independent implementation differ from the paper workflow.

Source → Linderman et al., Section 5.1, PMLR PDF page 5; pinned `rslds/examples/nascar.py` at commit `383ddac2472cdf3248c6e871baede9ecda6f14cb`, lines 31–52, 167–174, and 579–622.

Confidence → High for dimensions, trajectory length, Gibbs count, and absence of a declared split; medium-high that the paper used exactly one sequence because its wording is singular and the released Figure-3 script creates one.

**2026-08-18T02:52:33Z**

Calibration correction — Question → Does the classic tied-source generator support a primary worst-source transition RMSE evaluated on a shared counterfactual source/state grid? Answer → No. The first paper-scale calibration coordinate made the identifiability failure visible: both implementations fit occupied transition support far better than the counterfactual grid. Shared/native CFFG training-support RMSE was 0.0940/0.0937 but shared-grid worst-source RMSE was 0.4838/0.4844. Adapter training-support RMSE was 0.0533/0.0541 while its per-chain shared-grid RMSE ranged from 0.4132 to 0.4883. The classic truth broadcasts recurrence rows, and the trajectory confounds source with occupied state support, so counterfactual source-row recovery is not identified. The classic primary now uses affine- and label-aligned transition-probability RMSE pooled over naturally occupied held-out truth state/source pairs. The heterogeneous family keeps the shared-grid worst-source and source-contrast primaries because its design establishes distinct rows and full-rank source/gate coverage. Source → checkpointed calibration-only seed 3101 replicate 0; experiment commits 6f68f80 and 9853cfe. Confidence → High. The invalid protocol evidence is manifested under artifacts/rslds-dual-calibration-invalid-classic-grid; no final seed was generated.

**2026-08-18T04:53:06Z**

Calibration checkpoint correction — Question → Did the corrected classic primary work, and why was the first corrected coordinate still not ready? Answer → Both CFFG tiers converged in eight outer iterations and the new held-out on-support transition RMSE was 0.09185 shared and 0.09150 native over all 2,500 held-out states and all four occupied sources. The paired effects and bounded rollouts were also available. The coordinate was not ready only because the optional secondary scorer required the five-regime matched-budget SLDS control to have a four-label permutation. That control itself converged in eleven iterations and its primary metrics were valid. Mismatched-K controls now omit that truth-specific optional score, while K=4 primary arms retain it. Calibration resume now archives validated not-ready checkpoints and retries them as new append-only attempts; attempt 1 and its ledger snapshot remain immutable while attempt 2 runs. Source → calibration-only canonical seed 3101 replicate 0 checkpoint; experiment commits fd94b1d and 1af3d2f. Confidence → High. Focused regression result: 34 passed. No final seed was generated or fit.

---
id: cff-0oqo
status: in_progress
deps: [cff-qulu, cff-q1au, cff-lqz8, cff-95ck, cff-rk7i, cff-nbo9]
links: [cff-oyie]
created: 2026-08-20T15:41:17Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, learning, validation]
---
# Recover source-specific transition functions below the bounded error cap

## Background in Plain English

The one final heterogeneous dataset with a converged CFFG fit provides useful
descriptive evidence, but it does not meet the registered recovery target. On
seed 2101, shared initialization produced worst-source transition error 0.195
and worst source-contrast error 0.250. Native initialization produced 0.376
and 0.374. Every registered ceiling was 0.150.

The learned model responds to source regime, so this is not the original
broadcast-row defect. It is an accuracy problem in source-specific recurrence
learning, alignment, or identification at the observed data scale. The next
step is to locate that error and improve the declared learning path without
using final seed 2101 for tuning.

## Outcomes / Acceptance Criteria

1. A calibration-only decomposition attributes transition and contrast error
   by source, gate, state-grid region, and initialization tier.
2. The diagnosis separates recurrence-learning error from latent affine
   alignment, regime-label alignment, and insufficient source/gate data.
3. Both initialization tiers achieve worst-source transition and source-
   contrast root-mean-square error at most 0.150 on a declared set of
   heterogeneous development seeds.
4. The improvement preserves the classic generator golden, normalized
   transition probabilities, exact-zero support, and the serialized
   `LearnableSpec` path. On declared classic development seeds, on-support
   transition error remains at most the corresponding frozen classic ceiling.
5. Raw recurrence coefficients remain secondary unless an exact canonical
   coefficient mapping and identifiability proof are added.
6. Float32 and a source/gate batch at least ten times the smallest fixture stay
   finite, normalized, and source-specific.
7. The cff-oyie descriptive values and not-ready decision remain immutable.
   Fresh final evidence requires a new calibration and freeze.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Plot or tabulate common-state probability error by source before changing
  optimization.
- Compare the sufficient statistics and exposure weighting for the weakest
  source row against an exact small oracle.
- Resolve cff-qulu first if a development fit cannot reach a certified outer
  state.

## Future work

A later ticket may run a fresh source-heterogeneous confirmation after this,
cff-q1au, and cff-qulu close. Balanced trees and destination-softmax
recurrences remain owned by cff-0f3g and cff-dlrj.

## Definitions

- **Common-state grid:** The fixed set of latent states at which every source
  row is evaluated, including source/state combinations absent from one
  observed path.
- **Source contrast:** The difference between two source rows' transition-
  probability functions at the same latent state.
- **Worst-source transition error:** The largest probability root-mean-square
  error among source regimes on the common-state grid.

## Notes

Architecture handoff: use the declared rSLDS learning change path across model
construction, recurrence sufficient statistics, learnable replacement, and
post-fit functional alignment. Compiler, VM, schedule, kernels, ledger, and
serialization must be checked if the learned-factor contract changes.

Duplicated surfaces: shared and native initialization, every source and gate,
classic tied rows, heterogeneous distinct rows, common-state evaluation,
alignment, float32, and retained secondary reporting must stay aligned.

Hostile-input decisions: exact log-zeros stay exact, vacuous or improper
statistics fail loudly, dtype guards derive from machine precision, and the
large batch in criterion 6 prevents a small-fixture-only repair.

Legend: prerequisite --> dependent

```text
[ ] cff-qulu stable outer learning --+
[ ] cff-q1au robust coverage --------+--> [ ] cff-0oqo recovery
                                                |
                                                +--> [ ] fresh extension run
```

**2026-08-21T01:23:37Z**

Independent-review synthesis: keep 0.15 as the capability ceiling and increase information or improve inference rather than loosening it to match the failed calibration. After robust coverage and certified outer convergence, reevaluate both tiers on fresh 5,120-step training paths with on-support and common-state-grid metrics together. Similarity to the adapter collapse mode is currently correlational; compare occupancy, gate norms, likelihood, and transition surfaces directly before choosing a restart, prior, or recurrence update repair. Unified plan: notes/rslds-debug/synthesis/plan.md.

**2026-08-21T08:41:33Z**

Research handoff (question -> answer -> source -> confidence). What must a prospective diagnostic retain? Point recurrence weights/offsets, terminal posterior means and regime probabilities, pooled recurrence exposure/linear/quadratic readout, fit/stream/config identities, and no generating truth trajectory; regenerate permitted development truth from seed after verifying the observation digest. Source: sibling experiment `confirmatory.py` fit callback, `secondary.py`, `functional_recovery.py`, and CFFG recurrence readout/learning paths. Confidence: high.

What is missing now? Current coordinate JSON keeps aggregate functional errors plus affine/permutation alignment, but discards fitted parameters, posterior beliefs, full source x grid x destination tensors, and grid-origin metadata; old heterogeneous fits cannot be rescored. Confidence: high.

How should gate error be attributed? After destination-label alignment, derive truth-order conditional stick acceptance `p(g)/sum_{j>=g}p(j)` using dtype-scaled remaining-mass guards; raw learned gate-row comparison is invalid because decision-list order is non-identifiable. Preserve source, truth gate, grid point, signed boundary region, and all deduplicated grid-origin memberships. Confidence: high.

Primary falsifiers: known affine/permutation must remove synthetic alignment error; low exposure/conditioning may explain weak cells, while persistent error in well-covered cells falsifies coverage; recomputing recurrence posterior from retained sufficient statistics plus prior must match installed parameters, otherwise rollover/replacement owns the defect. If it matches but surfaces remain wrong, inference/statistics/identification owns the next diagnosis. Confidence: medium-high.

**2026-08-21T09:09:30Z**

Prospective diagnostic foundation integrated in sibling experiment commit 36c8360. Every screened heterogeneous CFFG fit can now emit an append-only truth-free checkpoint containing point parameters, coherent terminal posterior means/regime probabilities, pooled recurrence exposure/linear/quadratic/validity, observation digest, and exact candidate/stream/config/code identity. The pure diagnostic regenerates only authorized development truth after digest verification, preserves grid origins and signed regions, aligns affine/labels across all 24 K=4 permutations, decomposes destination and conditional-stick errors by source/truth-gate/region, and checks retained-statistic posterior consistency. Tampering and generating-truth fields are rejected. Combined focused validation passed 106 tests; no scientific coordinate ran. Empirical diagnosis is intentionally funneled through cff-8w4c screen fits so the same expensive CFFG fits serve both tickets.

**2026-08-22T01:10:56Z**

Authenticated seed-5000 baseline post-fit diagnosis. The isolated experiment branch cff-0oqo-seed5000-diagnostic preserves code commits a8e6bfa and 217f6ce. Artifact results SHA-256 ca2cd604e5060683004ac09aaf6d108d71f5aff3ba0a489401b0ee9f986d53a2; manifest SHA-256 9376dacabe0a0b9f4b68decb1f56d7800f2077d407516fa2ea1aff101dcac9bd. A second semantic replay matched both retained baseline tier checkpoints exactly. No fit, adapter, control, calibration seed, or final seed ran.

Cap decision: at the declared 0.150 ceiling, shared initialization missed worst-source transition 0.260685 and worst contrast 0.283856; native missed 0.191202 and 0.280330. Both fits had converged outer certificates. All 24 regime-label permutations were evaluated. The best shared permutation still had worst-source 0.259411 and worst contrast 0.272118. The best native permutation still had 0.173472 and 0.268559. No permutation met both caps.

Localization: shared was worst at source 3 and source pair 2 versus 3. Its well-covered conditional-gate failures were source/gate 0/2, 1/2, 2/2, and 3/0; their worst regions were respectively negative 0.816476, negative 0.563286, negative 0.672990, and boundary 0.344175. Native was worst at sources 2 and 3 and source pair 2 versus 3. Its well-covered failures were 0/2 positive 0.337731, 1/2 negative 0.742471, 2/0 positive 0.151482, 2/2 negative 0.678703, 3/1 positive 0.245639, and 3/2 positive 0.581489. Every conditional-stick cell passed the dtype-scaled remaining-mass validity check at threshold 5.68434e-14. Persistent misses in well-covered cells, including cells with posterior-effective exposure above 800 and 2000, falsify insufficient coverage as a general explanation.

Alignment and posterior decision: affine latent RMSE was 0.006474 shared and 0.006769 native with condition number about 3.56, and exhaustive label search did not rescue the caps. The native retained recurrence parameters matched their posterior statistics within the 0.1 tolerance: normalized difference 0.043329. Shared failed that independent consistency check: normalized difference 0.175048. The common failure is therefore real functional recurrence recovery, not only label, affine, coverage, or diagnostic error; shared also has a separate installed-parameter versus retained-posterior inconsistency to investigate. Keep cff-0oqo open: acceptance criterion 3 is not met, and this one seed does not establish the required declared multi-seed recovery.

**2026-08-22T14:09:33Z**

Four-prior seed-5000 post-fit comparison is authenticated on the committed cff-8w4c screen. Isolated experiment commit 6e720f7 preserves `artifacts/rslds-heterogeneous-recovery-development/seed-5000-prior-screen`; results SHA-256 14e27153d025fc8d4db51cf749ccc45bafe0025d8c322983159f629da9b660fd and manifest SHA-256 02c346f4b562759e2b020f93b726fb524535d97522a41fbba7d88b6eb7020bd8. The screen stage, all candidate protocols, eight checkpoint/evidence pairs, and attempt manifests validated. A separate semantic replay reproduced every source/gate/region summary and independent posterior solve exactly. No fit, chain, adapter, control, calibration seed, or final seed ran.

Selected-label worst-source/worst-contrast errors by recurrence prior were: shared 0.001 0.260685/0.283856, 0.1 0.291951/0.255385, 0.3 0.279334/0.236083, 1.0 0.275405/0.224425; native 0.001 0.191202/0.280330, 0.1 0.191802/0.275918, 0.3 0.189756/0.273012, 1.0 0.171525/0.240256. Every value misses at least one 0.150 cap. Exhaustive search over 24 labels found zero joint passes for every tier/prior.

The prior changes preserve the error topology rather than repairing it. Against prior 0.001, prior 1.0 gate-error Pearson/Spearman correlations were 0.946/0.965 shared and 0.992/0.986 native; region correlations were 0.773/0.913 and 0.992/0.980. At prior 1.0, shared gate 2 worst-region errors remained 0.797468, 0.676457, 0.672060, and 0.672942 across sources 0 through 3. Native gate 2 remained 0.386771, 0.695052, 0.611433, and 0.527108. Native source/gate 2/2 had posterior-effective exposure 2049.20; 3/2 had 742.92. This falsifies prior strength and insufficient coverage as complete explanations, although stronger priors modestly improve native source and contrast errors.

CFFG posterior rollover exactly equaled an independent solve of `(prior precision + quadratic) inverse times (prior eta + linear)` in all eight checkpoints, with zero maximum algebraic difference. Native retained posterior consistency passed for every prior. Shared failed only at prior 0.001: normalized terminal recurrence-map difference 0.175048 despite the returned previous-map fit residual 0.067588. Shared priors 0.1, 0.3, and 1.0 passed at 0.037572, 0.041456, and 0.056625. The baseline next recurrence update would worsen functional errors to 0.310908/0.304102. This falsifies an algebraic `PGWeights.update` or rollover defect and shows one separate terminal fixed-point rebound in the weak-prior shared arm; it does not explain the common native miss.

Ranked next mechanism: first, biased latent/posterior responsibilities or the mean-field recurrence statistic, especially late decision-list gates; second, initialization-dependent occupancy and basin effects; third, the separate weak-prior terminal convergence certificate. Smallest path: keep cff-0oqo open and first add a development-only truth-clamped recurrence M-step oracle on this same seed to decide whether the observed sequence identifies the gates when hidden states are exact. If that oracle passes, compare retained factor-local soft statistics against an exact small-chain oracle before changing recurrence VMP or statistic code. If it fails, improve the heterogeneous design or identifiability rule instead of the engine. Track the terminal full-map recheck as a separate engine bug because it changes convergence truthfulness, not recurrence recovery. Do not choose prior 1.0 as a fix: it still misses both native caps and worsens shared worst-source recovery.

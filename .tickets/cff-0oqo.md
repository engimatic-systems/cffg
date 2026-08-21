---
id: cff-0oqo
status: in_progress
deps: [cff-qulu, cff-q1au]
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

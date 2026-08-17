---
id: cff-ah4t
status: open
deps: [cff-0l31, cff-3vhn]
links: []
created: 2026-08-17T18:24:08Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, statistics, performance]
---
# Refreeze a feasible source-specific reference scale

## Background in Plain English

The cff-0l31 calibration measured about 8.5 seconds for one paper-adapter chain with 60 Gibbs iterations and 64 training steps. The frozen final protocol requires 436 iterations, 8,000 training steps, four chains, two tiers, and 24 seeds. Linear extrapolation is about 2.1 hours per chain and 17 days for the 192 sequential chains before CFFG fits and reporting. A measured four-chain batch was slower than sequential execution because the legacy stack saturated memory and compute.

No final seed has run. cff-0l31 explicitly allowed a separate reference-driven scale tier when justified, but its frozen document selected the canonical scale before this end-to-end projection was reviewed. This ticket chooses and freezes a scientifically useful execution scale that can finish while preserving the 24-seed paired design, calibrated sampling budget, and strict diagnostics.

## Outcomes / Acceptance Criteria

1. A committed benchmark on calibration seeds measures complete-chain wall time and peak memory at no fewer than three candidate training horizons. It records the command, iteration count, environment, and exact extrapolation rule.
2. The chosen primary comparison scale has enough transitions per source regime and decision-list gate to support the registered recovery and predictive metrics. The ticket states that evidence with observed calibration occupancy rather than relying only on total T.
3. The replacement preregistration names one exact data horizon and train/held-out split for both primary tiers and both learned SLDS controls. All arms use identical observations, splits, and data seeds.
4. The final design retains seeds 2001 through 2024, both primary tiers, all six simultaneous primary effects, the calibrated margins, four independent chains, ESS and split R-hat gates, strict failure handling, and retained posterior samples.
5. Canonical T=10,000 survival remains a separate CFFG robustness claim. If the primary reference comparison uses a smaller tier, README and final reports state that limit prominently and do not call the comparison canonical-scale parity.
6. The chosen scale has a measured whole-run projection and a checkpoint/restart plan that preserves append-only raw seed artifacts without inspecting aggregate effects early.
7. A new immutable preregistration and manifest replace the cff-0l31 freeze before cff-oyie begins. They preserve the old freeze as superseded provenance and record why it was not executed.
8. No final seed is generated or fit by this ticket.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound method that satisfies the outcomes.

- Benchmark candidate training horizons such as 64, 128, 320, and 640 using only seeds 3101 and 3102.
- Prefer more observations when regime/gate occupancy and total runtime remain acceptable; development cost alone is not a reason to weaken evidence.
- Use per-seed append-only execution so an interrupted multi-hour run can resume without replacing completed seeds.

## Future work

cff-3vhn owns the missing secondary report surfaces. cff-oyie owns final execution of the replacement frozen protocol. cff-nkbj owns the scoped conclusion and must name any non-canonical comparison limit.

## Definitions

- **Candidate horizon:** A calibration-only sequence length measured before the final scale is frozen.
- **Checkpoint/restart plan:** A deterministic way to retain completed seed artifacts and resume only missing registered seeds.
- **Gate occupancy:** The number of observed transitions assigned to each source regime and decision-list decision gate after fitting-time truth boundaries end.
- **Reference-driven scale tier:** A comparison scale selected because the independent reference implementation cannot practically execute the canonical horizon.
- **Whole-run projection:** A stated calculation from measured per-chain and per-fit time to the complete registered seed, tier, chain, and control set.

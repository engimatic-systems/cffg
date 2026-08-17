---
id: cff-ah4t
status: in_progress
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

The cff-0l31 freeze selected T=10,000 before its simple linear runtime extrapolation was checked against complete fits. Direct calibration-only measurements now project 4.69 hours at T=80, 5.02 hours at T=160, and 7.37 hours at T=400 for all 24 seeds, both tiers, four reference chains, both CFFG fits, both controls, and 25% headroom. Runtime alone therefore does not justify weakening the experiment.

The same audit exposed a more important coverage boundary. Fitting sees only the 80% training split; several source/gate cells have one-sided binary outcomes even at T=3,200, and the NASCAR truth repeats identical recurrence rows across source regimes. The existing generator can measure transition-probability error on realized source/state support, but it cannot demonstrate recovery of distinct source-specific gate coefficients. No final seed has run. This ticket must resolve that evidence scope, measure the chosen design, and freeze the replacement protocol while retaining the 24-seed paired design, calibrated sampling budget, and strict diagnostics.

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

## Notes

**2026-08-17T18:36:32Z**

Pre-implementation handoff: docs/ARCHITECTURE.md change-path index is N/A because benchmarking and replacement freeze remain in the sibling experiment repository. Duplicated execution surfaces: shared and native paper-adapter chains at each candidate horizon; CFFG shared/native fits for whole-run projection; both learned SLDS controls; regime and gate occupancy; append-only seed artifacts; replacement preregistration and manifest. CFFG compiler, VM, schedule, kernel, ledger, serialization, and production learning are N/A. Legend: prerequisite --> dependent. [x] cff-0l31 --> [x] cff-3vhn --> [ ] cff-ah4t (ready/in progress) --> [ ] cff-oyie --> [ ] cff-nkbj. cff-ah4t owns calibration-only scale evidence and the replacement freeze; cff-oyie retains final-seed ownership.

**2026-08-17T19:13:50Z**

Decision handoff after the measured scale and coverage audit. No final seed has been generated or fit.

Measured result: calibration seeds 3101 and 3102 completed all reference chains and both CFFG/control rows at T=80, 160, and 400. Target projections with 25% headroom were 4.69, 5.02, and 7.37 hours respectively. The runtime evidence is committed in the experiment repository at 73e0873. A later provenance fix now captures the source commit before the first fit.

Audit correction: the first artifact's occupancy field counted the full trajectory even though fitting sees the 80% training split. The separately manifested correction at experiment commit 4717a79 names and hashes the corrected artifact. Training minimum gate exposures for seeds 3101/3102 are T=400: 3/2; T=800: 5/5; T=1,600: 11/10; T=3,200: 21/21. Every candidate still has at least one source/gate cell with zero successes or zero failures. T=400 seed 3102 also has augmented design rank two for a three-parameter affine gate. The benchmark now records exposures, binary outcomes, and augmented-state design rank explicitly.

Research question: what does that coverage identify, and is any current scale enough for a source-specific recovery claim? Answer: an exposure is one reached decision-list gate and therefore one binary logistic datum for the source/gate's two-dimensional slope plus intercept. Coefficient recovery also needs full-rank state variation and both outcomes without complete separation. Ground-truth path occupancy diagnoses generator coverage; it is not the posterior-effective exposure used by either observation-only fitter. The canonical NASCAR generator repeats the same recurrence slopes and offsets across every source row, so it exercises the source-specific representation but does not contain source-heterogeneous recurrence truth. None of the audited horizons supports individual recurrence-coefficient or source-specific boundary recovery. T=3,200 is the first audited horizon above the judgmental floor of 20 training exposures per source/gate and can support only the narrower registered transition-probability RMSE on realized source/state support. Sources: Linderman Eq. 4, 6, and 7 as mapped in `reference/adapter-provenance.json`; exact hard gate statistics in `reference/legacy-runtime/src/legacy_runtime/paper_adapter.py`; soft exposure derivation in `docs/DERIVATIONS.md`; canonical tied recurrence in `src/cffg_nascar_validation/nascar.py`; on-support probability error in `src/cffg_nascar_validation/secondary.py`. Confidence: high on the exposure semantics, one-sided cells, and tied generator; medium-high on the conservative 20-exposure floor, which is a stated engineering judgment rather than a paper theorem.

Decision required: (A, recommended) add a prerequisite that redesigns calibration and final generation with genuinely different source rows, state coverage around every boundary, and per-source/gate full-rank designs with at least five successes, five failures, and 20 total training exposures; then repeat calibration and scale measurement. This preserves the epic's source-specific evidence intent. (B) retain the tied-source NASCAR generator, freeze T=3,200 after a direct whole-run measurement, and narrow the epic/final claim to same-model non-inferiority and transition probabilities on realized support, explicitly excluding source-specific coefficient recovery and evidence for source heterogeneity. cff-ah4t cannot honestly satisfy acceptance criterion 2 until that scientific scope is chosen.

Implementation already landed on the stacked experiment branch: measured low-scale benchmark and manifest (73e0873); append-only per-seed checkpoints with provenance and per-attempt manifests (34c6a8c); start-time benchmark provenance (065f4a2); corrected training-coverage diagnostics (dc9ec66); and the manifested occupancy correction (4717a79). The preregistration test file passes 23 tests at the checkpoint commit, and focused scale tests pass after the audit correction. The replacement freeze is intentionally not written before this decision.

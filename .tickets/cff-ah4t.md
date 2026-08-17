---
id: cff-ah4t
status: closed
deps: [cff-0l31, cff-3vhn]
links: []
created: 2026-08-17T18:24:08Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, statistics, performance]
---
# Measure rSLDS scale and audit source-gate coverage

## Background in Plain English

The cff-0l31 freeze selected T=10,000 before its simple linear runtime extrapolation was checked against complete fits. This ticket measured the complete comparison at three smaller calibration horizons and added a restart-safe execution path. The measurements show that runtime alone does not justify weakening the experiment.

The coverage audit found a more important boundary. Fitting sees only the 80% training split. Several source/gate cells have one-sided binary outcomes even at T=3,200, and the canonical NASCAR truth repeats identical recurrence rows across source regimes. That dataset can test the classic demonstration and transition behavior on visited states, but it cannot demonstrate recovery of distinct source-specific gates. Cooper chose to retain exact canonical NASCAR and add a separately named source-heterogeneous extension. cff-8xve owns that generator; cff-i00o owns the dual-family runner and replacement freeze. No final seed ran in this ticket.

## Outcomes / Acceptance Criteria

1. A committed benchmark on calibration seeds measures complete reference chains, both CFFG tiers, and both learned controls at no fewer than three horizons. It records wall time, peak memory, command, iteration count, environment, and its exact whole-run projection rule.
2. A separately manifested correction counts only fitting-time transitions and records each source/gate's exposure count, binary successes, binary failures, and augmented-state design rank.
3. The audit states which claims the tied-source canonical generator can and cannot support. It distinguishes classic benchmark outcome evidence from distinct source-specific coefficient recovery.
4. Final execution writes each completed seed once, hashes its result and raw attempt manifest, preserves interrupted attempts, and resumes only missing registered seeds with matching protocol and preregistration hashes.
5. The roadmap retains exact canonical NASCAR and assigns the source-heterogeneous extension and dual-family immutable freeze to named downstream tickets before cff-oyie.
6. No final seed is generated or fit by this ticket.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound method that satisfies the outcomes.

- Measure whole workflows rather than extrapolating every cost linearly from T=64.
- Count only the training split when judging fitting coverage.
- Report gate outcomes and affine-design rank in addition to total exposure.
- Keep completed seed artifacts append-only and place a retry in a new attempt directory.

## Future work

cff-8xve owns the source-heterogeneous NASCAR extension and its calibration-only coverage proof. cff-i00o owns dual-family runner integration, calibration, claim registration, and the replacement immutable freeze. cff-oyie owns final execution. cff-nkbj owns the scoped claim ladder.

## Definitions

- **Augmented-state design rank:** The rank of the gate's continuous-state columns plus its intercept column. Rank three is required to locate a two-dimensional affine gate without relying only on its prior.
- **Checkpoint/restart plan:** A deterministic way to retain completed seed artifacts and resume only missing registered seeds.
- **Gate exposure:** A transition that reaches one decision-list gate and therefore contributes one binary observation for that source/gate pair.
- **Source-heterogeneous:** The recurrence slope or offset changes with the current discrete regime.
- **Source-homogeneous:** The recurrence slope and offset are tied across current discrete regimes.
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

**2026-08-17T20:50:00Z**

Cooper selected dual evidence. Exact source-homogeneous NASCAR remains the faithful classic benchmark. A separately named source-heterogeneous extension will exercise distinct current-regime gate rows. Passing the classic family may support preregistered empirical non-inferiority on that benchmark; passing the extension may support broader source-specific capability evidence. Neither result alone implies authors-code, inference-algorithm, message-equation, or free-energy parity. The implementation continues in cff-8xve and cff-i00o; this measurement and audit ticket is complete without a final seed.

Legend: prerequisite --> dependent

    [x] cff-ah4t: measured scale and coverage audit
          |
          +--> [ ] cff-8xve: heterogeneous generator (ready)
                    |
                    +--> [ ] cff-i00o: dual-family integration and freeze
                              |
                              +--> [ ] cff-oyie: immutable final execution
                                        |
                                        +--> [ ] cff-nkbj: scoped claim ladder

    Preserved parallel evidence inside cff-i00o/cff-oyie:
    exact canonical_homogeneous + source_heterogeneous extension

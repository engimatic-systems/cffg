---
id: cff-8xve
status: closed
deps: [cff-ah4t]
links: []
created: 2026-08-17T20:46:52Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, generator]
---
# Implement a source-heterogeneous NASCAR extension

## Background in Plain English

The canonical NASCAR generator uses four different motion regimes, but it repeats the same three recurrence gates for every current regime. It is faithful to the classic demonstration, yet it cannot show whether distinct source-specific gate rows work. Cooper chose to keep that exact benchmark and add a separate extension whose switching probabilities genuinely depend on the current regime.

This ticket creates that extension in the sibling `cffg-nascar-validation` repository. It must remain recognizably NASCAR-like, use the same source/destination and parameter conventions as CFFG and the paper-equation adapter, and provide enough calibration-only gate coverage to identify every two-dimensional affine gate. It does not run confirmatory seeds or freeze statistical claims.

## Outcomes / Acceptance Criteria

1. The existing canonical generator and its golden outputs remain unchanged and continue to be named the source-homogeneous classic benchmark.
2. A separately named source-heterogeneous generator has at least two numerically distinct recurrence source rows in both its declared parameters and its transition probabilities at shared probe states.
3. The extension retains four destination-regime motion fields, a bounded long rollout, repeated track traversal, and visits to all four regimes without using generating states during fitting.
4. Calibration uses only seeds excluded from confirmatory execution and chooses one fixed generator design before any final seed runs.
5. At the selected calibration horizon, both calibration seeds give every source/gate at least 20 training exposures, at least five successes, at least five failures, and rank three for the augmented two-dimensional state-plus-intercept design.
6. A manifested calibration artifact records the generator parameters, seed, horizon, train split, data identity, source/gate exposures, outcomes, design ranks, long-rollout bounds, periodicity, regime rates, and exact generating commit.
7. The extension uses explicit PRNG seeds, remains finite in float32 and float64 generation checks, and survives a T=10,000 generation-only run without an absolute numeric floor or clamp.
8. Sparse or one-sided source/gate coverage fails calibration loudly instead of being accepted through a prior or aggregate total.
9. Tests prove that CFFG and reference mappings receive the same source axis, destination convention, observations, and truth-free fitting boundary for the extension.
10. No seed in the registered final set is generated or fit by this ticket.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound method that satisfies the outcomes.

- Start from the canonical four motion fields and introduce modest source-dependent gate translations or rotations instead of replacing the track geometry.
- Search or construct parameters with calibration seeds only, then commit the selected values and reject later tuning.
- Evaluate gate outcome balance and augmented-design rank on the 80% training split.
- Keep canonical and extension constructors explicit rather than adding a boolean whose meaning can leak across artifacts.

## Future work

cff-i00o defines the dual-family statistical protocol. cff-9o39 builds family-aware execution, cff-s6rs performs measured calibration, and cff-huxt writes the replacement freeze. cff-oyie owns final seeds. cff-nkbj owns the eventual claim ladder.

## Definitions

- **Affine gate:** One logistic decision boundary with a two-dimensional slope and scalar intercept.
- **Augmented design:** The observed source states with an added constant column for the intercept.
- **Classic benchmark:** The unchanged source-homogeneous NASCAR generator used by the original demonstration.
- **Source regime:** The current discrete regime `z[t]` from which a transition starts.
- **Source-heterogeneous:** At least one recurrence slope or offset differs across current discrete regimes and changes transition probabilities on shared states.

## Notes

### 2026-08-17 — implementation handoff and refined dependency path

`docs/ARCHITECTURE.md` change-path row: N/A. This ticket changes only the sibling validation experiment's declared dataset generator and calibration evidence; it does not change a CFFG engine owner or execution surface.

Duplicated execution surfaces checked before implementation:

- canonical constructor and a full parameter/trajectory golden;
- explicit heterogeneous constructor and family dispatch;
- shared source-row/destination-regime conventions consumed by CFFG and the reference adapter;
- calibration-only coverage, outcome-balance, rank, float32/float64, and long-rollout checks;
- final-seed generation and fitting: N/A and forbidden by this ticket.

Hostile-input decisions:

- Sparse, zero, one-sided, or rank-deficient gate support fails calibration before fitting.
- Improper or vacuous incoming messages are N/A because this ticket only generates datasets.
- Float32 transition probabilities and generated values must remain finite and normalized.
- A calibration-only `T=10,000` rollout exercises a larger domain than inference fixtures.
- Every stochastic operation starts from an explicit legacy-compatible `RandomState` seed; no numeric floor, ridge, or clamp is introduced.

Legend: prerequisite --> dependent

```text
[x] cff-ah4t scale and coverage audit
      |
      +--> [ ] cff-8xve heterogeneous generator (ready/in progress)
              |
              +--> [ ] cff-i00o protocol and claims
                      |
                      +--> [ ] cff-9o39 family-aware execution
                              |
                              +--> [ ] cff-s6rs measured calibration
                                      |
                                      +--> [ ] cff-huxt replacement freeze
                                              |
                                              +--> [ ] cff-oyie final run
                                                      |
                                                      +--> [ ] cff-nkbj evidence epic
```

### 2026-08-17 — calibration-only generator research

- Question: Can a fixed heterogeneous generator exercise each source/gate cell without changing the classic generator? Answer: yes. Keep the canonical dynamics and emission arrays, rotate the three recurrence normals by `(-12, -4, 4, 12)` degrees across sources, and choose source offsets from fixed anchor transition probabilities. Sources: `src/cffg_nascar_validation/nascar.py`, `.reference-sources/rslds/examples/nascar.py`, and the cff-ah4t occupancy artifact. Confidence: high.
- Question: Does the proposed design satisfy the ticket's coverage gates? Answer: on calibration seeds 3101 and 3102 at `T=1,600` with the 80% training split, the minimum cell has 45 exposures, nine successes and nine failures, and augmented design rank three. At `T=10,000`, both rollouts are finite and bounded; float32 transition probabilities remain finite and normalized on the checked state grid. Source: calibration-only construction audit performed before implementation; the committed artifact must reproduce these measurements. Confidence: high.
- Question: What recovery outcome is defensible? Answer: use aligned transition probabilities on both realized states and a shared source-by-state grid, plus source-function contrasts. Raw recurrence coefficient error is not automatically label invariant because relabeling destinations changes the sequential gate order. Sources: `reference/adapter-provenance.json`, `paper_adapter.py`, and `docs/DERIVATIONS.md` source/destination convention. Confidence: high.

### 2026-08-17 — completed

Implemented in experiment commits `1de5478`, `a021473`, `cec2d57`, and `ab9d976` on `cff-8xve-heterogeneous-nascar`.

- The canonical family is protected by a complete rounded parameter/trajectory hash and remains unchanged through explicit family dispatch.
- The separately versioned heterogeneous family keeps the classic dynamics, emissions, noise, and legacy RNG stream while supplying four fixed recurrence source rows.
- The independent mapping matches the generator's source-conditioned probabilities; existing observation-only fit boundaries accept no generating truth.
- The calibration-only artifact at `artifacts/rslds-source-heterogeneous-design` is eligible on seeds 3101 and 3102 at `T=1,600`, with minimum exposures `52/45`, minimum binary outcomes `9/9`, rank three in every cell, worst design conditions `1.362/1.478`, and worst Fisher conditions `9.532/9.121`.
- Both `T=10,000` rollouts are finite and bounded, visit all four regimes, and have periodicity `0.830/0.797`. The minimum pairwise shared-grid source-surface distance is `0.307` mean total variation.
- Artifact result SHA-256: `049967b2c6b66b8db36c523d9c65da106ef00a967fc4ccaee8dec12a0827d977`.
- Final validation: `46 passed in 24.17s`. No registered final seed was generated or fit.

Next: cff-i00o is ready to define the family-aware schema, absolute functional recovery outcomes, and fixed-sequence claim gatekeeping before execution code is generalized.

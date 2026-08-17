---
id: cff-8xve
status: open
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

cff-i00o integrates both families into calibration, statistical registration, restartable execution, reporting, and the replacement freeze. cff-oyie owns final seeds. cff-nkbj owns the eventual claim ladder.

## Definitions

- **Affine gate:** One logistic decision boundary with a two-dimensional slope and scalar intercept.
- **Augmented design:** The observed source states with an added constant column for the intercept.
- **Classic benchmark:** The unchanged source-homogeneous NASCAR generator used by the original demonstration.
- **Source regime:** The current discrete regime `z[t]` from which a transition starts.
- **Source-heterogeneous:** At least one recurrence slope or offset differs across current discrete regimes and changes transition probabilities on shared states.

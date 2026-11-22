---
id: cpt-j81k
status: in_progress
deps: [cpt-ydse]
links: [cff-wt1c]
created: 2026-11-22T21:28:35Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym, benchmark]
---
# Compare four fixed-model reactor controllers on thirty paired seeds

## Background in Plain English — WHY
A one-seed pilot verified four implementations and found two pairs of nearly identical returns. Cooper wants to know whether the control and forecast differences repeat across independently drawn measurement noise.

## Outcomes / Acceptance Criteria — WHAT
1. Complete 59-step and 599-step control for all four frozen controllers on 30 new paired seeds under the one-second warmed deadline.
2. Score twelve matched 60-step forecasts per seed for each controller, retaining every failure and pairing the future commands and random particle draws.
3. Publish seed-level differences, uncertainty, failures, tracking, latency and forecast accuracy in a local HTML report with raw evidence.

## Methodological Suggestions — HOW
Reuse pilot reference gates and frozen solvers. Primary outcome is the CFFG minus peer summed return on 599 decisions. Use seed-level paired bootstrap intervals and disclose numerical ties. Architecture path is external runtime and numerical readout consumers. Engine/JIT, vmap, data serialization and production model math are unaffected. Sparse support, dtype expansion and higher-dimensional modelling are outside this fixed float64 fixture.

## Future work
cff-wt1c owns broader model, learning and runtime comparisons.

## Definitions
A paired seed supplies matching measurement noise to independent controllers. A forecast origin is a point where observations stop and prediction begins. An interval across resampled seeds measures uncertainty due to this source of variation.


## Notes

**2026-11-22T21:33:11Z**

The scored protocol was frozen at 15457bf before measuring the first held-out return. Seeds 1001 through 1030 had no role in selecting the adapters. Each seed has eight episodes and four forecast batches, run serially in fresh processes. Short and long horizons reuse the same seed. Arm order rotates across seeds and phases. Use the one-second warmed deadline, held actions on failure, unchanged clipping and all archived outcomes. Forecast truth and particle seeds are respectively 100000 + control seed and 200000 + control seed. The fixed open-loop commands give the same physical truth across seeds. Forecast uncertainty is across measurement and particle draws.
Primary comparison is mean paired 599-decision return (CFFG minus each peer), with 20000 paired seed bootstrap draws, 95% pointwise percentile intervals, and numerical ties at absolute return differences up to 1e-8. Short return and forecasts are secondary. Manifest, qualification and per-seed raw data are in ~/agent-share/pcgym-paired-30. Launch command from the worktree is uv run --frozen --no-sync python tools/pcgym_timing/run_paired.py --output ~/agent-share/pcgym-paired-30 --peer-root ~/agent-share/pcgym-four-library-timing. All four fresh qualification workers passed. The measurement session is live.

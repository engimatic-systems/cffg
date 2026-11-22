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


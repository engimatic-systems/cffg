---
id: cff-vsn3
status: in_progress
deps: [cff-961d]
links: []
created: 2026-11-24T05:32:53Z
type: feature
priority: 2
assignee: coopwilliams
parent: car-kwjj
tags: [axiom, vision]
---
# Initialize and grow spatial slots on real pixel frames

## Background in Plain English — WHY

A fixed set of slot positions can explain known objects. New objects need a reproducible first frame and a way to claim a free slot when a pixel is poorly explained. This is the stage between the proven spatial equations and temporal tracking.

## Outcomes / Acceptance Criteria — WHAT

1. Produce normalized row-major position/RGB tokens from raw uint8 frames using the released scales. Provide an explicitly seeded initializer using the fifth spatial template and separate immutable priors, learned arrays and current positions.
2. Force the first frame onto the first slot, and perform the specified coordinate and parameter updates. On later frames run ordinary position/assignment updates without a parameter write until the minimum best absolute likelihood crosses the threshold. Choose the first unoccupied row and worst explained pixel deterministically. Distinguish good fit, exhausted capacity, exhausted growth budget and numerical failure.
3. Exercise well-explained and novel scenes, repeated growth and capacity, dormant slot preservation, a finite posterior at reference size, and a real recorded development frame. Report intermediate assignments, positions, counts and growth trace for review. Keep thresholds and fixed update budgets distinct from convergence claims.

## Methodological Suggestions — HOW

Use cff-zpk8 and cff-961d. Separate the statistics update from the model-specific growth rules. Check each substep with hand-built pixel/slot histories and pinned release descriptions in car-8rgl. Include dtype/support/input guards and compare a masked write to the pointwise prior on each slot.

## Future work

car-kwjj carries slot motion, identity, visibility and reuse across observations and supplies inputs to car-mps6. A full-agent campaign waits for all of these.

## Definitions

A hard winner is the slot with the largest assignment probability for a pixel. A vacant slot has no hard winners, even when its total soft probability is positive. A growth attempt forces one pixel to a vacant row before updating the appearance model.


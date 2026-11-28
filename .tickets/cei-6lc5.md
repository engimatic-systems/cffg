---
id: cei-6lc5
status: in_progress
deps: [cff-u06e]
links: []
created: 2026-11-28T06:17:51Z
type: feature
priority: 2
assignee: coopwilliams
parent: cenc-luez
tags: [numerics, intervals]
---
# Carry full supported interval cells through an invertible activation with a fixed capacity

D13a requires independent interval potentials on continuous coordinates. An array of fixed length can silently drop a thin supported interval when a knot or a likelihood boundary splits it. Build a reusable device kernel with an observable capacity result, as the foundation for the interval factor.

## Design

New factor or rule family and Runtime numerical or status invariant apply. Represent sorted disjoint [lo, hi) cells per coordinate, finite log heights and an explicit live mask. Encode missing unit as a single whole-real cell, zero measure as no live cells. Split crossed knots exactly. Form intersections and check capacity before committing. Keep the raw scale for evidence. The pointwise map sends the forward density divided by its slope and the backward likelihood by composition alone.

## Acceptance Criteria

JIT/vmap and scalar results match an independent interval integration in both precisions from 1 to 32 coordinates. Preserve separated cells, vacuous inputs and structural zeros, reject occupied-cell overflow and forward/inverse collapse without a partial output. Report impossible, improper, numerical and overflow separately. Compute log mass and differential entropy on proper finite support. No covariance floor or Gaussian replacement. Serialization and graph scheduling belong to the integration ticket.


## Notes

**2026-11-28T06:18:24Z**

Design before kernel: per coordinate at most C sorted disjoint half-open cells with strictly ordered endpoints, finite absolute log height; an occupied tail may have an infinite endpoint. Empty support is zero, and a unit message has one (-inf,+inf) cell of log height 0. Unary knot split produces at most n+1 cells, product intersects at most n+m-1 if both are nonempty. Preserve adjacent breaks as declared; no tolerance merging. Insufficient capacity rejects the complete update transaction (no live prefix). Check transformed finite endpoints and strict ordering within cells and across positive gaps, along with finite log-height arithmetic. Distinct flags for invalid declaration, numerical loss and capacity. Log mass and entropy as independent-coordinate sums; zero support dominates a divergent tail on another coordinate. Scalar and batch share the same functional kernel/shape, and graph schedules and source/codec binding are downstream in cei-k93f.

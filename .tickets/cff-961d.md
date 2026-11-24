---
id: cff-961d
status: closed
deps: [cff-zpk8]
links: []
created: 2026-11-24T05:26:05Z
type: feature
priority: 2
assignee: coopwilliams
parent: car-kwjj
tags: [axiom, vision]
---
# Infer spatial slot positions and pixel assignments for a frame

## Background in Plain English — WHY

An appearance fit supplies the conditional probability of a pixel given a candidate object. The agent also needs the inverse update: given all the uncertain pixel assignments, where is each object? This is an independently testable part of spatial perception.

## Outcomes / Acceptance Criteria — WHAT

1. Use the per-channel regression posterior and Gaussian position prior to update every slot position from fractional pixel assignments. Keep covariance across spatial coordinates, include coefficient and bias uncertainty, and preserve the perspective of a fixed prior for a frame.
2. Infer assignments from expected pixel log likelihood and expected log mixing probabilities, then positions given those assignments. Preserve finite normalized assignments on supported slots, zero excluded support, previous estimates for an unassigned slot, and report the best absolute likelihood per token and hard occupancy.
3. Verify dense independent one-step and multi-step oracles, exact zero mass, arbitrary slot permutation, uncertainty, float32/64 and one full-resolution development frame. Record boundary between these operations and growth/temporal tracking.

## Methodological Suggestions — HOW

Compute the two Gaussian natural parameters from the expected squared pixel error. A static bias coordinate of one contributes to the linear term. Check the answer independently by evaluating the expected log joint on a finite-difference grid or with dense observation equations. Use the frame-local weighted regression primitive and the released assignment-then-position schedule. The existing frame prior and current posterior have distinct roles during each iteration. Architecture rows are Agent, planning, or hierarchy and Runtime numerical or status invariant. General solve schedules, checkpoints and the graph ledger are outside this frame operation.

## Future work

car-kwjj adds initialization, threshold-driven proposals, identity, and carrying slots between frames. The frozen observation tapes belong to car-8rgl.

## Definitions

A spatial slot is a Gaussian belief about an object position. A responsibility assigns a fraction of a pixel to a slot. Hard occupancy counts the pixels that prefer a slot over every other slot.


## Notes

**2026-11-24T05:30:00Z**

Two-sweep 32-slot / 33,600 pixel smoke passed on Explode base frame 0; fully coupled update versus independent information solve using the returned final assignments differed by max 1.15e-5 normalized coordinate and 1.20e-7 covariance. Three targeted spatial tests (two precisions, independent quadratic fitting, permutation, zero support) plus four underlying frame regression tests. Full suite pending.

**2026-11-24T06:02:02Z**

The retained AXIOM full-frame equation includes a rank-one 1e-8 contribution to position precision, including the negative off-diagonal covariance its two-axis fixture predicts. Independent posterior and full-frame checks pass. The separate growth/initialization work uses the same equations and passed on real frames.

---
id: cff-zpk8
status: in_progress
deps: []
links: []
created: 2026-11-24T05:20:14Z
type: feature
priority: 2
assignee: coopwilliams
parent: car-kwjj
tags: [axiom, learning, vision]
---
# Fit a weighted pixel appearance model with one noise precision per observed channel

## Background in Plain English — WHY

The spatial mixture assigns each image pixel partly to each candidate object. Its appearance model predicts five measured values from an uncertain slot position and a constant bias. Each output needs its own learned noise level. This mathematical step can be qualified on a single frame while model growth and continual storage are under review.

## Outcomes / Acceptance Criteria — WHAT

1. Score observed output vectors under a full Gaussian input for each slot and a matrix-normal linear transform with one Gamma precision per output. Provide one shared input precision per slot, an optional bias through input augmentation, and a mask over participating slots.
2. Given fixed pixel assignment weights, compute the conjugate posterior from an immutable prior for a frame. Preserve zero-mass slots, reject invalid inputs atomically, and compare the mean, design precision, shape and rate to an independent scalar or dense oracle. Include uncertain inputs and a deterministic bias.
3. Exercise float32 and float64, exact zeros, invalid prior, near-collinear inputs, translated predictions, eager and compiled calls. Score and update 210 by 160 pixels across 32 slots, including deterministic assignments; record size and observed execution cost with a synchronized CPU smoke.

## Methodological Suggestions — HOW

Use centered observation residuals when computing the posterior rate. An independent augmented least-squares fit with a square root of the input covariance provides an oracle. The change follows the New factor or rule family and Learnable or checkpoint feature index for standalone conjugate primitives. No registered message rule, solver schedule, graph schema or long-lived bank snapshot is required for a frame-local fit.

## Future work

car-kwjj supplies the actual slot detection, identity and frame-to-frame inference schedule. Serialized continual bank state is handled by cff-xfnx and car-k6oi.

## Definitions

A slot is a candidate object. A Gamma distribution represents precision, the inverse noise variance. A fractional assignment is the share of a pixel attributed to a slot.


## Notes

**2026-11-24T05:25:00Z**

Frame regression and a full-resolution probe landed on the shared branch. Direct CPU x32/x64 oracles passed four targeted tests. On 33,600 RGB+position tokens across 32 slots, max scaled discrepancies versus a full float64 raw-moment oracle were 2.43e-7 in scores, 2.47e-5 in means and 3.12e-5 in rates. Five synchronized compiled development calls took 4.79–5.52 ms. These are local frame-fit measurements with supplied slot beliefs, including score, softmax and fit, not full frame-to-action times. Full gate with parent pending.

---
id: car-kwjj
status: open
deps: [car-k6oi, car-8rgl, cff-cpxj]
links: []
created: 2026-11-23T21:48:38Z
type: feature
priority: 2
assignee: coopwilliams
parent: car-zieb
tags: [axiom]
---
# Infer and track AXIOM object slots and identities from raw frames

## Background in Plain English — WHY

A Gameworld frame contains colors and positions. AXIOM has to find objects, keep track of them as they move and recognize their types on later frames. A Gaussian mixture over pixels solves only the first part. The temporal bookkeeping determines which object can teach the motion and interaction models.

## Outcomes / Acceptance Criteria — WHAT

1. Add weighted matrix-normal regression with a separate Gamma noise precision per output for the slot appearance fit. Score and aggregate pixel position/color evidence across active slots, including background and unused slots. Verify one batched frame against a scalar weighted-regression oracle, with zero-weight pixels, sparse assignments, float32 and near-degenerate features.
2. Implement the spatial slot mixture (sMM) with prediction from the previous frame, per-frame assignment updates, worst-explained-token growth and the release’s conditions for parameter updates. Keep the initialization template, active mask and observation mask independently testable. Match reference assignments, positions, visibility and growth events on a small pixel scene and at 210×160 with 32 slots.
3. Implement the identity mixture (iMM) on the applicable object features. Preserve the distinction between previously visible foreground used for identity learning and motion/use history used to decide which objects teach dynamics. Exercise a static foreground object, disappearance, reappearance and a vacant slot being reused. Compare identity responsibilities, selected writes and sufficient statistics with replay fixtures.
4. Export one frame-to-frame state contract with slot positions, uncertainty, identity beliefs, motion/use summaries and the masks consumed by car-mps6. Check resumed and continuous updates and slot permutations on a labeled synthetic fixture. Inject known motion predictions to qualify this module without waiting for the full world model.

## Methodological Suggestions — HOW

Keep Gameworld-specific feature geometry, birth thresholds and heuristics in the AXIOM model. Put reusable weighted regression and batched reductions in the library. Use cff-cpxj for existing plate semantics where appropriate and car-9ccf for identity updates. Review “New factor or rule family” and “Learnable or checkpoint feature”; check scalar/batched, host/compiled, current/legacy bundles and both execution schedules for any registered rules.

## Future work

Connecting live predicted motion and the public real-observation cycle belongs to car-zieb. Broader out-of-distribution object recognition and a general-purpose vision API can be evaluated separately.

## Definitions

A pixel assignment expresses how strongly a pixel belongs to a slot. A type groups objects with similar features. The Gamma precision controls inverse noise variance for one output dimension.

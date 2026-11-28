---
id: car-kwjj
status: closed
deps: [car-k6oi, car-8rgl, cff-cpxj, cff-zpk8, cff-961d, cff-vsn3, cat-825z]
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

## Notes

**2026-11-24T06:00:54Z**

The first image layer, hard-winner births, soft updates, frame-to-frame tracks and identity learning are implemented on the evaluation branch. The actual observations from three games produced finite updates in float32 and float64. All six halfway reloads reproduce the final arrays, including a real episode reset. Correctness evidence is mathematical, synthetic and fixed-action. The seeded absent-neighbor value is an explicit per-run JAX draw; the author release drew it from NumPy at trace time. A direct paired trace of the authors’ own solver would need separately authorized outputs (car-ozw0).

**2026-11-28T04:27:05Z**

Published-main audit after #207, a774a54. The frame regression and spatial/slot/tracking tests cover AC1, the small-scene equations and growth gates, and the static/disappearing/reused-slot clocks. The archived 12 full frames give 403,200 hard assignments with zero f32/f64 disagreements from a common start. The 33,600-pixel oracle checks position updates conditional on the production assignments. The six real 48-action scene streams (three games × two dtypes) restore exactly from action 24. These are candidate precision and restart checks. We still need an independent full-size expectation for pixel membership, position, visibility and birth on the same input. The scene carries the identity bank and appearance features; the public identity helper returns only the first maximum. Complete AC4 with a derived probability readout from the saved scene, including empty-bank and failed-read behavior, and a labeled persistence/permutation check. Independent synthetic frames suffice. Under the Learnable or checkpoint feature change path, exercise eager/JIT, f32/f64, sparse/empty support and checkpoint replay. The derived readout needs no new IR, inference schedule, serialized field or schema migration.

**2026-11-28T04:35:31Z**

Implemented the remaining checks on independent synthetic RGB frames. The NumPy/SciPy oracle solves each position by augmented least squares, computes each pixel likelihood and assignment, and fits appearance from the immutable regression prior and weighted raw observations. Starting with two known appearances and injected nonzero motion predictions, it finds the single unexpected magenta pixel, chooses the first vacant slot from its own hard winners, and computes both refinements following the forced fit. The observed frame path matches all soft probabilities, each hard label and occupancy, the visibility mask consumed by tracking, position means/covariances, the birth pixel and slot, and the stopping status. This covers a small 8×10 frame in float32 and full 210×160 frames with 32 slots in both precisions. The code uses no game pixels, author-agent outputs or production numerical helpers to calculate expectations.

Scene.identity_beliefs() derives current per-slot type probabilities from the trained bank and saved appearances. It supports both sizes-plus-color and color-only features, the uniform empty-bank case and explicit invalid evidence. The existing hard identity remains the first maximum. The independent labeled fixture checks unequal, analytically known odds, a vacant component, a hidden slot's retained color, permutation of slot histories, and exact NPZ restoration. It adds no fields to archived scenes. Its analytic learned rows use kappa=1. In an exploratory artificial float32 bank with active rows still at the unused prior kappa=1e-4, small likelihood differences lost about 1.1e-5 in one reported probability, owing to the common term d/kappa in the existing scorer. No new sub-ULP precision claim is made.

Focused validation included all 22 tests in test_axiom_slots.py, test_axiom_observation.py and test_axiom_mixture.py, including the opt-in cases. Rechecked the five final identity/spatial variants, then all three spatial variants after adding the explicit decoded visibility and stopping-status assertions (3 passed in 7.11 s). The parent coordinates the final short suite. Full-size synthetic and optional color-only checks are marked slow, with small default examples retained. The implementation and module fixtures qualify the independent CFFG behavior. Direct author-model trajectories remain unavailable as recorded in car-ozw0.

**2026-11-28T05:26:42Z**

Integrated gate on fd9a0f7, based on #208. The four-worker dashboard passed all 2,844 selected tests in 455.037 seconds. Full-size synthetic spatial cases also passed in the focused run recorded above.

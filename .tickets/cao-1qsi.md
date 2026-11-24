---
id: cao-1qsi
status: closed
deps: [cao-z3o7, cao-m63a, car-ne0a]
links: []
created: 2026-11-24T05:43:25Z
type: feature
priority: 2
assignee: coopwilliams
parent: car-mps6
tags: [axiom]
---
# Plan from learned AXIOM interaction and motion tables on a supplied scene

## Background

The sampled planner accepts a prediction callback. Real tracking supplies slot states, identity labels and the tracking mask. Connect the trained interaction and motion tables to that callback so an observed transition can affect later simulated decisions.

## Outcomes

1. Freeze read-only acquired rMM and tMM tables at each real decision. Construct contextual scene evidence with a supplied simulated action, hide the two outcome heads, sample the next switch and advance the simulated scene. Preserve stable component labels, tracking, full-bank novelty mass and explicit keys.
2. Check reward, novelty, state, and component assignments against the full-bank calculation including a newly learned contact. Verify the source tables stay intact and an invalid snapshot leads to a failed decision. Include empty support, float32/64 and a 32-slot scene.
3. Measure a bounded warmed simulation subset using 5,000 components with sparse occupancy and 500 motion classes. Record shape, compiler preparation, synchronization, working memory, effective dimensions and any limits to extrapolating to the full policy budget.

## Method

Use the architecture change paths for numerical status and learnable snapshots. Compare eager/compiled and scalar/batched rollouts. Graph flooding and checkpoint schema changes are inapplicable. Keep stable indices when compacting occupied snapshot rows, and retain the complete concentration sum for the released count bonus. Use a small independently scored scene before the capacity check.

## Future work

The enclosing perception task supplies pixel-derived tracking. Run the paired held-out study only under its own frozen protocol.

## Definitions

A snapshot is the learned model as of one real decision. A rollout is an imagined sequence of actions from that state; its predicted outcomes do not train the model.


## Notes

**2026-11-24T05:43:30Z**

Qualify on learned and supplied slot histories, then run a bounded K=5000 timing check. Host compacting occurs once per real snapshot. Runtime weights retain masses of dormant components in the reciprocal-count baseline.

**2026-11-24T05:54:59Z**

Qualified freeze_world/predict_scene/rollout_step on both precisions against full component and slot axes. Absolute scores, responsibilities, stable physical class indices, next state, reward and novelty match. An observed contact acquired tMM class 4 and a positive rMM reward; the read-only sampler reused that class and selected action 0. Reset starts with zero tracked objects and yields zero scores. A tracked evaluation-only scene with no previously acquired interactions returns NO_TRACKED_INTERACTION_SUPPORT; the release's finite mask would sample dormant rows. Timing: Apple M2 Max JAX CPU, float32, K=5000 (8 occupied), M=500, 32 objects (6 tracked), default geometry, 16 candidates x 3 draws x 8 frames = 384 imagined transitions. Final blocked runs 9.18, 8.30 and 11.56 ms; first/warm snapshot preparation 806/.746 ms, compile 1.389 s, XLA temporary 1,837,088 bytes, executable inputs 110,404 bytes, process peak 703,856,640 bytes. Prior development build: 7.01–7.19 ms at the same dimensions. A simple 128x projection gives 0.9–1.5 s per game-budget action at this occupancy; batching and occupancy change this rate. No complete game-budget measurement on this live-model shape. Reproduce with uv run --frozen --no-sync pytest -q -s -m slow tests/test_axiom_rollout.py (passed). Final ordinary bridge/planner check: 13 passed, 2 slow deselected in 37.87s. Parent owns cumulative suite and shared documents.

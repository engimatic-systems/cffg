---
id: cat-825z
status: closed
deps: []
links: []
created: 2026-11-24T05:39:54Z
type: task
priority: 2
assignee: coopwilliams
parent: car-kwjj
tags: [axiom]
---
# Carry observed slots across frames and gate identity and dynamics learning

## Background in Plain English — WHY

A spatial fit describes slots in the current frame. The identity learner needs the preceding appearance of a visible object, while the motion and interaction learners need the current tracking decision and the actual transition. Confusing those times makes an object teach the wrong learner. Cooper owns spatial fit and growth. Dirac owns the motion representation, interaction features and shared parameter updates.

## Outcomes / Acceptance Criteria — WHAT

1. Consume decoded slot means and covariances, hard pixel occupancy and total soft assignment mass. Preserve the eleven coordinates used by the motion/interaction models, the last observed covariance, and per-slot visibility, foreground and motion/use history. Use the existing observed_state implementation for motion coordinates.
2. Provide separate boolean write masks for identity and dynamics. Tie foreground to current maximum soft mass, identity to previous visibility, and dynamics to current tracking including the explicit background exclusion. A reset observation can advance the frame state with all learner writes disabled.
3. Exercise first observation, static and moving foreground, missing and reused slots, teleport suppression, tied background masses, malformed and nonfinite inputs, both working precisions and 128 slots. Rejection keeps the previous carry usable. Reordering foreground rows commutes with the update. Reloading the saved tracking arrays gives the exact next state and masks.
4. Show the previous and current states and the returned masks feed existing motion and interaction inputs directly. Scope any parity claim to the independently constructed fixtures and pinned source interpretation.

## Methodological Suggestions — HOW

Use the release description in car-8rgl and read only the necessary symbols at b99f625. The tracker's inputs are decoded moments in normalized image coordinates. A probability mass is soft membership summed over pixels. Occupancy means at least one pixel chose a slot. Check the new owner beside axiom_motion.observed_state on the integration branch.

## Future work

car-kwjj owns the joint spatial/identity stage. car-mps6 uses the real transitions and masks to learn motion and reward. car-zieb owns the complete agent checkpoint, including the model, simulator and RNG. car-8rgl owns the reference tapes.

## Definitions

A carry is all state passed from one frame to the next. An exponential moving average gives a fixed fraction of its weight to the newest observation. A teleport is a displacement at or above the configured motion threshold.


## Notes

**2026-11-24T05:40:07Z**

Architecture review: this is an experiment-owned deterministic state transform in the Learnable or checkpoint feature path. Compare scalar and batch behavior, host and JIT status, and a restored tracking carry. Both graph schedules, registered factor rules, learned bank persistence and bundle migration are N/A. Reuse axiom_motion.observed_state at the shared local integration head 5c50479. Initial qualification uses injected decoded moments and thin masks. The frozen Gameworld reference PR is #201 at 69d7315.

**2026-11-24T05:45:04Z**

Implementation ef55367 on feature/axiom-slot-tracking, based on local integration commit 5c50479. The ticket commit is 67fe6af. The tracker takes the result of slot_descriptors(model,positions) plus the same frame's SpatialReadout through DecodedSlots.from_readout(mean,cov,readout,accepted=...). Call initial_tracks on the first observation and track_frame(previous,decoded,update=...) on subsequent observations, including a reset with update=False. The scene contains eleven coordinates. Missing slots keep their last appearance and covariance. Save all TrackCarry arrays and the tracking settings. Invalid measurements preserve the previous carry and disable both writes.

Evidence clock for the integration owner: use the previous scene's appearance under identity_eligible, and the previous and current scenes under dynamics_eligible for observed motion. The interaction features take the previous scene and current tracked mask plus the actual action, realized reward and inferred motion switch. Cooper owns the iMM bank factory, learn/query wrapper and causal scene integration. Dirac confirmed the generic learner interfaces and checked the ownership split. The reference branch remains independent.

Five focused cases passed in 7.50 seconds at ef55367. They exercise float32 and float64, JIT, static and moving foreground, a teleport contributing to the motion average before the stored velocity is zeroed, previous-visible evidence, missing/reused slots, invalid inputs, hard/soft membership, 128 slots, foreground permutations, exact NPZ carry continuation, closing writes on reset and actual acquire_motion/interaction_batch calls. Ruff and diff checks passed. From this worktree use PYTHONPATH="$PWD/src" uv run --project /Users/ravd/projects/cffg-axiom-online --frozen --no-sync python -m pytest -q tests/test_axiom_tracking.py.

A separate four-slot smoke used the first two RGB frames in the committed reference fixture stream. Cooper's slot_descriptors source file SHA-256 was df91d00e029f55a146f19142684d6b70201613ce76627a8b7681a46e9458a4ac. Initial and subsequent tracking produced valid [4,11] scenes. The single allowed growth attempt produced the explicit spatial GROWTH_BUDGET status. This checks the shape and numerical handoff; the caller decides how to accept that status during a study.

The child implementation and focused qualification are complete. The short suite belongs at the integrated candidate head before publication. There were no campaign runs. car-kwjj still owns full-scene execution.

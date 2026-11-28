---
id: cao-zqih
status: closed
deps: []
links: []
created: 2026-11-24T14:52:23Z
type: task
priority: 2
assignee: coopwilliams
parent: cao-srin
tags: [axiom, benchmark]
---
# Replay native Gameworld actions across simulator process restarts

## Background in Plain English — WHY

A long run can lose the simulator process between checkpoints. The actor must receive the same next observation after recovery. The development smoke adapter supports one game. The primary evaluation needs all ten.

## Outcomes / Acceptance Criteria — WHAT

1. Each game accepts the published integer actions in order. Each reset and step returns the native pixels and its clock position. Each step also returns the applied action, native reward and end flags.
2. A fresh process recreates the game from its seed and a sequence of receipts. Recovery checks each frame digest, reward, end flag, action, tick and episode.
3. Development checks exercise all games against their published development tapes, including a terminal step followed by a reset. Record the source and results.

## Methodological Suggestions — HOW

The MIT Gameworld source and locked environment in the reference archive supply the native simulator. Reuse its isolated random stream and action inventory. The CFFG architecture change-path rows are N/A for this experiment tool. Host/JIT, flooding/scheduled, single/vmap and model checkpoint decoding are N/A. Verify the process boundary against archived native frames.

## Future work

The parent ticket cao-srin owns durable actor checkpoints, execution and scoring of the evaluation.

## Definitions

A receipt records what the simulator actually returned. Its SHA-256 frame digest identifies the exact bytes of the RGB observation. A tick counts an applied action across episode resets.


## Notes

**2026-11-24T14:55:20Z**

The new simulator runs the installed MIT Gameworld in the pinned private uv overlay. Construction, each step and each reset share the game's original random stream. The parent gets read-only native frames and records separate reset and step receipts. A fresh worker can reconstruct a checkpoint prefix by replaying the applied actions and checking every receipt. The caller binds the full header and ordered receipts to the checkpoint. Recovery costs one native step per prior action.

Focused validation after the final code change: uv run --frozen --no-sync python -B tools/axiom/evaluation_game_check.py --output ~/agent-share/axiom-gameworld-reference/development/evaluation-game-check.json. All ten action inventories, 480 base transitions and 11 resets matched the archived bytes, rewards and clocks. A second process restored a prefix and continued exactly on all ten. Aviate recovered immediately after termination before the tick-47 reset. The checker exercised both eight-action Explode perturbation prefixes and confirmed the configured action-5000 switch. Corrupt receipts and out-of-order commands were rejected. Development game seeds 41010–41019 only. The private overlay emitted its usual Gymnasium reset-signature deprecation notices. Isolated simulator tooling; focused validation covers its process and data boundary. Downstream campaign recovery and scoring remain with cao-srin.

---
id: cao-urtu
status: in_progress
deps: [cao-36dd, cao-1qsi]
links: []
created: 2026-11-24T05:59:40Z
type: task
priority: 2
assignee: coopwilliams
parent: car-zieb
tags: [axiom, development]
---
# Act and learn on 24 real Explode transitions with the integrated scene

## Background

Individual tests cover scene updates and learned rollouts. A real closed loop must choose the command from current pixels, execute it once, and train on the returned pixels and native reward.

## Outcomes

Use MIT Gameworld Explode v0.1.0, development seeds game 270037 and model 87037, 24 real actions, 32 slots and identities, 500 motion rules, 5000 interaction rules, float32, and H8 x 16 candidates x 3 draws. Verify the qualified simulator source and packages in the reference overlay. Record every actual action, reward, termination and reset receipt. Report partial failure with its location if encountered.

Compare every observed model leaf before and after planning. At action 12 save and reload the exact observation model, proposal and next random key from disk. Recompute the next decision and require identical proposal, sampled policies, score arrays and command. Apply that command just once. Record model and planner timing separately from simulator transport, disk and audit costs. Count actual learner writes/births, tracked slots, unique actions, nonconstant scores, and realized rewards.

## Method

Use the learnable/checkpoint and runtime-status architecture paths. Compare original versus restored callers of the same public planner and use the observed tick counter. The release starts with all objects untracked. A native terminal/truncation triggers a reset observation with zero dummy action and reward, learning disabled and action count preserved. Separate simulator and model interpreters because the two locked NumPy versions differ. Verify byte-level RGB and action clock at their boundary.

## Future work

The epic owns longer runs, scheduled reductions and held-out claims. This isolated tool changes no production model or learner.

## Definitions

A real action is a Gymnasium step. An imagined transition advances only the frozen model. Reload replays a decision from saved arrays and an explicit random key.


## Notes

**2026-11-24T05:59:47Z**

Freeze 24 real actions in the unperturbed game under the two fresh seeds. Collect synchronized first-call, first-call-by-shape and reused-shape latencies. Learner timing includes perception, tracking and the observed mixture updates; model hashing and serialization are outside these clocks.

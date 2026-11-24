---
id: cao-urtu
status: closed
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

**2026-11-24T06:04:12Z**

Completed the 24-action closed-loop development smoke from clean detached source 8605c13c4536d643e5bbbb2824d7e8550b02e1fe. The qualified Python 3.11 Gameworld child attested MIT source, versions and overlay lock; the shared model process attested all 28 loaded CFFG/tool/lock files against Git. Output ~/agent-share/axiom-online-precision/acted-smoke.json SHA256 c37fed8473489526a858365c21df5f6b9cad95b04586fdfbe03ad4dd2afaa1ea. Independent archive check matched each pre-frame to its preceding post/reset hash, all 24 planned/applied/observed actions and rewards, clocks and model hash pairs, plus the next-action checkpoint replay and the source bytes. Nine stay, eight left, seven right. No simulator terminal/reset or nonzero reward; reset branch is wired but unexercised on this trajectory. All spatial statuses valid. 21 decisions had differing novelty scores; predicted reward had zero spread. Final active identities/motions/interactions 3/13/52, identity/interaction writes 67/64. At tick 12 restored 104 leaves (50,877,880 bytes) and reproduced action 2, all plan arrays and next key. Warm decision median 10.41 ms (n17), first per bucket 1.46–2.27 s (n7); first observe 4.877 s then warm median 104.35 ms (n23). Source py_compile/help, scoped git diff --check and tk dep cycle passed. This isolated tool imports existing public model calls and has no runtime, CI or benchmark dependency changes. Reproduce from an immutable checkout with JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 uv run --frozen --no-sync python tools/axiom/acted_smoke.py --output <fresh-path>. Parent and report owner were sent the validated evidence digest. Longer acting, reduction and any held-out campaign remain separate.

---
id: cao-lxmq
status: in_progress
deps: [cao-36dd, car-ne0a, cao-fs5s, cff-wdhj]
links: []
created: 2026-11-24T06:03:43Z
type: feature
priority: 2
assignee: coopwilliams
parent: car-zieb
tags: [axiom, agent, checkpoint]
---
# Carry AXIOM world, planner and reduction state through a supported agent restart

The ordinary rolling-agent extension covers one Gaussian belief and the applied command. The AXIOM development runner needs spatial slot beliefs, tracking history, identity and interaction banks with reusable rows, motion rules, a sampled proposal, old simulated reduction records and random keys. An exact replay of a Python tree verifies the math but does not define what another user can load.


## Reviewable outcomes

1. Expose a typed entry point for a real RGB observation, proposed action, applied action, observed reward, optional reset frame and monotonic real-action counter. The serialized model owns its shape, units, layer settings, capacities, random-key implementation and release conventions. Preserve the existing categorical and Gaussian agent interfaces.
2. An exact checkpoint before a real action and another after observing its outcome reproduce all sampled action sequences, chosen command, subsequent posterior, component identities and scheduled reduction at the next boundary. A simulator reset leaves learned experience in place. Neither a replayed transition nor an abandoned plan may contribute evidence twice.
3. A compact checkpoint omits optional floating-point remainders. Document where compact restart can change the policy. Separate fixed model descriptions or reproducible priors from per-step mutable state; measure real default and optional file sizes and restore time at a populated 5,000-row bank. Report ≥10% default size or speed changes to Cooper before accepting them.
4. Validate complete schema and input assets before loading. Missing fields, invalid slot references, incompatible capacities, incorrect keys and nonfinite beliefs give an actionable failure with the current controller intact. Exercise fresh-process restart on a tiny enumerated world and a recorded or acted Gameworld scene.

## Method and boundaries

Review the Learnable or checkpoint feature path. Compare host/compiled and scalar/packed state, Compact vs Exact, legacy categorical and current Gaussian checkpoints. The experiment already supplies the numerical leaves. Start with a typed AXIOM state and documented reconstruction inputs. The reproducer does not need an arbitrary callback serializer or a new whole-graph objective. The paper's reward campaign and public reference-run provenance have separate tickets.

Prerequisite --> dependent

    observed scene + sampled action search + interaction reduction
                + supported rolling-agent boundary
                --> typed AXIOM continuation --> car-zieb

A continuation includes both what the agent learned and where it was in the real-observation / imagined-action cycle. A compact restart can round parameter evidence once at load. The actual observations, episode metadata and applied commands remain exact.

## Notes

**2026-11-24T14:53:18Z**

Evaluation checkpoint slice on experiment/axiom-10000. The controller factory owns its model layout. The actor owns the scene, pending decision if supported, planner and reduction state. The runner supplies exact source/stage/game/seeds/goal/tick/trace metadata and replays the native environment externally. Use the Learnable or checkpoint feature change path. Check disk restart in a fresh actor, clock/metadata and complete array inventory, precision and keys, accumulated reduction histories, and rejection before mutating the destination. Numerical inference and its host/compiled/schedule paths are shared and unchanged. This assignment covers exact evaluation restart only. Compact/published facade qualification remains separately tracked.

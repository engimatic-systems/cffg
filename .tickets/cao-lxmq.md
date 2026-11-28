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

**2026-11-24T15:00:12Z**

Exact evaluation codec landed at 382e0f9. The caller supplies source and trace digests, stage, game, both seeds, action budget and completed tick. The fixed Actor factory also identifies reset pixels and action cardinality. Restore checks the SHA-256 across names, dtypes, shapes, values and metadata, reduction generator and tape, fixed priors, bank lifecycle, spatial belief and planner support before installing. The positive check uses a two-slot synthetic actor with a live interaction and a nonzero float32 remainder at action 499. A separate interpreter restores the pending plan, takes the observed transition and actual 2,000-draw reduction, restores again at 500 and takes the next decision. Candidate/score/key digests and every archived byte agree at 500 and 501. Incompatible metadata, version, dtype, key, lifecycle, missing field and altered content fail without changing the destination. uv run --frozen --no-sync pytest -q -m "slow or not slow" tests/test_axiom_evaluation_checkpoint.py passed both checks in 30.17 s. The durable evaluation runner owns native simulator replay and full-capacity prequalification. The public agent facade, compact export and its size/restore tradeoff remain open on this ticket.

**2026-11-28T06:22:31Z**

Begin from #223 and the committed evaluation actor/codec from #204. The Learnable or checkpoint feature and Runtime numerical or status invariant change paths apply. Verify the two sealed Explode snapshots, reset image, identity and trace SHA before promoting the single implementation. Preserve the schema-1 exact reader and default, pending action clock, native reward, PRNG stream and accumulated reduction tape. Exercise fresh-process exact and explicitly rounded compact restore, invalid/partial/foreign file rejection without mutation, f32 and conjugate remainder checks, unchanged categorical and Gaussian facades. The same JAX kernels serve host/compiled, planner schedules and packed actors; no new numeric rules or engine schedule changes are needed. Use the original retained full-size state and timed read/write comparisons only; no game run.

**2026-11-28T06:32:22Z**

Rechecked original pilot provenance and both native archives before promotion. The real reset image from development/Explode/base.npz matches the pilot hash. Tick 9500 has 339 live interaction rows of 5,000 and 1,275,968 bytes. Tick 10000 has 344 live rows and 1,334,882 bytes. Both content digests and trace prefixes recompute, and the final file and state digest agree with the published result. Original evidence is read-only. The installed AxiomAgent reuses the existing planner and reducer. Six focused legacy and public checks passed in 84.54 s, including fresh interpreters crossing a real 2,000-record reduction at action 500, compact agreement with the independent lifecycle codec, and rollback on malformed state. After binding the global JAX precision flag in the header the schema check passed again. The four relocated implementation modules imported from the built wheel.

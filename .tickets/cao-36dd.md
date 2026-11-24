---
id: cao-36dd
status: closed
deps: [cao-0pbk, cff-vsn3, cao-z3o7]
links: []
created: 2026-11-24T05:44:59Z
type: feature
priority: 2
assignee: coopwilliams
parent: car-zieb
tags: [axiom, integration]
---
# Initialize AXIOM identity and interaction banks and replay the first observed transitions

Connect the released Gaussian and categorical priors, previously learned identities, current motion switches and the real applied action and reward. Compare individual writes before launching any acting experiment. Keep the seeded, unused parameters needed for unmasked reduction separate from prior sufficient statistics.


## Reviewable outcomes

1. Supply the untuned identity and interaction priors in either dtype. Distinguish a never-used row's seeded, read-only appearance from the shared reset prior and a previously released row. Check all categorical widths and concentration values against the frozen model configuration.
2. Decode an RGB observation, carry spatial beliefs, compute the two tracking masks and fit each identity from its previous visible feature exactly once. Infer motion from the observed transition, use the newly learned identity bank to label the previous scene for interaction learning, and supply the actual switch, applied action and observed reward. Preserve all parameters atomically on an invalid transition.
3. Inspect a finite fixed-action sequence with births, later matches, per-head writes and no use of Gameworld labels. Preserve trace hashes, test an invalid input and compare resumed and continuous state. The complete acting loop, public facade, reduction, planning performance and seed-level return have their own tickets.

## Method and scope

Use the Learnable or checkpoint feature and Runtime numerical or status invariant architecture paths. Compare host to the compiled record update, float32 to float64 on controlled cases, and one-slot to batched behavior. A prior is the parameter record restored on an empty slot. A dormant slot has a reserved row but has yet to receive any real evidence.

## Notes

**2026-11-24T06:00:54Z**

At immutable 09702a4 both source precisions completed 48 real actions apiece in Explode, Bounce and Aviate. Across 288 real transitions there were two extra reset observations; resets preserved every dynamics/identity/interaction array and did not advance the action clock. Exact midway NPZ replay matched every final array on all six runs. Final live interaction counts were 105/105 in Explode, 129/112 in Bounce, 72/72 in Aviate (float32/float64). The independent native seeds share a numerical seed but produce different initial arrays across dtypes; source attribution for the tracking difference is under review. See ~/agent-share/axiom-online-precision/final-scene-{f32,f64}.json. This is a fixed-action development trace, not a policy reward study.

**2026-11-24T06:02:02Z**

The release CLI defaults to float32 (--precision_type float64 enables x64). The development replay exercised both explicitly. The public evidence inventory cannot establish the dtype of the published runs. The scene clock distinguishes a reset observation from a real gated update, and the release’s NumPy trace-time absent-neighbor draw is modeled as an explicit JAX seed so replay is deterministic.

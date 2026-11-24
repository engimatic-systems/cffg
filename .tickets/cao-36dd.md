---
id: cao-36dd
status: in_progress
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

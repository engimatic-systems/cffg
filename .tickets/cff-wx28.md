---
id: cff-wx28
status: in_progress
deps: []
links: []
created: 2026-07-09T03:50:32Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-ahps
---
# Parameter initialization templates: canonical warm starts as blob specs

Solve the warm-start problem generally (Cooper 2026-07-09): parameters rarely want to be uniform or random — they want structured canonical forms (diagonal-dominant 'sticky' transitions, a Gaussian band along the diagonal, one-hots, shift dynamics, seeded Dirichlet draws), and Dirichlet learners want those same shapes as COUNTS with a confidence knob. Design: (1) cffg/inits.py — a registry of pure, seeded initializer functions make(kind, shape, **params); every probability-template kind takes an optional strength parameter that converts it into Dirichlet counts (strength x P + positive floor) — one knob turns any template into a confidence-weighted warm start for learn_a/learn_b/initial beliefs. (2) BlobRef grows an optional init spec (canonical sorted key/value string tuple, like ConstraintSpec.detail) so a template is fully self-contained without external blob files; serialization additive. (3) Compiler resolution order: explicit arrays > existing blob file > init spec > error — so continual-learning checkpoints (files) override the warm start on reload. (4) Structure hash erases init specs exactly like blob paths (they are parameter data, not structure — colony batching unaffected). Vocabulary v1: uniform, one_hot, sticky, banded (Gaussian along the diagonal), shift, random_stochastic, constant, zeros, eye, random_spd. Registry is open: new kinds are table entries.

## Acceptance Criteria

Property tests per kind (stochasticity/SPD/shape/seed-reproducibility); counts-strength semantics tested; serialization + hash-erasure round trip; end-to-end compile from an init-only template (no arrays, no files); docs updated in place.


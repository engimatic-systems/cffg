---
id: cff-63dd
status: open
deps: [cff-ifbr]
links: []
created: 2026-08-22T15:57:08Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-0oqo
tags: [rslds, serialization, checkpoint]
---
# Version structured recurrence persistence and model semantics

## Background in Plain English — WHY

D9 changes what a `recurrence_link` inference state and learned coefficient
posterior mean. The serialized factor spelling and parameter shapes stay the
same, but a checkpoint produced from D8 singleton-product statistics is not
equivalent to one produced from D9 pair statistics. Restoring it silently
would mix two learning objectives under one identity.

This ticket makes that behavior change explicit across CFFG bundles,
learnable checkpoints, rSLDS helpers, and downstream checkpoint consumers.

## Outcomes / Acceptance Criteria — WHAT

1. The persisted executable or learning identity distinguishes D8
   singleton-product recurrence semantics from D9 structured-pair semantics
   wherever runtime or posterior state can be restored.
2. A model-only bundle with no runtime or learned recurrence state remains
   loadable and reruns inference under the current documented semantics.
3. A legacy checkpoint containing recurrence posterior state is rejected by
   default or accepted only through a named, explicit migration that preserves
   its legacy meaning. It cannot silently resume structured learning.
4. Current structured checkpoints round-trip pair/statistic identity,
   learnable hyperparameters, factor/source ownership, dtype, and shapes
   without changing values.
5. `cffg.models.rslds`, `RSLDSLearning`, and fit/checkpoint helpers document
   which objects can be reused as priors and which require fresh inference.
6. Structure hashing and serialization remain deterministic. If no IR field is
   added, tests prove why template identity alone is insufficient for runtime
   checkpoint compatibility and where the executable identity is stored.
7. Tampered, incomplete, unknown-version, wrong-factor, wrong-shape, and
   non-finite recurrence checkpoint data fail before any inference or learning
   update.
8. README, ARCHITECTURE, and LIMITATIONS describe the current structured
   semantics and the legacy migration boundary in place. No new Markdown file
   is created.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method.

- Version the smallest recurrence-specific checkpoint boundary instead of
  changing unrelated generic learnable families.
- Keep model bundles reconstructible from template plus blobs; executable
  semantics belong in a manifest or checkpoint identity, not a large inline
  blob.
- Add hostile decoding tests before writing a migration.

## Future work

cff-h0hx will update experiment checkpoint consumers and generate fresh
development evidence. Historical scientific artifacts remain immutable and
must continue to name their original CFFG commit and semantics.

## Definitions

- **Model-only bundle:** A serialized template and parameter blobs without an
  inference state or learned posterior checkpoint.
- **Executable identity:** A version or digest that states which inference and
  statistic semantics produced persisted runtime evidence.
- **Migration:** An explicit conversion or compatibility path between two
  persisted schema or behavior versions.

## Notes

Architecture handoff: serialization manifests, structure hashes, bundle
loading, `Learnables.state_dict/load_state_dict`, `RSLDSLearning` checkpoints,
agent checkpoints where recurrence learnables are present, and experiment
consumer expectations are applicable. Compiler execution, message kernels,
BFE, GFE, scheduling, visualization, and plate expansion are N/A.

Duplicated surfaces: current/legacy, encode/decode, bundle/checkpoint,
model-only/runtime-state, direct/facade learnables, tamper/round-trip, and
float32/x64 arrays must agree.

---
id: cff-63dd
status: closed
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

**2026-08-22T18:27:40Z**

Completed by ba67d3f.

Persistence boundary:

- Learnables.state_dict emits a scalar cffg.recurrence.d9-structured-pair/1.0.0 manifest only when it owns recurrence_link PG weights. The manifest binds checkpoint version, exact factor and source ownership, target parameter blobs, hyperparameter names, shapes, canonical dtypes, and SHA-256 digests of exact C-order values.
- Learnables.load_state_dict validates the entire manifest and payload before committing any values. Current f32 and f64 checkpoints round-trip their original dtype and values. Missing identity, D8 identity, unknown versions, incomplete members, wrong factor/source/shape/dtype, non-finite values, and digest tampering fail atomically.
- Pre-manifest recurrence v2 and slash-key checkpoints are rejected because D8 and early D9 arrays cannot be distinguished. No migration relabels them. Legacy migrations for non-recurrence families are unchanged.
- Model-only cffgz bundles and point-parameter files remain loadable. They rerun current D9 inference and are initializations, not resumable recurrence posteriors. The unchanged recurrence_link IR spelling and structure hash cannot prove D8 versus D9 statistic provenance.
- RSLDSLearning marks direct or factory construction as an unlearned prior and marks update, fit output, and validated checkpoint restore as current D9. Numeric fit residual snapshots intentionally exclude the scalar checkpoint manifest.
- RollingAgent already namespaces and round-trips all Learnables.state_dict members. No shipped agent builder owns a recurrence learnable, so no separate agent schema was added.

Validation:

- Focused affected selection: 128 passed, 4 deselected in 138.66 seconds.
- Final exact-head default suite through tools/dashboard.py --run-suite at ba67d3f: 1474 passed, 39 deselected, 0 failed in 584.20 seconds pytest time and 592.714 seconds dashboard wall time.
- git diff --check, Python compilation, and dependency-cycle validation passed.

cff-h0hx handoff:

1. Pin ba67d3f or its integrated descendant and record cffg.recurrence.d9-structured-pair/1.0.0 in every new claim-bearing heterogeneous fit, factor-belief, ladder, and result manifest.
2. Bump and validate custom experiment schemas before generation or replay. Reject missing, D8, unknown, tampered, wrong-source or stick-shape, non-finite, or marginal-inconsistent pair evidence instead of inferring semantics from the CFFG commit or unchanged template hash alone.
3. The heterogeneous fit checkpoint can reconstruct a coefficient posterior from retained prior plus recurrence linear and quadratic evidence. It therefore must bind that evidence to D9, source ownership, dtype, shape, and value digests. The recurrence factor-belief checkpoint must likewise bind its pair beliefs and pair-derived at-risk and signed masses to D9.
4. Historical artifacts remain immutable read-only evidence under their recorded CFFG commits. Do not feed their recurrence arrays or sufficient statistics into current Learnables.load_state_dict or present them as D9.
5. Existing periodicity parameters.npz and other point-only checkpoints may initialize a fresh D9 inference run, but cannot resume learned posterior statistics. If a consumer starts persisting the generic Learnables state, it must retain the full state_dict including the scalar manifest key; filtering to numeric v2 keys is unsafe.
6. Fresh cff-h0hx evidence must use current pair beliefs and statistics and rerun the registered development gates. No historical artifact is rewritten by this ticket.

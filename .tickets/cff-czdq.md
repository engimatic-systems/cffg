---
id: cff-czdq
status: open
deps: [cff-ocxl, cff-f3jz]
links: []
created: 2026-09-14T01:11:24Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-zov5
tags: [rslds, inference, hybrid]
---
# Version hybrid EC learning statistics and persisted state

## Background in Plain English — WHY

D12 recurrence statistics condition state moments on the inferred regime pair and use different auxiliaries from D9. Shape-compatible D9 learning state must not silently resume under that meaning.

## Outcomes / Acceptance Criteria — WHAT

1. Recurrence pair readouts, source-conditioned first/second moments, auxiliaries, and dynamics statistics derive from the returned accepted component belief and objective.
2. Learnables.replace consumes these statistics through declared family contracts, with no model-owned parameter-update route. Replacement and incremental learning semantics remain explicit.
3. Learnable manifest, executable identity, checkpoints, and CFFGZ learning-state identity receive a named semantic version. Successful public D12 CFFGZ save/load preserves declarations, compiled identity and learning state, and resumes the same accepted state. D9/legacy state is rejected or explicitly migrated before mutation; compatible shapes alone never authorize resume. This completes the bundle/executable preservation guarantee whose precompile transport is owned by cff-k6g5 and fixed-parameter reconstruction by cff-7sw5.
4. Current/legacy checkpoint, JSON/hash, plate scoping, member/parameter addresses, lifecycle initialization/binding, readout ownership, Agent/Colony cadence, and bundle surfaces are classified under Learnable or checkpoint feature and Derived component coordinate.
5. Independent learned-increment tests include sign/variance twins, literal zeros, inactive sources, float32, and uncomfortable scale. Single/vmap and host/JIT statistic behavior agree; resume reproduces the same declared state and wrong identity fails transactionally.
6. README documents version/migration and unsupported resume; ARCHITECTURE and D12 link ownership/tests. Focused tests and one final short-suite gate pass.

## Methodological Suggestions — HOW

These suggestions are optional; another sound method meeting the outcomes is welcome.

Preserve Learnables as the posterior replacement authority. Reuse existing semantic-manifest/version gates instead of inferring semantics from array shape.

## Future work

cff-kul2 validates the integrated public composition contract. cff-x0if alone owns observation-only source-recovery qualification.

## Definitions

- D12: the hybrid-regime-ec-v1 mathematical contract in docs/DERIVATIONS.md.
- EC: expectation consistency, matching declared moments between regional beliefs.
- Hybrid separator: one boundary carrying regime probabilities and regime-conditioned state moments.
- Adapter: an explicit equality region connecting a hybrid boundary to an ordinary categorical or Gaussian copy.
- JJ / xi: the Jaakkola–Jordan logistic quadratic bound and its squared auxiliary coordinate.
- FE / ledger: free energy and the single assembly of its owned energy/entropy terms.
- GFE: generalised free energy with P-substitution, excluded from D12.
- IR: serialized intermediate representation of model semantics.
- Arena / member_of: compiled array storage and the map from factors to its members.
- JIT / vmap: JAX compiled and batched execution surfaces.
- CFFGZ: the project bundle format containing template, data, and optional learning state.
- Gauge: redundant numerical coordinate that leaves a belief unchanged.
- Krylov: a matrix-free iterative linear solve using bounded stored vectors.
- D9 / D10: the current factor-local recurrence approximation and the retired whole-model candidate.
- Moment / covariance: an expected feature or product, and its centered second moment.
- Semantic identity: the versioned meaning of a declaration, executable, or persisted statistic.



## Notes

**2026-09-14T11:02:30Z**

Persistence handoff from independent k6g5 research: this ticket explicitly retains the full successful D12 public CFFGZ save/load/resume guarantee, including declarations, compiled identity and learning state. K6G5 owns precompile declaration transport and rejection of unknown identities; 7SW5 owns compiled identity plus fixed-parameter empty-learning bundle reconstruction. Existing CZDQ current/legacy transactional rejection/migration, manifests and checkpoint requirements remain unchanged. Sources bundle.py:181/288/336, learn.py:666, learning_schema.PersistentStateDescriptor/validate_persistent_manifest, core.serialize.legacy_learning_structure_hash and _agent_checkpoint._validate_versioned_envelope. Confidence high; no new mathematical or supported-scope decision.

---
id: cff-k6g5
status: closed
deps: [cff-ocxl]
links: []
created: 2026-09-14T01:11:24Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-zov5
tags: [rslds, inference, hybrid]
---
# Declare serialized hybrid regions and approximation identity

## Background in Plain English — WHY

D12 accepts hybrid-regime-ec-v1 as an explicit approximation. The runtime still executes D9, and no template currently selects the new region ownership. The serialized template must select the mathematics before compilation can derive a plan.

## Outcomes / Acceptance Criteria — WHAT

1. Serialized declarations identify the D12 hybrid scopes, ordinary identity adapters, factor ownership, fixed active-support semantics, and approximation version without inspecting model classes or factor names.
2. Validation rejects unsupported source/pair-selected channels, nonlinear or GFE constraints, overlapping objective owners, and any association-producing factor cut across separate ordinary regime/state ports, naming the offending declaration.
3. JSON round-trip, structure hashing, compact plate scoping, edits/composition, and bundle/executable identity preserve the declaration under renaming and reordering. Existing D9 templates retain their meaning.
4. The Derived component coordinate and New IR field change paths classify validation, serialization, plates, edits/composition, topology, specialization, registration, resolution, assembly, and persistence; runtime host/JIT, flooding/scheduled, and single/vmap are N/A until the execution ticket. Current/legacy semantic identity is applicable.
5. Sparse support, empty declarations, invalid shapes, and support-changing edits are validated; float32 and numerical scale are N/A because this ticket adds no numerical solve.
6. README explains declaration construction and unsupported compositions; ARCHITECTURE records authority. Focused serialization/composition tests and one final short-suite gate pass.

## Methodological Suggestions — HOW

These suggestions are optional; another sound method meeting the outcomes is welcome.

Keep declarations independent of derived array addresses. Apply the existing New IR field path to every semantic field.

## Future work

cff-7sw5 consumes these declarations. Local kernels (cff-ffsg), execution (cff-f3jz), and learned-state migration (cff-czdq) remain separate.

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

**2026-09-15T13:20:19Z**

Closed as not pursuing D12 on this PR142-based branch at Cooper's direction; see cff-ocxl for the deferral summary. Serialized declarations and their transport/identity checks were implemented on the preserved PR151 branch, but that implementation is not incorporated here. This closure does not claim the acceptance criteria are delivered on this branch or activate a D12 runtime.

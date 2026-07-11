---
id: cff-6s31
status: open
deps: [cff-cpxj]
links: [cff-9xmq]
created: 2026-07-09T22:04:18Z
type: feature
priority: 3
assignee: coopwilliams
parent: cff-ahps
---
# Plates phase 2: index-table expansion — compile without materializing copies

Phase 1 (cff-cpxj) materializes plate copies as Python objects at grounding. Phase 2 expands INDEX TABLES instead: copies are structurally identical, so they drop into the existing [M, ...] group tensors by arithmetic — a 64x64 pixel plate should compile in ~1s with an arena exactly as vectorized as phase 1 produces. Also: resource estimator reports cost as an analytic function of dims without expanding (deferred from phase 1). FE-ledger subtleties, if any surface, come back to Fable per the cff-cpxj delegation note.

## Acceptance Criteria

64x64 2-D plate compiles in about a second; compiled arena bit-identical (or provably equivalent: same marginals + BFE) to phase-1 expansion on all four cff-cpxj use cases; estimator analytic in dims.


## Notes

**2026-07-11T18:30:45Z**

Follow-up from cff-vamy/PR #18: phase-1 validation now rejects any PlateSpec whose fragment declares plates, because expand_plates does not recurse and would otherwise ground an empty/incomplete model. When implementing phase-2 index-table grounding, deliberately lift this guard only after recursive plate namespaces, tied/per-copy blob resolution, bundle inventories, arena parity, and FE parity are covered at depth >= 2.

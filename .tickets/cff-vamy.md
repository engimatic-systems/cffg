---
id: cff-vamy
status: open
deps: []
links: [cff-fxmj, cff-cpxj]
created: 2026-07-10T17:57:24Z
type: bug
priority: 0
assignee: coopwilliams
tags: [adversarial-reviewer, serialization, plates]
---
# .cffgz silently drops plate specifications and nested parameter blobs

ADVERSARIAL REVIEWER FINDING (2026-07-10). bundle.save_bundle reconstructs Template at bundle.py:47-58 without copying template.plates, and enumerates only template.blobs. A valid plate is therefore erased from model.json; nested fragment blobs/per-copy blobs are not packaged. The manifest hashes the already-truncated template, so load_bundle verifies and returns the wrong model without warning. Repro: a host with one 3-copy PlateSpec saves successfully; original plates=['leaf'], loaded plates=[], original structure hash 42e1...946d, loaded/manifest hash 478a...f067. This violates the project invariant that serialization is the source of truth and makes the new plate/scalability path undeployable. Practical impact: checkpoint/export/import can silently replace a large model with an empty or incomplete graph while all integrity checks pass.

## Acceptance Criteria

1. Bundle round-trip preserves nested PlateSpec fragments, dims, bonds, per_copy_blobs, and structure_hash. 2. Recursively package every referenced blob (tied and per-copy), rewrite nested paths without collisions, and reject missing arrays before writing. 3. Round-trip a compiled plate model and prove identical arena structure, parameters, observations, marginals, and FE. 4. Add tamper/missing-blob tests for nested fragment and per-copy blobs. 5. Existing flat bundle format remains loadable or schema migration is explicit; never silently truncate unsupported structure.

## Notes

**2026-07-11T17:41:58Z**

CLOSED: ae9c310 preserves the complete PlateSpec hierarchy in model.json and hashes the untruncated template; bundle manifests now inventory all grounded tied/per-copy arrays with collision-free paths, per-copy {index} paths compile directly after extraction, missing arrays are rejected before opening the destination, and nested/per-copy missing or tampered members fail loudly. Original flat schema-1 manifests remain loadable; plate models without the new inventory are explicitly rejected. Acceptance proof: test_bundle compares plate dims/bonds/per_copy_blobs/hash, arena edges/groups/params/data bindings, marginals, and FE; extracted-root compilation also passes. Serialization hostile-input checklist: sparse support, dtype, and scale are N/A because arrays are byte-preserved without numeric transforms; missing/corrupt/inventory-collision cases are guarded. Verification: focused bundle+plate 32 passed; full suite/dashboard 1112 passed, 4 dependency warnings, 0 failures.

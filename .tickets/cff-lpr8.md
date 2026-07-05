---
id: cff-lpr8
status: closed
deps: [cff-q0ep]
links: []
created: 2026-07-05T22:12:45Z
type: task
priority: 0
assignee: coopwilliams
parent: cff-br53
---
# Canonical serialization + structure hash

to_dict/from_dict, JSON round-trip, sha256 structure hash excluding blob paths (JIT-cache key). Hash stable across blob relocation, sensitive to structural change.


## Notes

**2026-07-05T22:15:56Z**

Landed serialize.py: canonical sorted dict/JSON forms, file save/load, schema-version gate, structure_hash = sha256 with blob paths erased (names kept) so JIT-cache keys survive blob moves. Order-insensitivity + hash sensitivity tests. 22 tests total.

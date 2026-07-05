---
id: cff-lpr8
status: in_progress
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


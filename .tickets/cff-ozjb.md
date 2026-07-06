---
id: cff-ozjb
status: closed
deps: []
links: []
created: 2026-07-06T02:45:04Z
type: task
priority: 0
assignee: coopwilliams
parent: cff-xjlh
---
# .cffgz bundle: save/load with integrity manifest

cffg/bundle.py: zip of model.json (canonical serialization, blob paths rewritten to blobs/<name>.npy), blobs/*.npy, manifest.json (schema version, structure hash, per-blob sha256, timestamps). Load verifies hashes + structure hash; tamper -> loud error. Round-trip test: compile from loaded bundle reproduces marginals exactly.


## Notes

**2026-07-06T02:49:57Z**

Landed bundle.py: zip(model.json canonical + blobs/*.npy + manifest.json w/ structure hash + per-blob sha256). Load verifies everything; tamper + hash-mismatch + missing-array tests. Round-trip reproduces T-maze marginals to 1e-12.

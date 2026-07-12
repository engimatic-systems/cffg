---
id: cff-9xmq
status: open
deps: []
links: [cff-6s31]
created: 2026-07-10T03:30:52Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Plate Broadcast: balanced equality tree — O(log N) BP diameter on hub fan-outs

Broadcast expansion synthesizes a LINEAR equality chain, so flooding diameter grows O(N) in copy count: a 49-leaf hub needs ~70 sweeps to converge and sweep count (not per-sweep cost, not compile — a 784-leaf plate compiles in 1.2s/1.3GB) is what caps inference throughput on plate perception models (measured ~5 items/s where ~20 was the bar). Expand Broadcast to a balanced ternary equality tree instead: O(log N) diameter, same fixed point (equality is exact). Also applies to any future fan-out primitive.

## Acceptance Criteria

Balanced-tree expansion behind the same Broadcast spec (serialization unchanged); sweeps-to-convergence on a 49-leaf hub drops from ~70 to <=12; marginals identical to chain expansion at 1e-10; existing plate tests unchanged.


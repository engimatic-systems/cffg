---
id: cff-domo
status: in_progress
deps: []
links: []
created: 2026-07-05T22:44:10Z
type: task
priority: 0
assignee: coopwilliams
parent: cff-rlr2
---
# Compiler: grounding + PGMax-style arena layout + blob loading

Validated template -> flat log-space message arena [n_slots, K_max] with padding masks; factors grouped by (type, cardinality signature) with numpy gather/scatter index tables; blob loading (.npy/.safetensors or in-memory override) with stochasticity shape checks; port policy: DATA+GLUE compile (glue = FFG half-edge, implicit marginalization), PRIOR/GOAL rejected; param edges rejected until M3.


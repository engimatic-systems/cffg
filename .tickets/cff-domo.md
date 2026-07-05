---
id: cff-domo
status: closed
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


## Notes

**2026-07-05T22:49:06Z**

Landed compiler.py: [n_slots,k_max] log arena, NEG=-1e30 padding, groups keyed (type,cards) with numpy gather/scatter tables; blob loading .npy/.safetensors/in-memory; stochasticity+shape checks; DATA+GLUE ports compile (glue=half-edge/implicit marginalization), PRIOR/GOAL rejected; param edges deferred to M3.

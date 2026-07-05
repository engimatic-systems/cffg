---
id: cff-1zat
status: open
deps: [cff-alvf]
links: []
created: 2026-07-05T22:44:11Z
type: task
priority: 0
assignee: coopwilliams
parent: cff-rlr2
---
# Test tiers: tree-CSP enumeration parity, HMM forward-backward parity, vmap batch

Tree CSP: marginals vs exhaustive enumeration exactly. HMM T=4 via unroll+compose: smoothed marginals vs numpy alpha-beta, BFE vs -loglik, partial observations. eqx.filter_vmap batch == sequential.


---
id: cff-1zat
status: closed
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


## Notes

**2026-07-05T22:49:06Z**

Tiers landed: tree-CSP enumeration parity incl missing-data marginalization; HMM T=4 forward-backward parity (marginals 1e-10, BFE=-loglik 1e-9, partial obs); eqx.filter_vmap batch == sequential at 1e-10. x64 enabled in conftest. 48 tests total.

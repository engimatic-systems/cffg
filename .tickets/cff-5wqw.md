---
id: cff-5wqw
status: closed
deps: [cff-domo]
links: []
created: 2026-07-05T22:44:10Z
type: task
priority: 0
assignee: coopwilliams
parent: cff-rlr2
---
# BP rules + flooding engine (step/run/bind/marginals)

Log-space sum-product updates for categorical/matrix_cat/transition_mixture/equality, vectorized per group; jitted synchronous flooding step, params as inputs (colony contract); bind_data clamps; marginal extraction. Unit tests incl TM vs numpy einsum reference.


## Notes

**2026-07-05T22:49:06Z**

Landed rules.py+engine.py: log-space BP for categorical/matrix_cat/TM/equality vectorized per group; Engine with eqx.filter_jit step, State(messages,params) eqx.Module, params as inputs (colony contract verified by vmap test); run() returns delta trace as coherence signal; bind_data clamps; TM unit-tested vs einsum.

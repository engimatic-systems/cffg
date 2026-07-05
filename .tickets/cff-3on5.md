---
id: cff-3on5
status: closed
deps: [cff-q0ep]
links: []
created: 2026-07-05T22:12:45Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-br53
---
# Template composition + unrolling via glue ports

compose(left,right,bonds) with namespacing + cardinality checks, unroll(template,n,bonds) chaining copies. HMM-slice fixture unrolls and validates.


## Notes

**2026-07-05T22:18:52Z**

Landed compose.py: scoped() namespacing with '/', merge (clash-guarded), bond (port unification w/ cardinality check), compose(), unroll() = repeated self-composition along glue ports. Tests: 3-step unroll counts, determinism via structure hash, per-copy constraint retargeting, prior-onto-chain composition, closure (chain composes with chain). 38 tests. M1 epic complete.

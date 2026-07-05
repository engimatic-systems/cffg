---
id: cff-rlr2
status: open
deps: [cff-br53]
links: []
created: 2026-07-05T22:12:32Z
type: epic
priority: 1
assignee: coopwilliams
---
# M2: Discrete BP/VMP engine on arenas — CSP + HMM tiers

Compiler (grounding + PGMax-style arena layout), rule registry with BP/VMP for Cat/Dir/equality/TM, flooding VM with FE ledger. Acceptance: BP marginals match enumeration on CSP tier; smoothing matches exact forward-backward on HMM tier; step is vmap-compatible (Equinox state, filter_jit).


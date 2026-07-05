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


## Notes

**2026-07-05T22:49:06Z**

M2 core complete (compiler/rules/engine/FE/tiers, 4 tickets closed). Epic stays OPEN for cff-kv7p (VMP mean-field bead) — deliberately deferred to just before M3 since GFE nodes reduce to VMP for data-constrained observations (Part II Fig.3). Next session: start cff-kv7p, then M3.

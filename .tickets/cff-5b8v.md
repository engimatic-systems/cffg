---
id: cff-5b8v
status: closed
deps: [cff-iovq]
links: []
created: 2026-07-05T23:23:06Z
type: task
priority: 0
assignee: coopwilliams
parent: cff-1d6t
---
# Canonical T-maze model + qualitative epistemic tests

src/cffg/models/tmaze.py: build_tmaze(steps, alpha, c_util, gfe) with Part II layout P=(O,L,R,C), state pos(x)ctx 8, obs pos(x)outcome 16, B_u=P_u kron I2, A blockdiag, c=softmax(util), d=e_O kron (.5,.5). Tests: GFE agent argmax u1=C (cue); u2 L/R exactly symmetric (model symmetry invariance) and > O,C; BFE ablation prefers arms not cue; delta variant one-hot valid; converged BFE finite.


## Notes

**2026-07-05T23:32:49Z**

Landed models/tmaze.py (Part II layout, ctx-fastest kron convention matching helpers.jl) + 7 tests: cue-first argmax, exact L/R symmetry invariant (1e-9), context marginals 50/50, BFE ablation prefers arms + strictly lower cue value, delta MAP variant (C then arm, one-hot), horizon-3, GFE stability.

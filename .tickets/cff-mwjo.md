---
id: cff-mwjo
status: closed
deps: [cff-rlr2, cff-4oq8, cff-gjjk, cff-iwgl]
links: []
created: 2026-07-05T22:12:32Z
type: epic
priority: 1
assignee: coopwilliams
---
# M3: GFE composite node — Part II Fig.5 messages, Newton, IS

## Plain English

M3 delivers the Part II GFE composite as executable, testable inference
machinery. It includes the goal message, Newton-solved backward-state message,
importance-sampled parameter message, and exact free-energy accounting. Those
original milestone requirements are complete.

## Scope

P-substituted goal-observation composite: Dirichlet goal message, Newton-solved backward state message (Corollary 1), importance-sampled A message with explicit PRNG keys. Acceptance: unit tests vs hand-computed values; FE ledger exact.

## Notes

**2026-07-30T14:06:21Z**

2026-07-30 roadmap closeout. The original M3 scope is complete: the Fig. 5 composite, Dirichlet goal message, Newton-solved backward-state message, importance-sampled parameter message, explicit PRNG handling, and exact FE ledger are implemented and tested. cff-2gev is no longer an M3 child or dependency because its redesigned continual-curiosity study is later research, not unfinished M3 work. The numerical issue found by the invalid experiment is independently tracked in cff-weff.

Legend: prerequisite --> dependent; ... means linked only

[x] cff-rlr2 --+
[x] cff-4oq8 --+
[x] cff-gjjk --+--> [x] cff-mwjo (M3 closeout)
[x] cff-iwgl --+

Ready parallel follow-ups, neither an M3 blocker:
[ ] cff-weff (P2 ready) ... [ ] cff-2gev (P3 ready)
[ ] cff-l6i5 (P2 ready) --> [ ] cff-ahps (roadmap epic; other blockers remain)

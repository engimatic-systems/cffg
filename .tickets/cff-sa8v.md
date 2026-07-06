---
id: cff-sa8v
status: closed
deps: [cff-dn5y]
links: []
created: 2026-07-06T01:29:37Z
type: task
priority: 0
assignee: coopwilliams
parent: cff-olvl
---
# RollingAgent: infinite-horizon receding loop + checkpointing

cffg/agent.py: two engines total regardless of episode length — plan(past=0,steps=H) + filter(past=1,steps=1). act: plan from current belief d -> MAP/sample control; observe: filter update re-anchors d (the constraint flip + slide). Optional Dirichlet A-learning per tick (counts -> elog/counts param swap). Checkpoint = belief + counts + tick to npz; restore resumes identically.


## Notes

**2026-07-06T01:37:12Z**

Landed agent.py RollingAgent: plan(0,H)+filter(1,1) engines, belief re-anchor per tick via member_of param update (zero recompiles verified across 10-tick episodes), MAP/sample selection, optional per-tick Dirichlet A-learning, npz checkpoint/restore resuming identical actions.

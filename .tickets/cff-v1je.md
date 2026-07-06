---
id: cff-v1je
status: closed
deps: [cff-sa8v]
links: []
created: 2026-07-06T01:29:37Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-olvl
---
# Gridworld corridor: second model proves the pattern generalizes

models/gridworld.py via build_rollout: K cells, L/S/R shifts, noisy-identity A, goal prior on far cell. Tests: agent walks to goal within budget, belief tracks true position, checkpoint round-trip resumes identically.


## Notes

**2026-07-06T01:37:12Z**

Landed gridworld.py (corridor via build_rollout — second client, zero new engine code) + CorridorEnv. Agent walks 0->5 optimally (RIGHT x5 then STAY), belief tracks true position 7/8 ticks under 0.9-noise obs.

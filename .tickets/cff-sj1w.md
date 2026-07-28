---
id: cff-sj1w
status: closed
deps: [cff-dn5y]
links: [cff-rqqc]
created: 2026-07-06T01:29:37Z
type: task
priority: 0
assignee: coopwilliams
parent: cff-olvl
---
# Colony: instance table, structure-hash batching, vmapped tick

cffg/colony.py: spawn(template, arrays) caches Engine by structure_hash; instances = pytree states; run() stacks per group and executes one vmapped step-loop per unique structure; kill() shrinks batch. Tests: batched == sequential 1e-10; mixed structures -> engine cache size == n unique structures; spawn/kill integrity.


## Notes

**2026-07-06T01:37:12Z**

Landed colony.py: Engine cache keyed by structure_hash (5 instances 2 structures = 2 compiles), per-structure vmapped step loops, batched==sequential 1e-10, kill() integrity, host port routing (same graph different evidence different beliefs), T-maze golden posture reproduced inside a mixed colony.

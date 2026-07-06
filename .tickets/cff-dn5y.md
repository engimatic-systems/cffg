---
id: cff-dn5y
status: closed
deps: []
links: []
created: 2026-07-06T01:29:37Z
type: task
priority: 0
assignee: coopwilliams
parent: cff-olvl
---
# Generic POMDP rollout builder + member-targeted param updates

Extract build_tmaze's slice pattern into models/rollout.py build_rollout(name, steps, past, d/A/c/e/Bs, gfe, map_policy, learned_a) — tmaze delegates, golden tests must stay green. Compiler: Arena.member_of[factor] -> (group, idx) so runtime can swap a single member's params (same-cardinality categoricals share groups; whole-group broadcast would clobber siblings).


## Notes

**2026-07-06T01:37:12Z**

Landed rollout.py (generic POMDP window; tmaze delegates, LAIF golden parity intact) + Arena.member_of for member-targeted param swaps (same-K categorical groups would clobber siblings otherwise — gridworld's d/c collision made this real).

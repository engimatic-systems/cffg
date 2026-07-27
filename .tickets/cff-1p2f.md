---
id: cff-1p2f
status: closed
deps: [cff-d56d]
links: []
created: 2026-07-27T05:44:29Z
type: feature
priority: 3
assignee: coopwilliams
tags: [scheduling]
---
# Schedule extensions: iterative residual/colored schedules for loops; EF directed passes

cff-d56d ships exact two-pass schedules for acyclic discrete/Gaussian BP subgraphs; everything else floods (Schedule.reason records why). Two extensions remain from its first acceptance criterion: (1) cyclic BP graphs currently iterate synchronous flooding - a residual or colored (Gauss-Seidel) iterative schedule converges in fewer sweeps on many loopy models; (2) EF conjugate-BP subgraphs are tree-exact and deserve the same directed two-pass treatment (kernels already dispatch per group).

## Acceptance Criteria

Loopy discrete models offer an opt-in sequential/residual schedule with documented convergence semantics and parity at the fixed point; EF trees run two-pass with conjugate-BP exactness tests; flooding remains the default where semantics require it; boundaries documented in ARCHITECTURE.md.


## Notes

**2026-07-27T14:13:01Z**

Absorbed into cff-d56d after the 2026-07-27 review: colored Gauss-Seidel schedules for loopy pure-BP components and directed two-pass execution for EF conjugate-BP groups both landed on the cff-d56d branch (with parity tests), together with block-phased mixed scheduling. Nothing remains in this ticket's scope.

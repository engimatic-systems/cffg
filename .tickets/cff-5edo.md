---
id: cff-5edo
status: open
deps: [cff-xdip, cff-8whp]
links: [cff-fxmj, cff-j4mu, cff-i45j, cff-wbhu, cff-d56d]
created: 2026-07-10T03:31:09Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-cfqg
---
# Milestone: Truthful and reusable long-rollout planning

## Background in Plain English

The recorded horizon-25 failure had two separable concerns. First,
`RollingAgent` used a fixed sweep cap and could not expose a useful plan unless
that cap happened to be enough for the compiled depth. Second, every new
environment step discarded the converged work from overlapping planning
slices. The first concern is a correctness and status gate. The second is a
performance optimization.

This milestone now closes when `cff-xdip` makes the action path depth-aware and
status-gated, then `cff-8whp` proves a compatible warm start with a measured
speedup. The ordering prevents warm-start timing from hiding an unfinished
plan.

## Outcomes / Acceptance Criteria

1. `cff-xdip` is closed with depth-derived budgets, action gating, public plan
   diagnostics, hostile-input behavior, and the horizon-25 behavioral check.
2. `cff-8whp` is closed with warm/cold result parity, cache invalidation,
   bounded storage, hostile-input behavior, and the three-times steady-state
   speed threshold.
3. README, architecture, and limitations documentation distinguish convergence
   correctness from warm-start performance and record the measured envelope of
   each.
4. Completion does not weaken BFE/GFE, exact support, message coherence, inner
   solver, stochastic status, or public action-selection semantics.

## Methodological Suggestions

These suggestions are optional. An implementer may choose another sound method
that satisfies the numbered outcomes with equivalent evidence.

- Land `cff-xdip` first and preserve its cold-planning benchmark as the
  baseline for `cff-8whp`.
- Keep the warm cache derived and disposable so persistence correctness does
  not depend on an optimization.

```text
Legend: prerequisite --> dependent

[ ] cff-wjqg ordinary compiled loop
        |
        +--> [ ] cff-xdip status/depth gate
                    |
                    +--> [ ] cff-8whp warm start
                                |
                                +--> [ ] cff-5edo
```

## Future work

`cff-yl0a` owns slice-fused chain kernels for the remaining specialized-chain
throughput gap. `cff-t2g3` consumes the completed planning milestone for the
glucose controller. No other work is absorbed here.

## Definitions

- **Depth-aware planning:** Selecting a finite inference budget from compiled
  propagation depth and checking the resulting status.
- **Long rollout:** A planning window long enough that a fixed small sweep
  budget cannot propagate information through the full graph.
- **Warm start:** Reusing compatible derived state from the previous planning
  window.

## Notes

**2026-08-13T04:37:28Z**

Roadmap split, 2026-08-12. Legend: prerequisite --> dependent.

[ ] wjqg (ready) --> [ ] xdip (status/depth) --> [ ] 8whp (warm start) --> [ ] 5edo

The action correctness gate lands before the warm-start speed claim.

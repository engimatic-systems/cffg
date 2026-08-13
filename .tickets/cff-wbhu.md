---
id: cff-wbhu
status: open
deps: [cff-wjqg, cff-0fw2]
links: [cff-fxmj, cff-5edo, cff-d56d, cff-yl0a, cff-rqqc, cff-weff, cff-lv2w, cff-g7rc, cff-r6bl]
created: 2026-07-10T03:34:54Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-cfqg
---
# Milestone: Compiled Engine and Colony control loops

## Background in Plain English

The original ticket combined two distinct controllers: ordinary iterative
message passing and the deterministic mixed-GFE trust-region solve. Both suffer
from Python-per-iteration dispatch, but they have different state, numerical
proof obligations, and review surfaces. The 2026-08-09 trajectory survey
recommended splitting them if they could not fit one reviewable change.

This milestone now closes when `cff-wjqg` compiles ordinary Engine and Colony
iteration and `cff-0fw2` separately compiles the mixed-GFE controller. It keeps
the historical links and the common claim: repeated inference control and
per-instance status live on the device without changing public results.

## Outcomes / Acceptance Criteria

1. `cff-wjqg` is closed with its ordinary compiled-loop, status-isolation,
   hostile-input, parity, and eight-times Colony throughput evidence satisfied.
2. `cff-0fw2` is closed with its mixed-GFE compiled-controller, certificate,
   per-instance isolation, parity, and benchmark evidence satisfied.
3. Public Engine and Colony calls, results, statuses, traces, schedule
   fingerprints, serialized structures, and checkpoint schemas retain their
   documented meaning across both children.
4. Architecture and limitations documentation distinguish ordinary iteration,
   deterministic mixed-GFE control, and stochastic `mu3` batching.
5. Completion claims use the measured CPU workloads and do not imply universal
   accelerator scaling or specialized-chain parity.

## Methodological Suggestions

These suggestions are optional. An implementer may choose another sound method
that satisfies the numbered outcomes with equivalent evidence.

- Review and land `cff-wjqg` before `cff-0fw2`. The second child should consume
  the first child's device-side result and status conventions.
- Preserve the historical benchmark manifests so before-and-after throughput
  comparisons use the same workloads.

```text
Legend: prerequisite --> dependent

[x] j4mu + d56d + 44we --> [ ] cff-wjqg (ordinary loop)
                                      |
[x] lv2w -----------------------------+--> [ ] cff-0fw2 (mixed GFE)
                                                  |
                                                  +--> [ ] cff-wbhu

[ ] cff-wjqg --> [ ] cff-xdip --> [ ] cff-8whp --> [ ] cff-5edo
```

## Future work

`cff-rqqc` owns per-instance random keys and stochastic `mu3` state in Colony.
`cff-yl0a` owns slice-fused chain kernels. `cff-g7rc` owns tiny mixed-GFE cold
and fixed-certification overhead. None is absorbed into this milestone.

## Definitions

- **Compiled controller:** Device-side loop control around existing message or
  solver kernels.
- **Mixed GFE:** A Generalised Free Energy graph whose stationary coordinates
  use the trust-region outer solver.
- **Ordinary iteration:** Deterministic iterative message passing outside the
  mixed-GFE trust-region controller.

## Notes

**2026-08-13T04:37:28Z**

Roadmap split, 2026-08-12. Legend: prerequisite --> dependent.

[ ] wjqg (ready, ordinary loop) --> [ ] 0fw2 (mixed GFE) --> [ ] wbhu

The ordinary loop can land and unblock depth-aware planning before the mixed-GFE controller is complete.

---
id: csp-t7n3
status: closed
deps: []
links: []
created: 2026-11-28T06:34:04Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-g7rc
---
# Compile identical mixed-schedule propagation passes only once

A mixed model can use the same ordered message updates twice, once to prepare its initial state and once to reconstruct its forest of ordinary factors. The T-maze and D1 cold traces show that JAX compiles a fresh copy for each use. A model containing an ordinary cycle can require different passes.

## Design

Use the ScheduleProgram owner in the runtime change-path row. Both host and compiled trust-region controllers consume this owner. Compare the entire immutable stage sequence before sharing the compiled function. Flood, colored, exact two-pass, numeric kernels, batch transport and serialization require no changes.

## Acceptance Criteria

The equal-stage case uses one compiled callable and the unequal-stage case retains both behaviors. Compare inputs, native outcomes, cold compile counts and repeated timings on the two retained cases. Report warm costs separately. The broader solver target stays with cff-g7rc.


## Notes

**2026-11-28T13:56:04Z**

The mixed-schedule runtime can prepare its starting state and reconstruct its forest with identical ordered message updates. It now shares a compiled callable when the stages match. Different stages keep their own propagation functions.

Five fresh processes per fixture, controller and revision and five warm decisions per process gave identical native outcomes for each comparison. Cold XLA compilations of the propagation function fell from two to one in all 20 pairs. Medians on the M2 Max CPU with float64 follow.

| Fixture | Controller | Cold seconds | Warm milliseconds |
| --- | --- | ---: | ---: |
| tmaze | host | 2.876 → 2.620 | 82.39 → 81.39 |
| tmaze | compiled | 3.451 → 3.210 | 13.50 → 13.62 |
| d1 | host | 2.243 → 2.083 | 32.03 → 31.31 |
| d1 | compiled | 2.673 → 2.500 | 5.63 → 5.70 |

The broader cold-start target remains open in cff-g7rc. This work stacks on #236.

The required short suite passed 2,845 cases in 436.624 seconds at cc6b66b15b90.

---
id: cctr-1vvy
status: in_progress
deps: []
links: []
created: 2026-11-28T06:39:11Z
type: task
priority: 2
assignee: coopwilliams
parent: ctcp-w305
---
# Qualify the changing-column projection on masked categorical counts

The stationary factored planner has a native parameter-novelty objective. To make old evidence lose force under a declared reset hazard, the learner needs one bounded update per committed transition event. Verify the D1a projection and its represented support before a live learner consumes it.

## Design

Use the immutable reset concentrations, committed concentrations, allowed coordinates, scalar hazard and pooled observed or fractional counts. Produce a new concentration vector and explicit validity. Exercise eager CPU, JIT and batched columns against the independent D1a history and quadrature oracles. Integration, checkpoint codecs and study outcomes remain with the parent tickets.

## Acceptance Criteria

Check both hazard endpoints, unvisited and singleton columns, missing and fractional observations, correlated histories, exact excluded entries, and positive support on allowed unseen outcomes. Invalid inputs or unrepresentable outputs must preserve the committed state and report failure. Compare true float32 and float64 against independent reference values including a saturated history and at least ten times the fixture size. State the time and space cost per column; no helper may update the carry a second time during planning.


## Notes

**2026-11-28T06:39:29Z**

Runtime numerical or status invariant. Arithmetic is deterministic with no solver iterations. Freeze tests before integration: ordered and fractional oracle traces at unit scale compare with 128 dtype eps absolute/relative, repeated projections with 512 eps. Compare the stationary endpoints directly in the input dtype. Extend to rare allowed outcomes, structural zeros, one active entry and saturated concentration totals without inferring support from tiny positive values. No FE or planner score changes in this child.

**2026-11-28T06:42:48Z**

Initial independent float64 saturation comparison failed: at a shape 12 and c=0.1 the available JAX betaln differed from SciPy by about 2.6e-8 even in double. Replaced that attempted ratio with a rearranged Stirling log-Gamma difference for large concentration and exact one/zero-count shortcuts. Independent integration of digamma across the increment checks large-argument differences. Four true f32/f64 eager/JIT/vmapped tests now pass, including 80x6 concentrations above 100,000. Both branch masses and the between-branch variance enter the bounded projection.

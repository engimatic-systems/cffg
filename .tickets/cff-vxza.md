---
id: cff-vxza
status: in_progress
deps: []
links: []
created: 2026-09-25T07:25:11Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-ectw
tags: [pcgym, numerics]
---
# Preserve weak information in additive equality cavity messages

## Background in Plain English

The PCGym hybrid replay exposed cancellation in Gaussian equality messages. Adding every incoming precision and then subtracting the recipient can erase information that should be sent to that recipient. The same arithmetic is used for exponential-family natural parameters. For example, a strong incoming value of 1e18 plus a weak value of 1 rounds back to 1e18 in float64; subtracting the strong value incorrectly sends zero instead of one.

This is an independent amplifier of the hybrid defect, not its complete repair.

## Outcomes / Acceptance Criteria

1. Gaussian eta/precision and exponential-family equality outputs sum only the other legs, without total-minus-self cancellation.
2. Independent direct-exclusion oracles cover dominant/weak messages, exact vacuous zeros, mixed-sign natural parameters, float32 and float64, and at least 100 incoming legs.
3. Host/JIT and single/vmap results agree within dtype-scaled arithmetic tolerances, and Jacobians retain zero self-dependence and unit dependence on every other leg.
4. Nonfinite values on other legs remain nonfinite; the routine adds no ridge, floor or invented precision.
5. Existing schedules, posterior/free-energy checks and sparse-support coverage pass. The implementation does not allocate an arity-squared tensor.

## Methodological Suggestions

Choose another sound method if it meets the outcomes. Follow the Runtime numerical or status invariant row of docs/ARCHITECTURE.md. Before implementation, list local kernels, host/JIT, flooding/scheduled/exact, Engine/Colony and single/vmap surfaces; checkpoints are N/A unless stored state changes. Keep numerical guards relative to dtype and scale. Continue the validated PCGym PR chain, update the intended base before branching, and run focused tests plus one final short suite.

An exclusive prefix/suffix sum is one possible linear-storage method. Natural-parameter arithmetic is the authority; do not alter objective or convergence policy.

## Future work

- cff-ectw still requires mixture rank/collapse repair and a safeguarded initialization/default-solver path before its planning failure is resolved.

## Definitions

A cavity message excludes information from its recipient. Eta and precision are Gaussian natural parameters. A natural parameter is an additive coordinate of an exponential-family distribution. A Jacobian records how outputs change with inputs. Arity is the number of incoming legs. A vacuous message contributes no information. JIT is compiled execution; vmap batches independent instances; dtype is numeric storage precision.



## Notes

**2026-09-25T07:26:43Z**

Pre-implementation handoff: Runtime numerical/status invariant row plus New factor/rule family mathematical-message validation. Local owners: gauss.equality_messages and efrules.equality_messages; dispatched by kernel_runtime.gaussian_outputs and execution.registered, hence shared by flood/scheduled/exact Engine and Colony. Use one private additive exclusion helper, not separate arithmetic fixes. Host/JIT, single/vmap and derivative paths required. Current/legacy checkpoints N/A (no schema/state change); scalar objective/energy equations unchanged. Hostile inputs preregistered: exact zeros preserved, weak positive information beside a dominant recipient must survive; mixed-sign eta; other-leg nonfinite remains nonfinite; f32/f64; arity300 and linear-storage implementation. Direct NumPy exclusion and analytic Jacobian are independent oracles.

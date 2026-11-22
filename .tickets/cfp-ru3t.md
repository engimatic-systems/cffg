---
id: cfp-ru3t
status: closed
deps: [cfp-hlqd]
links: []
created: 2026-11-22T01:00:18Z
type: task
priority: 1
assignee: coopwilliams
parent: cfp-okq2
tags: [scheduling, performance]
---
# Execute recognized nonlinear chains with checked directional Gaussian messages

## Background in Plain English — WHY

Once the chain is recognized, its transition coefficients can be computed in forward order and its goal information propagated backward. Produce a normal engine state with fresh inputs, failure reporting and the frozen-surrogate objective.

## Outcomes / Acceptance Criteria — WHAT

1. A runnable engine path consumes the recognized ordering, current parameters, mask and goal data, and populates the same directed message slots and public readouts as the existing path.
2. Compare all returned messages, proper marginals, raw/canonical successor energy, residual, action and local statuses with the directed reference and the coupled solver on matched fixtures.
3. Cover float32/64, vacuous priors, partially informed goals, invalid noise, nonfinite physics and 10–100× fixture horizon or dimension. Specify where the route fails explicitly or delegates to the coupled solver.
4. Respect budgets, explicit schedules, and existing status meanings. Document or conservatively exclude unsupported Colony and differentiation paths; no silent numerical or objective change.

## Methodological Suggestions — HOW

Architecture row: Runtime numerical or status invariant. Trace schedule_program, engine, run_results, host/compiled checks, mixed/flood/exact interfaces, ledger and readouts; compare single/vmap and inference/gradient paths where admitted. Runtime state is derived from the template and live arrays. No new checkpoint format is proposed.

## Future work

cfp-okq2 owns automatic dispatch, public documentation and matched cold/warm qualification. The reactor control study proceeds independently.

## Definitions

A directed message carries one neighbor’s contribution to a belief. A frozen surrogate is the Gaussian model evaluated at the expansion points selected by the input cavities.

## Notes

**2026-11-22T19:48:07Z**

Architecture path: runtime numerical or status invariant. CPU automatic single-state inference assembles live Gaussian messages then checks the synchronous local updates and the normalized ledger. Explicit schedules and coupled trust-region retain stock behavior; Colony adaptive requests delegate to individual engines; gradients use existing pure JAX maps.

**2026-11-22T19:56:08Z**

Integrated the two passes, independent synchronous candidate/successor ledger certificate, method-specific diagnostics, fallback budget and per-instance Colony route. Focused checks: 18 passed across recognizer, corrupted-candidate, runtime, named-action readout, independent dense Gaussian, float32/64, invalid factor and live-binding cases. Cross-window and full integration gates tracked on cfp-okq2.

**2026-11-22T20:53:49Z**

Production gate passed at 8c4993a. Default Engine.run solved all 599 retained/held-out PCGym windows in one directed attempt, plus H194 and fresh/warm stock-map checks for both dtypes on the H2 model. Full numerical gate: 2755 passed, 194 opt-in cases deselected. See cfp-okq2 for timing boundaries and the post-gate suite-budget selection.

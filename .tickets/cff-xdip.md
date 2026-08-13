---
id: cff-xdip
status: open
deps: [cff-wjqg]
links: []
created: 2026-08-13T04:33:03Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-5edo
tags: [planning, status, trustworthiness]
---
# Gate RollingAgent actions on depth-aware converged plans

## Background in Plain English

The long-rollout failure recorded by the project was an under-convergence
failure, not a bad policy objective. At horizon 25, a fixed 120-sweep budget
left residual 5.5; 400 sweeps reduced it to about 4.9e-4. `Engine.run` now has
truthful statuses, but `RollingAgent.act` still uses one configured sweep cap
for every planning depth and immediately requires convergence without exposing
the planning result as a public diagnostic.

This ticket fixes the correctness boundary first. It derives a finite planning
budget from compiled propagation depth, refuses to return an action on an
unfinished plan, and records the measured long-horizon status envelope. It
does not warm-start the next window; `cff-8whp` owns that optimization.

## Outcomes / Acceptance Criteria

1. `cff-wjqg` is closed with the compiled ordinary loop and truthful status
   carry available to planning.
2. `RollingAgent` derives its default planning cap from the compiled maximum
   scheduled message-path length with documented headroom for iterative GFE
   coordinates.
3. A caller can override the cap and tolerance explicitly. The default is not
   one horizon-independent sweep count.
4. `RollingAgent.act` returns an action only after a `CONVERGED` planning
   result.
5. Budget exhaustion, divergence, and infeasibility raise or return their
   distinct public failure and cannot look like a successful action.
6. The agent exposes its last planning `RunResult`, including status,
   iterations, residual, free-energy trace, and active inner or stochastic
   diagnostics.
7. Planning through flooding, scheduled, exact, deterministic GFE, and any
   prepared stochastic GFE path uses `Engine.run` and the same status gate.
8. The recorded aliased-corridor protocol reports horizon, seeds, action rule,
   cap, tolerance, status, residual, free energies, success, steps, compile
   time, first execution, steady planning, and readout time.
9. On at least 100 paired episodes, the horizon-25 success estimate is no more
   than 0.05 below horizon 20, and a paired 95-percent bootstrap interval for
   their difference includes zero.
10. Hostile coverage includes literal zero support, truly empty support,
    vacuous but proper boundaries, float32, horizon 25, and a horizon-50
    scale-survival case. A truthful budget-exhausted horizon-50 result is
    acceptable; an action from that result is not.
11. The implementation does not weaken free-energy convergence, message
    coherence, exact support, inner-solver status, or action-selection
    semantics to pass the behavioral threshold.
12. Architecture, README, and limitations documents state the depth-aware
    budget, action gate, public diagnostic, and measured cold-planning
    envelope without claiming warm-start speedup.

## Methodological Suggestions

These suggestions are optional. An implementer may choose another sound method
that satisfies the numbered outcomes with equivalent evidence.

- Add a compiled structural depth readout or derive it from the authoritative
  schedule; do not infer depth from model-specific names.
- Keep the action gate next to `RollingAgent.act` and use
  `RunResult.require_converged()` as the public status boundary.
- Put the multi-episode behavior in a slow test and keep focused status/depth
  cases in the default suite.
- Use the **Runtime numerical or status invariant** change path. Walk
  flooding/scheduled and deterministic/stochastic planning paths.

## Future work

`cff-8whp` warm-starts overlapping planning windows after this correctness gate
lands. Slice-fused chain performance remains `cff-yl0a`. The glucose controller
remains downstream of the completed `cff-5edo` umbrella milestone.

## Definitions

- **Compiled plan depth:** The longest scheduled message path that constrains
  how many updates information needs to reach an action.
- **Cold plan:** A plan initialized without message state from the prior
  environment step.
- **Planning cap:** The finite maximum number of logical updates allowed in one
  planning attempt.
- **Status gate:** The rule that an action is available only after a successful
  converged result.

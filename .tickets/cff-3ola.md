---
id: cff-3ola
status: open
deps: [cff-gl89]
links: [cff-ws3c, cff-3kdk]
created: 2026-09-02T04:02:01Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-ws3c
tags: [engine, architecture, readout, refactor]
---
# Extract readouts and slim public Engine orchestration

## Background in Plain English — WHY

After execution, schedules, and objective accounting have focused owners, the
public engine still contains many readouts. These include discrete and Gaussian
marginals, node beliefs, recurrence pairs and statistics, mixture predictions,
exponential-family marginals, kernel readouts, and goal or parameter
diagnostics. Several readouts repeat plan and parameter selection.

This ticket gives readouts focused owners and leaves the public `Engine` as a
small orchestration facade. It preserves algorithms and output shapes.
Readout vectorization remains separate work in `cff-3kdk`.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-gl89` is closed with the focused objective owner passing its
   independent oracles.
2. Focused readout owners cover every engine readout documented by the
   `cff-sf29` baseline, including discrete, Gaussian, exponential-family,
   mixture, recurrence, goal, kernel, learning-statistic, GFE, and
   functional-evaluator results.
3. Each readout consumes a typed arena view and state. Parameter, observation,
   member, and role interpretation is delegated to the same compiled or
   execution owner used by inference.
4. A readout does not reimplement a message update or objective term.
5. Public `Engine`, `State`, `RunResult`, `RunStatus`, and
   `InferenceError` retain their documented imports, signatures, values, and
   failure behavior.
6. The public engine owns orchestration only: initialization, data binding,
   selection of compiled execution, status-bearing public calls, and delegation
   to objective and readout owners.
7. `src/cffg/engine.py` has fewer than 1,000 physical lines. No new or
   extracted file exceeds 1,000 physical lines, and each file has one stated
   responsibility.
8. Every applicable `cff-sf29` message, belief, status, diagnostic,
   objective, and readout remains unchanged within its dtype-derived tolerance.
9. Flooding and scheduled readouts agree for equivalent states. Direct and
   one-instance `vmap` readouts agree where both are supported.
10. Latest-schema and explicitly supported legacy RollingAgent checkpoints
    restore through the unchanged checkpoint decoder and produce the same
    public engine results. The extraction adds no checkpoint field or
    migration.
11. Literal zero support, vacuous messages, improper continuous beliefs,
    non-finite state, float32, and uncomfortable-scale readout cases retain
    their documented output or loud failure.
12. The ticket adds no vectorization target, latency claim, diagnostic feature,
    factor family, scientific conclusion, or release promise.
13. Focused readout, learning, recurrence, GFE, runtime-status, agent, public
    API, and checkpoint tests pass. The dashboard-backed short suite passes
    after the final code change.
14. `docs/ARCHITECTURE.md` names the readout owners and final public-engine
    responsibility in the same commit as the extraction.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Group readouts by representation or plan owner, not by whichever public
  method currently calls them.
- Keep public methods as small delegates until downstream callers can use any
  lower-level stable interface directly.
- Use the **Runtime numerical or status invariant**, **New factor or rule
  family**, and **Learnable or checkpoint feature** change-path rows.
- Preserve output order and dtype deliberately; do not combine this move with
  batching or caching.

## Future work

Closing this ticket closes the engine track `cff-ws3c`. `cff-ofv1` then
runs the final bounded extension proof. `cff-3kdk` remains the separate
readout-vectorization owner.

## Definitions

- **Facade:** A small public interface that delegates to canonical internal
  owners.
- **Learning-statistic readout:** Factor-local sufficient statistics consumed
  by a declared learning family.
- **Readout:** A public posterior, statistic, prediction, or diagnostic derived
  from an engine state.

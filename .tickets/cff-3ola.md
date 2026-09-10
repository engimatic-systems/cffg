---
id: cff-3ola
status: open
deps: [cff-gl89]
links: [cff-ws3c, cff-3kdk, cff-7kfk]
created: 2026-09-02T04:02:01Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-ws3c
tags: [engine, architecture, readout, refactor]
---
# Extract readouts and slim public Engine orchestration

## Background in Plain English — WHY

At authoring snapshot e488df3, `src/cffg/engine.py:1753-2478` is the scope
anchor for public readouts, not a promised future line map. The implementation
starts from the final published `cff-gl89` head, including the intervening
parameter-ownership and failure-diagnostic repairs. `cff-sf29` is the contract
inventory, not an instruction to restore its older implementation. After execution, schedules,
lifecycle, run control, and objective accounting have focused owners, the
public engine still contains discrete and Gaussian marginals, node beliefs,
recurrence pairs and statistics, mixture predictions, exponential-family
marginals, kernel readouts, and goal or parameter diagnostics. Several
readouts repeat plan and parameter selection.

This ticket gives readouts focused owners and leaves the public `Engine` as a
small orchestration facade. It preserves algorithms and output shapes.
Readout vectorization remains separate work in `cff-3kdk`.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-gl89` is closed with the focused objective owner preserving existing
   mathematical authorities and bounded characterization.
2. Focused readout owners cover every engine readout documented by the
   `cff-sf29` baseline, including discrete, Gaussian, exponential-family,
   mixture, recurrence, goal, kernel, learning-statistic, GFE, and
   functional-evaluator results.
3. Each readout consumes a typed arena view and state. Parameter, observation,
   member, and role interpretation is delegated to the same compiled or
   execution owner used by inference.
4. A readout does not reimplement a message update or objective term.
   4.1. `with_goal_priors`, which sits among the current readouts but replaces
        state parameters, delegates to a lifecycle or goal-parameter state
        owner. Keep its fresh-state guard and validation unchanged; do not
        make a readout owner responsible for state replacement.
5. Public `Engine`, `State`, `RunResult`, `RunStatus`, and
   `InferenceError` retain their documented imports, signatures, values, and
   failure behavior.
6. The public engine facade re-exports state/result/status types from their
   lower-level owner and retains public construction, compatibility delegates,
   selection of the explicit run strategy, shared result-finalizer calls, and
   delegation to lifecycle, objective, and readout owners. The facade injects
   the schedule program and objective evaluator into the selected run-control
   owner. It implements no family rule, schedule-construction algorithm,
   solver loop, or local objective formula. Invoking the existing schedule
   owner's builder from public construction remains facade orchestration.
7. `src/cffg/engine.py` has fewer than 1,000 physical lines because the named
   owners move responsibilities anchored by the e488df3 execution region
   (2511-3040), schedule region (3041-3669), lifecycle region (323-915), run-
   control regions (916-1752), ledger region (3670-3973), and readout region
   (1753-2478), with overlap resolved during extraction. No new or extracted
   file exceeds 1,000 lines.
8. Every applicable contract case for messages, beliefs, statuses, diagnostics,
   objectives and readouts is characterized against the final `cff-gl89`
   source and remains unchanged within its dtype-derived tolerance.
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
15. The ticket fits one implementation context because it moves only the
    readout responsibility anchored by the bounded e488df3 region after every
    other engine responsibility has a typed delegate.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Group readouts by representation or plan owner, not by whichever public
  method calls them at the `cff-sf29` named baseline.
- Keep public methods as small delegates until downstream callers can use any
  lower-level typed owner directly.
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

## Notes

**2026-09-10T08:18:52Z**

Pre-implementation inspection at ledger source 90c58b1: remaining Engine readouts are marginals, node_beliefs, dirichlet_marginals, goal_marginals, goal_prior_updates, gaussian_marginals, recurrence_pair_beliefs, gaussian_node_stats, recurrence_weight_stats, gaussian_mixture_predictives, ef_marginals, kernel_readouts and decode, plus hybrid_gfe_diagnostics near the lifecycle delegates. Mu3 diagnostics/epoch scoring already have Mu3Lifecycle owners; do not duplicate their implementation. with_goal_priors is a 55-line parameter-state replacement, not a readout: place it with lifecycle/goal-state ownership, preserving its unknown-blob check, fresh count-message guard, finite-positive shape validation, dtype cast and replacement State. Public inter-readout calls currently exist from goal_marginals/goal_prior_updates to dirichlet_marginals, gaussian_node_stats/recurrence_weight_stats to recurrence_pair_beliefs, and decode to marginals. Account for those explicit dependencies when designing typed views/callbacks; preserve output order and current parameter/observation interpretation, including legacy EF lookup timing. EfMarginal and DirichletMarginal value declarations need lower-level ownership with their existing engine import aliases preserved. Existing schedule construction may be invoked by the facade; its algorithm is already outside Engine. Local inspection, not independent agent review.

**2026-09-10T08:24:09Z**

Typed-view design note from the remaining source: the readout owners can use read-only structural protocols implemented by the immutable Arena, rather than duplicating segment-index arithmetic or copying plans per call. Derive each view from both direct field use and transitive transport-helper dependencies; a direct attribute scan alone misses helpers that receive the whole arena. Preserve the current public inter-readout delegation points noted above. This is an optional implementation approach, not new runtime validation or a performance feature. No source changes were made during the ledger final gate.

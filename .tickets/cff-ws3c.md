---
id: cff-ws3c
status: open
deps: [cff-3ola]
links: [cff-edoz, cff-3kdk, cff-cj5h, cff-wbhu]
created: 2026-08-13T04:26:16Z
type: epic
priority: 1
assignee: coopwilliams
parent: cff-cfqg
tags: [architecture, engine, extensibility]
---
# Engine track: Give runtime responsibilities focused owners

## Background in Plain English — WHY

At authoring snapshot e488df3, `src/cffg/engine.py` has 3,973 lines. Public
lifecycle/run control roughly at lines 323-1752, readouts 1753-2478, plan
execution 2511-3040, schedules/propagation 3041-3669, and the objective ledger
3670-3973 are scope anchors, not promised future line maps. The implementation
starts from the exact compiler-track head characterized by `cff-sf29`. The
module has 44 factor/group type-selection references at the authoring snapshot
and several repeated input/parameter interpretations.

This track first inventories bounded existing evidence. It then moves one
responsibility per context: rule execution, schedules, state/data/mu3
lifecycle, ordinary iteration, explicit outer/exact strategies, objective
accounting, and readouts. Engine and Colony use one shared result policy.

## Outcomes / Acceptance Criteria — WHAT

1. Direct dependency `cff-3ola` is closed. Its chain closes `cff-gl89`,
   `cff-jrbw`, `cff-ii9t`, `cff-5xdy`, `cff-nyad`, `cff-ny7f`, `cff-sf29`,
   and compiler track `cff-edoz` first.
2. `cff-sf29` records the exact commit that closes `cff-edoz` as the engine
   baseline, inventories existing mathematical authorities, and adds only the
   smallest old/new observable characterization needed for uncovered refactor
   seams. New scientific validation gaps remain separate work unless a gap
   proves the proposed move unsafe.
3. Per-plan local-update owners consume typed input/parameter blocks and invoke
   existing rules. Flat-arena and registerized-scan transport share that owner
   while preserving local solver facts and existing scan carry.
4. One schedule owner constructs flooding, stage, scan, support-check,
   exact-unit, and propagation programs without interpreting a factor rule.
5. Ordinary state/data lifecycle and specialized functional-mu3 lifecycle have
   narrow owners. Checkpoint decoding stays outside this extraction.
6. Ordinary iterative control, outer trust-region control, and exact-run
   control return typed internal outcomes. The specialized strategies remain
   explicit; no universal solver callback receives irrelevant state.
7. Engine.run selects a typed strategy. Engine and Colony use the shared
   termination policy and result finalizer delivered by cff-ii9t; neither
   duplicates the public status mapping. Runtime value definitions sit below
   the facade and retain their documented re-exports and identity.
8. One objective owner assembles factor, parameter, constraint,
   GFE/P-substitution, and edge-entropy terms. Focused readout owners consume
   typed views without reimplementing rules or objective terms.
9. The public engine facade re-exports public state/status/result types,
   construction and compatibility delegates, explicit strategy selection,
   shared result-finalizer calls, and delegation. It owns no family rule,
   schedule builder, solver loop, local ledger formula, or readout algorithm.
10. The below-1,000-line public-engine target follows from e488df3 source-region
    anchors:
    regions: lifecycle 323-915, ordinary control/status 916-1190 and
    1657-1752, outer/exact control 1191-1656, readouts 1753-2478, execution
    2511-3040, schedules 3041-3669, and ledger 3670-3973. Overlap and small
    compatibility delegates are reconciled during extraction. No extracted
    source file exceeds 1,000 lines.
11. Final imports remain acyclic: compiled values/rules feed execution;
    execution feeds schedules; lifecycle prepares state; the objective owner
    consumes typed plans/state and supplies an evaluator; ordinary/outer/exact
    control consumes schedules plus that evaluator; readouts consume typed
    state; the facade injects/selects these owners and sits above them. The
    ledger may land later via a compatibility delegate, but it never imports a
    controller and no controller imports the public facade.
12. Flooding/scheduled, direct/multi-instance and multi-structure Colony,
    host/JIT status, and
    latest-schema/supported-legacy restored-state pairs retain their stated
    applicability and observable behavior.
13. Literal zero support, vacuous/improper inputs, non-finite updates, float32,
    and uncomfortable-scale cases retain documented values or loud statuses.
14. No runtime-speed threshold, new solver/rule/factor, diagnostic feature,
    scientific claim, checkpoint migration, or public-release promise enters
    this track.
15. `docs/ARCHITECTURE.md` names every final engine owner and dependency.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Follow the child order and land one typed owner per atomic implementation
  change.
- Use **Runtime numerical or status invariant** as the primary change-path row;
  each child states the additional factor/IR/learning rows that apply.
- Keep source checks proving schedule code uses the shared execution owner and
  Engine and Colony use one shared termination/result owner.
- Preserve compatibility delegates until all internal consumers use typed
  owners.

## Future work

`cff-ofv1` verifies the bounded extension paths after this track. `cff-3kdk`,
`cff-cj5h`, and `cff-wbhu` remain independent readout, diagnostic, and
performance consumers.

## Definitions

- **Facade:** The public compatibility/orchestration layer above focused
  internal owners.
- **Lifecycle owner:** State initialization, observation binding, or specialized
  stochastic preparation without convergence decisions.
- **Run strategy:** One explicit internal control algorithm returning typed
  termination facts to `Engine.run`.
- **Status authority:** The sole public owner that constructs `RunStatus` and
  `RunResult` from typed facts for both Engine and Colony.

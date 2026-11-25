---
id: chr-d760
status: in_progress
deps: [cff-mjku]
links: []
created: 2026-11-25T06:08:03Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-pyy9
---
# Precommit a bounded policy for two future observations

## Background in Plain English — WHY

An agent that buys a cue can commit in advance to one response for each possible reading. The current rollout exposes action marginals and can replan after a reading. A precommitted policy must select its common first move before seeing the cue and execute its later choices without another solve.

## Outcomes / Acceptance Criteria — WHAT

1. A public builder expands one or two observation stages into an ordinary serializable CFFG with a common policy and a named action distribution at each history.
2. The native fixed-policy Bethe ledger gives the sum of the local negative log preferences weighted by their causal visitation probabilities.
3. A returned commitment executes every reachable history without inference and excludes impossible histories.
4. Independent depth-one and depth-two enumeration checks the scores and conditional marginals at float64 and float32 precision. Zero support, vacuous evidence and at least tenfold branch-entry scale are included.
5. A paid-cue commitment chooses compatible actions on both readings and reaches the analytic contingent payoff. A fixed sequence on the same task reaches the open-loop payoff.
6. JSON and CFFGZ reconstruction preserve named policy decisions. Over-depth and over-budget requests fail before expanding the candidate table.
7. Report native factor counts, compile and warm solve times and arena storage by depth.

## Methodological Suggestions — HOW

Use a common categorical variable for complete deterministic policy tables. Expose its clamp with observe_interior and distribute it using balanced equality fan-out. Give each history a preference observation and a conditional action readout with a distinct absent state. A uniform policy prior permits a native MAP readout and a fixed-policy score with a known prior offset. Freeze enumeration tolerances before inspecting results: absolute probability and energy errors within 256 epsilon times the number of history nodes, with relative energy tolerance 256 epsilon. Exercise forwards and restarts through public bundle and Engine APIs.

Change-path index: Runtime numerical or status invariant uses existing categorical factors, exact two-pass Engine and native BFE/status. Flooding, colored, mixed, host/JIT support and single/vmap operators have no kernel edits. Learnable or checkpoint feature uses standard template/blob reconstruction. No added IR field, learning record, kernel, schedule or legacy migration.

## Future work

The parent epic owns hierarchical execution, shared documentation, dashboard and the cumulative suite. Deeper policy trees are outside the supported domain.

## Definitions

A history is the sequence of future observations available when choosing an action. A commitment maps every reachable history to its action. A branch mass is the probability of reaching a history under the externally chosen actions. Bethe free energy (BFE) is the native ledger score; on a tree its value is minus the log evidence.


## Notes

**2026-11-25T06:08:17Z**

Implementation handoff: charged likelihood at a reached history equals its preference raised to its reach probability. The absent-action category permits reading causal visitation on the selected commitment. Benchmark and admission include every expanded history.

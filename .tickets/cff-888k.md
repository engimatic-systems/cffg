---
id: cff-888k
status: in_progress
deps: []
links: []
created: 2026-08-21T03:20:06Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-l5ag
tags: [rslds, validation, statistics]
---
# Expose paper-adapter component collapse

## Background in Plain English — WHY

An rSLDS chain can stop using one of its discrete regimes. That chain may then
look stable in some pooled parameter summaries even though it no longer
represents the registered four-state model. The frozen run kept every retained
regime draw, but its eligibility report does not distinguish a brief empty
component from persistent component death.

The adapter needs explicit, label-independent occupancy evidence before a
prior or sampler remedy can be judged.

## Outcomes / Acceptance Criteria — WHAT

1. Every retained chain reports regime occupancy per draw, missing labels,
   minimum and median occupancy, and the longest consecutive empty-regime run.
2. A registered rule distinguishes transient empty draws from persistent
   component collapse without using final data to choose its threshold.
3. Persistent collapse prevents adapter eligibility and names the affected
   chain and component evidence.
4. Occupancy summaries are invariant to regime-label permutations while raw
   labeled traces remain available for diagnosis.
5. Source, destination, and gate exposure counts are reported wherever the
   retained representation makes them identifiable.
6. Tests cover healthy label permutations, transient emptiness, persistent
   component death, one collapsed chain among healthy chains, exact-zero
   occupancy, and non-finite or malformed input.
7. New checkpoints, result schemas, manifests, and aggregate recomputation
   preserve the evidence; completed cff-oyie artifacts remain immutable.
8. The work changes no sampler equation and inspects or runs no registered
   final seed.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Derive the first implementation entirely from already retained regime draws.
- Use sorted occupancy vectors for label-independent eligibility while keeping
  original labels in the diagnostic payload.
- Add warmup occupancy to the isolated runtime only if a later development
  experiment needs to locate when collapse begins.

## Future work

`cff-888k` and `cff-jq9p` unblock the downstream shared-prior study `cff-8w4c`.
A sampler remedy such as tempered warmup
or a component-birth move requires a separate ticket if the development study
still shows persistent collapse. `cff-l5ag` owns the integration decision.

## Definitions

- **Component collapse:** A fitted chain persistently assigns no observations
  to one or more registered regimes.
- **Occupancy:** The number of time steps assigned to a regime in one retained
  draw.
- **Transient:** Present briefly but not sustained under the registered
  collapse rule.

## Notes

Architecture handoff: use `docs/ARCHITECTURE.md`'s canonical rSLDS consumer
validation boundary. The initial implementation is experiment reporting only;
the adapter equations, CFFG engine, compiler, VM, schedules, kernels, ledger,
serialization, and learnables are N/A.

Duplicated surfaces: four chains, classic and heterogeneous families, both
initialization tiers, raw checkpoints, result aggregation, calibration, and
future final execution must stay aligned. A large classic trajectory is the
uncomfortable-scale input. The reference runtime is float64-only, so float32
is N/A.

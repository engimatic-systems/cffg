---
id: cff-8whp
status: open
deps: [cff-xdip]
links: []
created: 2026-08-13T04:33:03Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-5edo
tags: [planning, performance, warm-start]
---
# Warm-start overlapping RollingAgent planning windows

## Background in Plain English

After `cff-xdip`, `RollingAgent` will refuse to act until its current plan is
converged. Each environment step will still initialize the next planning window
from neutral messages even though most time slices overlap with the previous
converged plan. That repeated work is an optimization problem, not a status-
truth problem.

This ticket reuses compatible state across adjacent planning windows and proves
that the warmed solve reaches the same result as a cold solve. It is downstream
of the depth and action gate so a speed result cannot hide an unfinished plan.

## Outcomes / Acceptance Criteria

1. `cff-xdip` is closed with depth-aware planning and the action status gate.
2. The warm-start mapping shifts every compatible message and inner-solver
   block for an overlapping slice into its next-window position.
3. The new window receives current boundary beliefs and parameter values, and
   its new terminal slice receives ordinary initialized state.
4. Expired actions, observations, hardening decisions, support state, and
   environment data do not survive into the new window.
5. A warm and cold plan with identical template, parameters, boundary beliefs,
   random keys, cap, and tolerance reach matching status, free energy, beliefs,
   action probabilities, and deterministic MAP action under dtype-derived
   tolerances.
6. A warm failure can retry from a clean cold state only under a documented
   policy that preserves the same stochastic realization and never reports the
   failed warm attempt as converged.
7. The cache is derived runtime state and is invalidated on episode reset,
   checkpoint restore, structure change, horizon change, shape change, dtype
   change, incompatible support, or invalid numeric content.
8. Restoring a checkpoint and cold-starting produces the same converged action
   as an uninterrupted compatible run. The warm cache is not required for
   checkpoint correctness.
9. Cache storage stays bounded by compiled planning-state size across 10,000
   environment steps.
10. Hostile coverage includes exact zero support during shifting, a vacuous
    boundary, non-finite or wrongly shaped cached state, float32, and horizon
    50.
11. A paired benchmark uses at least 100 post-warmup environment steps and
    reports compile, first execution, cold planning, warm planning, readout,
    hardware, dtype, cap, tolerance, statuses, and confidence intervals.
12. Median steady-state warm planning at horizon 20 is at least three times
    faster than cold planning while satisfying criterion 5.
13. Architecture and limitations documents state the cache lifecycle, measured
    speed envelope, and invalidation behavior.

## Methodological Suggestions

These suggestions are optional. An implementer may choose another sound method
that satisfies the numbered outcomes with equivalent evidence.

- Compile a mapping between old and new arena addresses from serialized slice
  identity rather than matching factor-name prefixes at runtime.
- Clone the same agent state before paired warm and cold runs and synchronize
  JAX work before recording time.
- Keep long behavior and storage checks marked slow; use compact parity and
  invalidation cases in the default suite.
- Use the **Runtime numerical or status invariant** change path and review
  checkpoint consumers even if the cache remains intentionally unpersisted.

## Future work

No further warm-start work is identified. Slice-specific kernel fusion and
accelerator throughput remain separate measured optimizations.

## Definitions

- **Cache invalidation:** Discarding derived state when its structural or
  numerical assumptions no longer match the next plan.
- **Overlapping slice:** A time slice represented in both the previous and the
  next receding-horizon window.
- **Warm plan:** A planning solve initialized from compatible state produced by
  the previous environment step.

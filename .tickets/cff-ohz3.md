---
id: cff-ohz3
status: closed
deps: []
links: []
created: 2026-08-22T16:50:40Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-lo4d
tags: [rslds, validation, statistics]
---
# Requalify the fixed adapter with a common retained-draw extension

## Background in Plain English — WHY

The preregistered cff-lo4d gate completed all sixteen adapter chains with no
collapse, stuck chain, restart, or infrastructure failure. Three of four
family/stream cells passed every family-authoritative variable. The classic
cell paired with CFFG shared-initialization streams missed only the held-out
on-support transition diagnostic: split R-hat was 1.068632 against 1.05 while
effective sample size already passed at 50.485 against 50.

This is the narrow condition cff-lo4d anticipated for a retained-draw
extension. No extension was registered before that run, so sampling cannot
resume or be selectively lengthened under the old protocol. A new development
protocol must fix one longer common policy before execution and preserve the
completed miss unchanged.

## Outcomes / Acceptance Criteria — WHAT

1. Before any new sampling, an immutable protocol doubles retained draws from
   200 to 400 while keeping warmup 100, thinning 1, paper equations, prior 1.0,
   fixed family initializer arrays, four chain streams, datasets, and numerical
   runtime unchanged.
2. A funnel reruns from the registered seeds the one classic/shared-stream cell
   that motivated the extension. Its first 200 retained draws reproduce the
   corresponding prior execution exactly for every retained array, proving the
   longer run changes only the stopping horizon.
3. The funnel passes only if every classic family-authoritative variable meets
   split R-hat at most 1.05 and effective sample size at least 50, with no stuck
   chain or persistent component collapse. An infrastructure failure or
   scientific miss stops expansion and remains manifested.
4. If the funnel passes, expansion executes the other three family/stream
   cells with the same 400-draw policy. Later calibration is authorized only if
   all four cells pass the unchanged family-qualified gates.
5. All sixteen chains remain independent, are retained regardless of outcome,
   and use one family-specific observation-derived initializer shared within
   the coordinate. There is no chain ranking, restart, drop, or truth-informed
   initialization.
6. Protocol, checkpoint, resume, aggregate, and manifest identity bind the
   parent 200-draw evidence, draw-prefix comparison, new sampler budget,
   prospective tier/cohort semantics, family inputs, chain streams, fixed
   policy, reference sources, and numerical environment.
7. Historical cff-oyie, prior-screen, native-basin, and cff-lo4d artifacts stay
   byte-identical and keep their original miss decisions. The extension remains
   development eligibility evidence, not calibration, non-inferiority,
   population, or adapter-initialization-robustness evidence.
8. Calibration and final seeds are rejected before data generation. No CFFG or
   control fit runs.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Build a schema-dispatched extension runner around the cff-lo4d chain and
  diagnostic primitives rather than altering the immutable v2 protocol.
- Rerunning deterministically from the original warmup seed is acceptable when
  the legacy sampler has no restartable state, provided the complete 200-draw
  prefix is compared and authenticated before the added draws influence a
  decision.
- Keep the funnel/expand structure so a failed longer classic cell prevents
  unnecessary additional sampling.

## Future work

If all cells pass, cff-lo4d and then cff-l5ag can adopt 400 retained draws as
the prospective calibration candidate. Calibration, budget repeatability, a
fresh freeze, and final evidence remain later tickets. If the funnel still
misses, diagnose its retained trace without weakening thresholds.

## Definitions

- **Retained-draw extension:** A longer fixed Markov-chain sampling horizon
  after the same warmup, with all other target and kernel choices unchanged.
- **Prefix reproduction:** Equality of every named retained array for the first
  200 draws between the old and extended deterministic runs.

## Notes

Architecture handoff: the sibling experiment's fixed-adapter protocol, nested
Python 3.11 runner, append-only attempt/checkpoint, family-qualified diagnostic,
occupancy, stage aggregate, and manifest boundaries apply. CFFG engine and
factor semantics are structurally irrelevant.

Duplicated surfaces: four family/stream cells, four chains, first-200 prefix,
added 200 draws, both family-authoritative variable sets, occupancy, tier/cohort
wording, resume, and authorization must remain aligned.

Hostile-input decisions: a changed prefix, missing draw, non-finite draw,
changed sampler seed, exact-zero support, persistent collapse, and one distinct
chain fail loudly. The reference runtime is locked to float64. Classic
T=12,500 and heterogeneous T=6,400 are the uncomfortable-scale cases.

Legend: prerequisite --> dependent

```text
[x] cff-upih fixed observation-only policy
      |
[x] cff-lo4d 200-draw gate: 3/4 pass, one narrow miss
      |
      +--> [ ] cff-ohz3 common 400-draw gate (ready)
                  |
                  +--> [ ] cff-lo4d / cff-l5ag adapter eligibility
                              |
                              +--> [ ] fresh calibration and freeze
```

**2026-08-22T18:18:17Z**

Completed and integrated on the active experiment branch through commit 8e2e210. The registered 400-draw extension completed all 16 attempt-0001 chains. All 208 named first-200 prefix comparisons were byte-identical and value-identical to the immutable cff-lo4d execution before added draws entered any decision. All four family/stream cells passed the unchanged family-qualified gates: classic/shared R-hat 1.0470401 and ESS 142.1712; classic/native 1.0313709 and 128.6865; heterogeneous/shared 1.0086379 and 447.6614; heterogeneous/native 1.0028601 and 491.5127. No chain was stuck or persistently collapsed. The authenticated result is complete_authorized and permits later calibration only. Integrated focused validation passed 53 tests; completed resume validation returned complete_authorized; git diff --check passed. No CFFG/control fit, calibration seed, or final seed ran.

Legend: prerequisite --> dependent

[x] cff-upih fixed observation-only policy
      |
[x] cff-lo4d 200-draw gate: narrow registered miss
      |
      +--> [x] cff-ohz3 authenticated 400-draw extension
                  |
                  +--> [ ] fresh calibration and freeze

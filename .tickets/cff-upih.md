---
id: cff-upih
status: open
deps: [cff-w42g]
links: []
created: 2026-08-22T14:26:39Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-l5ag
tags: [rslds, validation, statistics]
---
# Repair the heterogeneous native adapter basin without changing its target

## Background in Plain English — WHY

The source-heterogeneous seed-5000 screen found two different outcomes under
the same recurrence prior of 1.0. All four shared-initialization chains mixed
on the six claim variables. In the native tier, one chain reached the accurate
basin while three stayed near 89 percent regime accuracy; one of those chains
also kept a regime empty for 58 retained draws.

Centering each native trace within its chain reduced R-hat to about one and
raised effective sample size above 275. The chains are stable within different
basins, so retaining more draws alone is not a supported repair. The native
observation-only initializer begins much farther from the accurate regime path
than the mapped initializer, but the screen also used different Gibbs seeds by
tier. A crossed development study must separate initialization from sampler
randomness before code changes.

## Outcomes / Acceptance Criteria — WHAT

1. One development-only experiment crosses the same four Gibbs seeds with the
   mapped and native initializers on the same heterogeneous dataset, prior,
   kernel, warmup, and retained-draw policy.
2. Warmup checkpoints report family-authoritative functional diagnostics,
   component occupancy, regime accuracy, and per-chain basin membership before
   retained draws begin.
3. A fixed initializer and warmup policy is selected from declared development
   evidence without choosing, dropping, or restarting an individual chain
   because of its result.
4. Under that fixed policy, every declared heterogeneous chain has no persistent
   component collapse and meets split R-hat at most 1.05 and effective sample
   size at least 50 on every variable authorized by `cff-w42g`.
5. The paper Eq. 2, 4, 6, and 7 conditionals, the crossed study's fixed
   recurrence prior of 1.0, the full source-specific parameterization, and the
   retained-draw target stay unchanged and pass their independent density and
   transition oracles. The report states that this prior differs from the
   weak-prior cff-oyie target.
6. Generating states and parameters never enter sampling, initialization,
   warmup selection, or chain acceptance.
7. The isolated Python 3.11 runtime, pinned sources, seeds, initializer state,
   warmup checkpoints, raw draws, diagnostics, and manifests reproduce
   independently.
8. Calibration seeds and final seeds are rejected before data generation.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- First cross the existing mapped and native initializers with identical Gibbs
  seeds. This is cheaper and more informative than a kernel rewrite.
- If initialization controls the result, improve the observation-only
  PCA/AR-HMM/greedy-decision-list initializer while preserving independent
  chains.
- If both initializers still occupy different basins, checkpoint a declared
  warmup-doubling sequence. Do not thin as a mixing repair.
- Implement a new exact Markov kernel only after a small enumerated oracle
  proves its stationary target.

## Future work

`cff-l5ag` uses the completed study to decide whether the adapter is eligible
for a fresh comparison. A new calibration and immutable freeze remain
downstream. This ticket does not fix CFFG heterogeneous recovery.

## Definitions

- **Basin:** A region of posterior state space in which a Markov chain remains
  for the observed run.
- **Crossed study:** An experiment that applies every initializer to the same
  sampler seeds so their effects can be separated.
- **Warmup:** Sampling iterations used to approach the posterior before draws
  are retained for inference or mixing diagnostics.

## Notes

Architecture handoff: the sibling experiment's isolated reference-adapter,
initializer, retained-draw, checkpoint, and protocol boundaries apply. CFFG
factor equations, IR, compiler, VM, schedules, kernels, ledger, learning, and
serialization are N/A.

Duplicated surfaces: both initializers, four identical Gibbs streams, every
heterogeneous source and gate, warmup checkpoints, claim diagnostics,
occupancy, adapter provenance, and resume validation must stay aligned.

Hostile-input decisions: an initially empty component, exact-zero transition
support, saturated logits, a non-finite draw, and one chain remaining in a
distinct basin must remain explicit. The adapter is float64, so float32 is
N/A. The 5,120-step training path is the uncomfortable-scale case.

Legend: prerequisite --> dependent

```text
[ ] cff-w42g family-qualified authority (ready)
      |
      +--> [ ] cff-upih native-basin study
                  |
                  +--> [ ] cff-l5ag adapter repair
                              |
                              +--> [ ] fresh comparison freeze
```

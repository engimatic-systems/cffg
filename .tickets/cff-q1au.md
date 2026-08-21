---
id: cff-q1au
status: open
deps: []
links: [cff-oyie]
created: 2026-08-20T15:41:17Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, statistics]
---
# Make heterogeneous final-seed coverage robust before the next freeze

## Background in Plain English

The heterogeneous generator passed every coverage rule on its two calibration
seeds, but untouched final seed 2102 did not. Source 3 reached gate 2 fifty-one
times and produced four successes and forty-seven failures. The frozen rule
required at least five of each outcome. The design still had rank three, but
the runner correctly stopped before fitting because the binary outcome count
was too small for the registered capability claim.

Two calibration trajectories were not enough to show that a random 1,280-step
training path would reliably meet every source-and-gate rule. The next design
must make this eligibility property robust without selecting or replacing a
final seed after seeing it.

## Outcomes / Acceptance Criteria

1. A calibration-only study measures the distribution of exposures,
   successes, failures, rank, and conditioning for every source and gate over a
   declared set of nonfinal development seeds.
2. The next heterogeneous scale or generator makes the minimum five-success,
   five-failure, twenty-exposure, rank-three contract hold by construction or
   establishes a one-sided 95% confidence lower bound of at least 0.99 on the
   probability that one complete dataset passes every coverage cell.
3. The source rows remain different on the common-state grid, and 10,000-step
   rollouts retain periodicity at least 0.25 with amplitude and final-to-initial
   variance ratios at most 10.
4. The classic source-homogeneous generator and its legacy random stream remain
   byte-for-byte compatible.
5. The prefit eligibility check continues to reject every one-sided, sparse, or
   rank-deficient final dataset before any fitter runs.
6. Any new confirmatory run freezes fresh final seeds before generating them.
   Seed 2102 remains an immutable not-ready result.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- First test whether a longer training path provides enough margin without
  changing the heterogeneous transition surfaces.
- Prefer a reliability calculation or conservative lower quantile over a
  hand-picked list of paths.
- Keep the eligibility check independent from fit quality and outcome scoring.

## Future work

A later ticket may recalibrate thresholds and run fresh final seeds after this
design and the learning repairs close. This ticket does not change the
completed cff-oyie artifact.

## Definitions

- **Augmented-design rank:** The rank of the matrix containing both latent-state
  coordinates and an intercept; rank three is required to identify a two-
  dimensional affine gate locally.
- **Coverage cell:** One source-regime and binary-gate combination.
- **Final seed:** A preregistered random seed that is not used to choose the
  generator, scale, budget, or threshold.

## Notes

Architecture handoff: the sibling generator and prefit evidence boundary apply.
Core CFFG execution surfaces are N/A because no fit may begin on an ineligible
dataset.

Duplicated surfaces: heterogeneous generation, data identities, stochastic
streams, source-design audit, scale selection, freeze validation, and prefit
runtime checks must agree. The classic generator is a required compatibility
surface.

Hostile-input decisions: exact-zero and one-sided outcomes, fewer than twenty
exposures, rank loss, poor conditioning, float32 generation, and trajectories
at least ten times the smallest fixture must all be exercised.

Legend: prerequisite --> dependent

```text
[x] cff-oyie seed-2102 eligibility miss
      |
      +--> [ ] cff-q1au robust source/gate coverage (ready)
                  |
                  +--> [ ] fresh heterogeneous freeze and run
```

**2026-08-21T01:23:37Z**

Independent-review synthesis: use 6,400 total / 5,120 training steps as the conservative heterogeneous candidate. The 4,800 / 3,840 result is the smallest measured coverage-only candidate, while 5,120 training steps adds margin for the separately observed recovery-identifiability problem. Reproduce the reliability evidence as a manifested experiment artifact and benchmark resources; improved recovery and outer contraction remain hypotheses to test. Unified plan: notes/rslds-debug/synthesis/plan.md.

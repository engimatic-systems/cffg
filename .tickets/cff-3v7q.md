---
id: cff-3v7q
status: in_progress
deps: []
links: [cff-oyie]
created: 2026-08-20T15:41:17Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, learning]
---
# Stabilize classic NASCAR periodic rollout adequacy

## Background in Plain English

All six classic CFFG fits converged and their long rollouts stayed bounded.
However, the native-initialization fit on seed 2001 had periodicity score
0.4058, below its registered 0.4110 floor. The official classic conclusion is
not ready because the reference adapter failed its mixing gate, so the runner
did not issue a formal per-effect decision. The preserved CFFG value still
shows a narrow task-adequacy miss that should be understood before any fresh
parity run.

The other five classic scores exceeded their tier-specific floors. This ticket
therefore owns robustness across data and rollout streams, not a claim that the
model never produces periodic behavior.

## Outcomes / Acceptance Criteria

1. A calibration-only study separates fit variation from rollout-stream
   variation for the bounded periodicity score.
2. Both initialization tiers exceed their preregistered periodicity floors on
   a declared set of classic development datasets without selecting a favorable
   rollout seed.
3. On declared classic development seeds, the changed model stays within 0.1
   held-out mean predictive log-likelihood and 0.02 label-aligned regime
   accuracy of the pre-change model, keeps held-out on-support transition error
   at most 0.15, and leaves the canonical generator hash unchanged.
4. Long rollouts still report amplitude and final-to-initial variance ratios;
   an unbounded trajectory can never receive a passing periodicity score.
5. Float32 and 10,000-step rollouts remain finite and use dtype-scaled numerical
   guards.
6. The cff-oyie seed-2001 value remains unchanged. Fresh confirmatory evidence
   requires a newly frozen protocol and seeds.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Reuse the frozen scoring code while varying only registered development fit
  and rollout streams.
- Diagnose whether the miss comes from recurrence, dynamics, or rollout
  initialization before changing learning.
- Keep the periodicity floor independent from the reference adapter.

## Future work

A future parity rerun may depend on this ticket and cff-l5ag. This ticket does
not change the heterogeneous recovery experiment.

## Definitions

- **Periodicity score:** The bounded-rollout autocorrelation score registered as
  an absolute NASCAR task-adequacy outcome.
- **Rollout stream:** The fixed random-number stream used to simulate from a
  fitted model after fitting is complete.

## Notes

Architecture handoff: the sibling experiment's classic rollout and secondary-
outcome boundary apply first. Core recurrence or dynamics ownership becomes
applicable only if the diagnosis locates the miss in learned parameters.

Duplicated surfaces: both initialization tiers, fit and rollout streams,
periodicity and boundedness scoring, classic transition recovery, and final
reporting must stay aligned.

Hostile-input decisions: unstable, constant, non-finite, float32, and
10,000-step trajectories must produce explicit bounded or failed evidence.
Sparse support and improper beliefs remain governed by the fitting path.

Legend: prerequisite --> dependent

```text
[x] cff-oyie classic rollout evidence
      |
      +--> [ ] cff-3v7q periodicity robustness (ready)
                  |
                  +--> [ ] fresh classic parity run
```

**2026-08-21T06:31:13Z**

Safe prerequisite landed in the sibling experiment as commit 8604da9 and was integrated as fdde5d4. The frozen seed-2001 native score is 0.4058051268 against floor 0.4109805203, a 0.0051753935 miss. Exact retrospective decomposition is impossible because cff-oyie retained no native CFFG parameter arrays or initial-state law. The new development-only diagnostic registers 3 data seeds x 2 tiers x 2 fit replicates = 12 fits, each crossed with 4 rollout streams, and hardens constant/nonfinite/unbounded/float32/10,000-step scoring without changing frozen evidence. Shared-prior baseline classic coordinates will durably provide the six replicate-0 checkpoints through integration commit 7bb020d; six declared replicate-1 development fits remain before replay. No fit, chain, calibration seed, or final seed ran for this ticket.

Legend: prerequisite --> dependent

[ x ] score/checkpoint/replay prerequisite (8604da9)
  |
  +--> [ ] six baseline replicate-0 checkpoints from cff-8w4c
  |       |
  |       +--> [ ] six replicate-1 classic development fits
  |                 |
  |                 +--> [ ] 12-fit x 4-rollout decomposition
  |                           |
  |                           +--> [ ] cff-3v7q decision
  |
  +--> [ ] fresh parity freeze (downstream)

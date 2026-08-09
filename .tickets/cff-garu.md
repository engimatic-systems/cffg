---
id: cff-garu
status: in_progress
deps: []
links: [cff-l7y4]
created: 2026-08-09T05:13:04Z
type: bug
priority: 2
assignee: coopwilliams
parent: cff-ahps
tags: [rslds, numerical-robustness, scheduling]
---
# Stabilize inner rSLDS solves after repeated parameter replacement

## Background in Plain English

The observed-only fitter from cff-l7y4 alternates hidden-state inference with point-parameter replacement. On the compact NASCAR pilot (T=240, K=4, latent d=2, observed p=10, seed 101), two replacement coordinates followed by final inference converge under the registered damped schedule. After a third replacement, the next unchanged-protocol inner solve exhausts its iteration budget: the Bethe free-energy change becomes small while the absolute message-delta certificate remains large.

This boundary does not invalidate the preregistered cff-l7y4 protocol, whose outer budget is exactly two. It does prevent callers from assuming that additional finite coordinates monotonically improve or preserve solvability. This ticket owns diagnosing and stabilizing that repeated-replacement path without weakening the public converged / budget_exhausted / diverged distinction. cff-l7y4 owns the fitter and comparison; cff-r6bl owns cold setup performance.

## Design

Use the Runtime numerical or status invariant change-path row. Retain the pilot mechanism as a bounded regression: preserve the observations, initializer, immutable priors, parameter-write cadence, schedule, damping, and tolerances while recording per-phase Bethe free energy, message delta, recurrence coordinate status, and parameter scales before and after the third replacement.

Determine whether the failure is a scale-dependent certificate, a recurrence mean-field cycle, or a Gaussian-message instability. Fix the owning layer and keep exact-zero support, float32, and long-horizon behavior explicit. Do not tune the cff-l7y4 preregistration or relabel exhausted inference as converged.

## Acceptance Criteria

1. A compact deterministic regression reproduces the post-third-replacement budget exhaustion using observations and an explicit key, without receiving generating states or parameters.
2. The regression records enough per-phase evidence to distinguish a recurrence-coordinate failure, a non-finite or indefinite Gaussian belief, a finite message cycle, and a convergence certificate whose scale is inappropriate.
3. The corrected path completes at least five parameter replacements and a final hidden-state solve with RunStatus.CONVERGED under a preregistered bounded budget, or returns RunStatus.DIVERGED with a falsifiable mathematical reason when the coordinate map has no supported fixed point.
4. The correction does not weaken the engine rule that convergence requires finite Bethe free energy and the documented message/status certificate.
5. Exact-zero regime support remains literal, improper beliefs remain loud, and both float32 and x64 use dtype-derived guards.
6. The compact regression runs in the short suite; larger NASCAR confirmation remains in the sibling experiment.
7. docs/ARCHITECTURE.md records the owning convergence or scheduling mechanism, and docs/LIMITATIONS.md records any accepted repeated-coordinate boundary.

## Future work

This ticket does not change the observed-only parameter coordinates, add a
joint parameter objective, or optimize canonical cold setup. cff-l7y4 owns the
first fitting and comparison contract, cff-ahps owns any future joint
variational objective, and cff-r6bl owns repeated-slice lowering and setup
cost.

## Definitions

- **Inner solve:** One hidden-state and regime inference run between parameter
  replacements.
- **Message-delta certificate:** The finite message-change condition the engine
  requires in addition to a stable Bethe free energy before reporting
  convergence.
- **Parameter replacement:** Recomputing a one-sequence posterior from fixed
  priors and installing its point parameters into a fresh state.
- **Repeated-replacement path:** More than two consecutive inference and
  parameter-replacement coordinates on one observed sequence.


## Notes

**2026-08-09T05:13:25Z**

Intelligible-ticket review, 2026-08-08. Capability owners are explicit: this ticket owns the repeated-replacement inner-solve mechanism; cff-l7y4 owns fitting and comparison; cff-r6bl owns cold setup. The bounded wiggle scan returned no retained candidates.

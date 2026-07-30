---
id: cff-weff
status: open
deps: []
links: []
created: 2026-07-30T02:49:10Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-mwjo
---
# Globalize mixed-GFE outer fixed-point inference

## Plain English

Mixed GFE plans will converge through a safeguarded outer-loop method when synchronous message updates fall into a limit cycle. The runtime will preserve the same local message equations and free-energy objective, but it will use a deterministic globalization strategy instead of relying on one fixed damping guess. This unblocks the factored annealed-novelty experiment without accepting unstable beliefs.

## Technical description

The cff-5ipi/v3 run completed all 16 exact-sensor known/plain reference cells, then learn-only seed 0 exhausted 400 sweeps at episode 1 step 2 with FE/message oscillation and residual 1.69. Its inner Newton status was CONVERGED. An excluded-seed lifecycle reproducer and independent restarts at fixed damping .5 through .99 (up to 15,000 sweeps) also limit-cycle after learned counts change. Implement a deterministic safeguarded outer solver for Engine mixed schedules (for example block-coordinate GFE member updates or Anderson acceleration with FE/residual safeguards), retaining fixed points and the existing node/edge FE ledger.

## Hostile-input plan

Exact support and delta beliefs: reproduce the 8x2 factored corridor after one and one full episode of conjugate transition updates. Vacuous initialization: the existing uniform-prior cases must remain convergent. Float32: all guards and restart/acceptance criteria derive from finfo(dtype). Scale: cover horizon 8 with a 24-logit GFE-TM composite and retain the 32-position x 4-selector rule test. Non-finite FE/delta is divergence; a globalization budget that cannot certify a fixed point remains BUDGET_EXHAUSTED.

## Acceptance Criteria

The safeguarded method certifies finite FE/messages and RunStatus.CONVERGED for the cff-2gev one-update and excluded-seed lifecycle mechanisms without changing local rules or weakening tolerances; existing convergent fixtures retain their marginals and FE to tolerance; a deliberately unsolved case returns BUDGET_EXHAUSTED and NaN input returns DIVERGED; float32 and uncomfortable-scale hostile tests pass; ARCHITECTURE and DERIVATIONS describe the outer solver and its fixed-point/ledger invariants.


## Notes

**2026-07-30T02:50:20Z**

Discovery evidence: cff-5ipi/v3 clean run completed 16 reference cells, then learn-only seed 0 failed at episode 1 step 2 after one exact transition update (outer residual 1.6901; inner CONVERGED). Excluded seed 999 fails at episode 2 under damp .8/.85; frozen independent restarts .9/.93/.95/.97/.98/.99 all exhaust 5k--15k sweeps with finite converged inner solves. Invalid traces: /Users/ravd/agent-share/cffg-annealed-novelty-cff-5ipi-v3. Source confidence high; regression should reproduce the mechanism without analyzing outcomes.

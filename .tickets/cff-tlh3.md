---
id: cff-tlh3
status: closed
deps: []
links: []
created: 2026-08-21T04:33:20Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-qulu
tags: [rslds, learning, validation, numerics]
---
# Validate selected-scale outer convergence

## Background in Plain English — WHY

The certificate replay showed healthy convergence at 1,280 training steps, but
the coverage reliability study selected 5,120 training steps for heterogeneous
recovery. A longer trajectory changes the amount of evidence and the cost of
every outer update. The repair cannot choose a later budget from the smaller
replay alone.

The existing dual-scale runner would repeat obsolete horizons and paper-adapter
chains that cannot answer this question. This ticket adds a focused,
checkpointed development cohort for the four learned outer loops only.

## Outcomes / Acceptance Criteria — WHAT

1. A development-only command runs source-heterogeneous trajectories with 6,400
   total steps and 5,120 training steps on seeds 5000, 5001, and 5002.
2. Each seed runs shared- and native-initialization rSLDS fits plus same-regime-
   count and closest-parameter-budget SLDS controls under the canonical runtime
   residual and unchanged 0.1 two-consecutive-map rule.
3. Calibration seeds 3101 and 3102 and every registered final seed are rejected
   before dataset generation.
4. Every arm has an append-only checkpoint, status, convergence-map count,
   legacy and runtime residual history, inner-work summary, wall time, and peak
   memory evidence; interruption can resume without rerunning a completed arm.
5. The runner executes seed 5000's shared rSLDS and same-regime-count control
   before expanding to the remaining declared arms and seeds.
6. Budget exhaustion is stored as right-censored evidence and cannot enter the
   completed convergence-time set.
7. All 12 declared arms converge before a later calibration budget is derived;
   the chosen budget receives 25% relative and at least four-map absolute
   headroom without weakening the certificate.
8. Compact schema, resume, tamper, float32, non-finite, and status tests pass
   before the first 5,120-step arm starts.
9. The complete result and every checkpoint are manifested; no paper-adapter
   chain, registered final seed, or cff-oyie artifact is run or changed.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Reuse the fit and phase serializers from the outer-certificate diagnostic.
- Store one immutable attempt directory per seed, tier, or control rather than
  rewriting one whole-cohort payload after each arm.
- Use a development measurement cap of 40 complete outer maps. Stop and create
  a decision ticket if an arm remains right-censored at that cap.
- Keep terminal hidden-state inference outside the outer-map count.

## Future work

`cff-qulu` uses the completed cohort to freeze the calibration outer budget.
`cff-0oqo` then evaluates source-function recovery at the selected scale. A
globalization algorithm requires a new ticket only if this cohort shows
noncontracting or cyclic runtime-map evidence.

## Definitions

- **Development cohort:** The fixed set of non-calibration, non-final datasets
  used to choose an execution budget before a freeze.
- **Right-censored:** The arm did not converge before its declared limit, so its
  true convergence time is only known to exceed that limit.
- **Outer map:** One complete infer, replace, install, and residual evaluation;
  terminal hidden-state inference is separate.

## Notes

Architecture handoff: use `docs/ARCHITECTURE.md`'s observed-only fitting and
canonical rSLDS consumer-validation boundaries. The command belongs to the
sibling experiment. CFFG IR, compiler, VM, schedules, kernels, ledger,
serialization, factor equations, and learnable semantics are N/A.

Duplicated surfaces: three development seeds, two rSLDS tiers, two controls,
runtime and legacy residuals, append-only attempts, resume, manifests, float32,
and x64 must stay aligned. Exact-zero support and improper beliefs retain the
fitters' existing loud-failure contracts.

**2026-08-21T08:30:27Z**

Completed selected-scale development evidence in the sibling experiment. All 12 declared source-heterogeneous arms (3 seeds x shared/native rSLDS x same-K/K=5 SLDS) returned converged outer and inner statuses; none were censored, divergent, or failed. Convergence map counts were 6-15, yielding a registered budget of 19 from `max(8, ceil(1.25*15), 15+4)`. Total learned-arm wall time was 12,718.911434 seconds and peak recorded process RSS was 5,531,402,240 bytes. No adapter, calibration seed, or final seed ran. Evidence commit 57e2e8e; historical validator boundary fix 7dbf84d; manifests and all checkpoints validate.

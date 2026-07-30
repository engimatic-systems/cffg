---
id: cff-weff
status: open
deps: [cff-d56d, cff-j4mu]
links: [cff-2gev]
created: 2026-07-30T14:04:30Z
type: bug
priority: 2
assignee: coopwilliams
tags: [gfe, solver, numerical-robustness]
---
# Globalize mixed-GFE outer fixed-point inference

## Plain English

Mixed GFE plans get an outer solver that either reaches a stable fixed point or
reports why it did not. The solver adapts its step instead of relying on one
fixed damping value, and it keeps inner Newton success separate from outer
convergence. This follow-up is independent of the completed M3 scope.

## Evidence

The retained run on closed PR #43 exposed the defect in its first learn-only
cell, before annealing could activate: after one learned transition count
changed, the inner Newton solve converged while the outer mixed-GFE message and
free-energy sequence entered a finite limit cycle. Fixed damping values from
0.5 through 0.99 and budgets up to 15,000 sweeps did not recover convergence.
No source or experiment-motivated Newton threshold from that branch is part of
this ticket.

## Design

Introduce a safeguarded outer update for mixed GFE composites, with an explicit
merit or residual measure and adaptive damping, backtracking, or another
documented globalization strategy. Preserve truthful converged,
budget-exhausted, and diverged statuses. Report enough diagnostics to
distinguish an inner message-rule failure from an outer fixed-point failure,
and keep finite oscillation from masquerading as progress.

## Hostile Inputs

- Exact log-zero support must remain finite where the model is feasible and
  fail loudly where it is not.
- Vacuous or improper incoming messages must produce a truthful status.
- Float32 guards and stopping rules must derive from the active dtype.
- Tests must cover learned categorical blocks 10--100 times larger than the
  small fixture and a reproducible finite limit cycle.

## Acceptance Criteria

- A regression reproduces the outer-cycle mechanism without depending on the
  abandoned annealing experiment.
- The outer solver globalizes its updates and exposes the residual or merit
  used to accept them.
- Inner Newton convergence cannot certify outer mixed-GFE convergence.
- Non-finite values, detected cycles, and exhausted budgets return distinct,
  truthful outcomes.
- Existing analytically checked GFE messages and free-energy values retain
  their tolerances in float64 and survive the contract cases in float32.

---
id: cff-ao02
status: in_progress
deps: []
links: []
created: 2026-08-21T03:07:46Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-qulu
tags: [rslds, learning, numerics]
---
# Expose canonical outer fixed-point residual evidence

## Background in Plain English — WHY

The rSLDS fitter decides whether its outer learning loop has converged by
comparing a mixed snapshot. That snapshot combines posterior hyperparameters
with arrays used only for reporting. Some of those values are not installed
into the next inference solve, and the initial-regime report does not use the
same categorical potential that the engine receives.

A safeguarded outer solver needs a clean measurement of the undamped update in
the parameters that actually define the next solve. This ticket adds that
evidence without changing the current steps or terminal statuses.

## Outcomes / Acceptance Criteria — WHAT

1. A host-side primitive reports overall, per-array, and per-family undamped
   map residuals for named runtime parameter blocks.
2. The rSLDS fitter records its legacy residual and canonical runtime residual
   side by side without changing either its accepted steps or terminal status.
3. A fixed, truth-free transition-function probe reports functional change as
   a diagnostic.
4. Exact-map, non-finite, float32, float64, and compact rSLDS tests pass.
5. `docs/ARCHITECTURE.md` assigns ownership of the canonical residual at the
   observed-only fitting boundary.
6. The work generates and inspects no registered final seed.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Add a focused host-side coordinate module instead of extending the
  688-line `rslds_fit.py` beyond the project's 700-line warning.
- Measure `F(theta) - theta` before any damping. Do not use the accepted
  movement as the convergence certificate.
- Derive probe states from the observation-only initializer and hold them fixed
  for the fit. Do not use a truth-derived evaluation grid.
- Keep cross-replacement Bethe free energy diagnostic until a comparable outer
  objective is derived.

## Future work

`cff-0xjq` uses this evidence and the corrected SLDS terminal contract to choose
the production convergence certificate. `cff-qulu` owns any later safeguarded
outer step. This ticket does not change the algorithm.

## Definitions

- **Canonical runtime parameter:** A value or normalized potential installed
  into the engine for the next inference solve.
- **Outer fixed-point map:** The operation that infers hidden beliefs and maps
  the current parameters to a complete replacement candidate.
- **Undamped residual:** The difference between that complete candidate and the
  current installed parameters before any partial step.

## Notes

Architecture handoff: use `docs/ARCHITECTURE.md`'s observed-only fitting row for
status-gated execution, snapshot comparison, and terminal solve. IR, compiler,
VM, schedule, kernel, ledger, serialization, and `RunStatus` changes are N/A.

Duplicated surfaces: initial regime, dynamics, recurrence, emission, legacy
snapshot, canonical runtime snapshot, functional probe, float32, and float64
must stay aligned. A non-finite block fails loudly.

---
id: cff-rl7r
status: open
deps: [cff-0qry]
links: []
created: 2026-08-30T16:00:47Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-0qfy
tags: [consolidation, recurrence, numerics]
---
# Port direct recurrence covariance repair

## Background in Plain English — WHY

Commit `30aa87e6` fixes a recurrence-learning covariance statistic after the
declared rSLDS fitting work. The old branch contains surrounding calibration
and ticket-history commits that are not part of the fix. This ticket ports only
the commit's implementation, tests, and current canonical documentation.

The applicable architecture change-path rows are **Learnable or checkpoint
feature** and **Runtime numerical or status invariant**. The recurrence readout
owns the sufficient statistic. The learning adapter consumes it without
reconstructing covariance from already-subtracted moments.

## Outcomes / Acceptance Criteria — WHAT

1. The behavior introduced by `30aa87e6` is present relative to the new PR #65
   port without importing its old parent commits.
2. Recurrence learning receives the direct Gaussian covariance needed by its
   natural statistic and does not cancel covariance through a second moment
   subtraction.
3. Engine readout, hybrid learning, generic learning dispatch, and rSLDS fitting
   agree on the repaired statistic and its shape.
4. Exact-zero support stays exact; improper covariance and non-finite values
   fail loudly; float32 guards scale with dtype; the retained enlarged fixture
   remains finite and positive semidefinite within its stated tolerance.
5. Duplicated surfaces are checked as follows: flooding and scheduled readouts
   agree; host consumers receive the same JIT-produced statistic;
   single-instance tests cover the change and no new `vmap` consumer is added;
   checkpoint encoding is not changed and is therefore not applicable.
6. Focused conjugate, hybrid-readout, recurrence-numerics, and rSLDS-fit tests
   pass.
7. `README.md`, `docs/ARCHITECTURE.md`, and `docs/LIMITATIONS.md` describe the
   repaired covariance boundary as it exists on `main` commit `d805135`.
8. Old ticket changes, calibration records, and old dashboard runtime values are
   absent. Test-dashboard registration is retained only if the new focused test
   file requires it.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method.

- Apply `30aa87e6` without committing, remove excluded paths, and resolve the
  readout against the new declared-learning interfaces.
- Compare the statistic with a direct covariance oracle before testing it
  through the full fitter.

## Future work

cff-s6yy depends on this repair and will replace singleton-product recurrence
statistics with structured consecutive-state pair beliefs.

## Definitions

- **Direct covariance:** The covariance supplied by the Gaussian node belief,
  before any consumer combines it with means.
- **Natural statistic:** The additive linear or quadratic quantity used to
  update a conjugate posterior.
- **Positive semidefinite:** A symmetric matrix with no materially negative
  eigenvalue.

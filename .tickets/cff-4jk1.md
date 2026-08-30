---
id: cff-4jk1
status: open
deps: [cff-s6yy]
links: []
created: 2026-08-30T16:00:47Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-0qfy
tags: [consolidation, rslds, numerics]
---
# Port installed-parameter outer convergence evidence

## Background in Plain English — WHY

Commits `9dbcc017` and `7405d609` make the parameters installed into the next
inference solve the authority for rSLDS outer convergence. The old ancestry
also contains preregistered studies, audits, review records, and unrelated
ticket transitions. This ticket ports only the reusable coordinate-fit code,
model integration, tests, and canonical documentation from the two commits.

The applicable architecture change-path row is **Runtime numerical or status
invariant**. The coordinate-fit module owns installed-parameter residuals. Each
model names its installed parameter blocks. The fitters own terminal-state
coordination, while the engine still owns inner BFE and inference status.

## Outcomes / Acceptance Criteria — WHAT

1. The production behavior from `9dbcc017` and then `7405d609` is present
   relative to the structured-recurrence port without importing their ancestry.
2. A host-side primitive reports overall, per-array, and per-family undamped
   residuals for named installed runtime parameters.
3. rSLDS records the installed-parameter residual as its outer convergence
   authority while retaining the legacy mixed and truth-free functional
   residuals as named diagnostics.
4. Learned SLDS controls use the same installed-parameter residual semantics
   and return a state inferred from the parameters they report.
5. Non-finite residuals or free energy are divergence. Budget exhaustion remains
   distinct from convergence and is not reported as a completed convergence
   time.
6. Exact-map, exact-zero support, improper input, float32, float64, compact
   rSLDS, and the retained longer-chain coverage satisfy their explicit status
   and dtype-scaled tolerance contracts.
7. Duplicated surfaces are checked as follows: the host outer residual consumes
   the same installed arrays produced for JIT inference; flooding and scheduled
   inner solves retain their existing contract; rSLDS and SLDS are the two
   model consumers; no checkpoint format changes, so current/legacy decoding is
   not applicable.
8. `README.md`, `docs/ARCHITECTURE.md`, and `docs/LIMITATIONS.md` describe the
   installed-parameter authority and distinguish inner BFE from outer residual.
9. Old ticket-state changes, development-cohort records, preregistration,
   audits, and old generated runtime metadata are absent.
10. Focused coordinate-fit and rSLDS-fit tests pass, followed by one final
    default short-suite run at the complete stack head.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method.

- Port the two commits separately, preserving the residual primitive before
  changing which residual certifies convergence.
- Keep the coordinate code independent of rSLDS factor names and compare
  normalized arrays in the exact form installed into the engine.
- Use only existing non-final fixtures; do not run or inspect a registered
  scientific seed.

## Future work

No future work is identified for this consolidation. New globalization methods
or scientific reruns require separate tickets.

## Definitions

- **Installed parameter:** A normalized value or potential written into the
  engine for the next inference solve.
- **Outer residual:** A scale-normalized change across one complete learning
  replacement map.
- **Undamped:** Measured before any partial-step interpolation changes the
  proposed replacement.

---
id: cff-95ck
status: in_progress
deps: [cff-9skg]
links: []
created: 2026-08-22T14:26:39Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-0oqo
tags: [rslds, learning, numerics]
---
# Certify the returned rSLDS terminal parameter coordinate

## Background in Plain English — WHY

The weak-prior shared heterogeneous development fit reported convergence with
a final runtime-map residual of 0.0676. After the fitter installed that update
and ran its terminal hidden-state solve, the retained recurrence posterior and
installed recurrence parameters differed by 0.175, above the fitter's 0.1
outer tolerance. Stronger-prior fits did not show the same rebound.

The current cadence certifies one parameter coordinate, installs its proposed
replacement, and then returns a hidden-state posterior from the replacement
coordinate. A converged result must not certify one coordinate and return a
different, unchecked coordinate. This is separate from the common functional
recovery miss and does not justify outer damping or a looser tolerance.

## Outcomes / Acceptance Criteria — WHAT

1. A `converged` rSLDS result returns hidden-state beliefs, installed runtime
   parameters, and learned posteriors from one named parameter coordinate whose
   undamped runtime-map residual satisfies the existing 0.1 threshold.
2. The two-consecutive-map convergence rule remains unchanged and cannot be
   satisfied by a small accepted movement or by a diagnostic-only residual.
3. If a post-install coordinate rebounds above tolerance, the fitter continues
   within its declared map budget or returns `budget_exhausted`; it never
   reports the unchecked coordinate as converged.
4. Phase evidence names the evaluated, proposed, installed, certified, and
   returned coordinate identities and retains per-block runtime residuals.
5. Inner convergence, terminal hidden-state coherence, retry ownership,
   non-finite propagation, and immutable-prior replacement remain unchanged.
6. The matched SLDS control follows the same coordinate contract, or its
   experiment-owned implementation is explicitly linked and blocked before a
   new comparison freeze.
7. A compact rebound regression reproduces the coordinate mismatch without a
   NASCAR-scale fit; a development-only replay verifies the repaired status and
   returned-coordinate consistency without using final seeds.
8. Float32, exact-zero categorical support, improper posteriors, and a map with
   a non-finite proposed parameter fail according to the numerical robustness
   contract.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Prefer returning the already evaluated coordinate when it is certified, or
  explicitly evaluate the full map at any newly installed coordinate before
  returning it.
- Extend the focused coordinate-fit primitive rather than adding more policy
  to the already large `rslds_fit.py` module.
- Keep the inner Bethe free energy out of the outer certificate; it is not a
  joint parameter objective.

## Future work

`cff-0oqo` continues the separate functional-recovery investigation.
`cff-lqz8` determines whether correct latent responsibilities are sufficient
for recovery. A fresh calibration must remeasure map budgets after this status
change. No other future work is identified.

## Definitions

- **Parameter coordinate:** One complete set of values installed for the
  initial, dynamics, recurrence, and emission parameter blocks.
- **Runtime-map residual:** The scale-normalized difference between installed
  parameters and the complete undamped parameter update computed from their
  converged hidden-state beliefs.
- **Rebound:** A proposed coordinate passes the residual threshold, but the next
  full map evaluated at that coordinate exceeds it.

## Notes

Architecture handoff: the applicable row is `docs/ARCHITECTURE.md` section 2d,
"Status-gated run, identical-state retry, runtime-map certificate, terminal
solve." The named residual primitive and rSLDS coordinate view are also in
scope. IR, compiler, VM, schedules, kernels, ledger, factor equations, and
serialization are N/A unless implementation evidence proves otherwise.

Duplicated surfaces: rSLDS and matched SLDS cadences, shared/native
initialization, converged/exhausted/diverged states, terminal solves, phase
serialization, checkpoints, budget derivation, and future protocol freeze must
stay aligned.

Hostile-input decisions: exact-zero categorical support remains exact;
improper/vacuous and non-finite maps fail loudly; dtype guards use machine
precision; a long synthetic map sequence exercises at least ten times the
smallest two-step fixture without a NASCAR-scale graph.

Legend: prerequisite --> dependent

```text
[ ] cff-lqz8 truth-clamped oracle (ready) ----+
                                              +--> [ ] cff-0oqo recovery repair
[ ] cff-95ck terminal coordinate (ready) -----+               |
                                                              +--> [ ] fresh extension run
```

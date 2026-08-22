---
id: cff-lqz8
status: closed
deps: []
links: []
created: 2026-08-22T14:26:39Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-0oqo
tags: [rslds, learning, validation]
---
# Run a truth-clamped recurrence M-step recovery oracle

## Background in Plain English — WHY

All eight source-heterogeneous seed-5000 fits missed at least one 0.150
functional-recovery ceiling. Changing the shared recurrence prior from 0.001
through 1.0 left almost the same source, late-gate, and grid-region error
pattern. Several failed cells had posterior-effective exposure above 800 or
2,000. The retained recurrence posterior also matched an independent
closed-form update exactly in all eight fits.

Those facts rule out a simple rollover algebra bug and insufficient exposure
as general explanations. They do not distinguish biased latent responsibilities
from a design that cannot identify the declared transition surfaces. A
development-only M-step that uses the generating latent states and regimes can
make that distinction without changing the production learner.

## Outcomes / Acceptance Criteria — WHAT

1. A diagnostic-only path computes recurrence sufficient statistics and one
   posterior update from generating latent states and regimes on an authorized
   heterogeneous development dataset.
2. The diagnostic reuses the production recurrence statistic equations and
   prior mapping, and an independent implementation reproduces its posterior
   natural parameters.
3. The resulting transition functions are scored on the unchanged common-state
   grid with worst-source and source-contrast ceilings of 0.150.
4. The artifact issues one falsifiable decision: a passing truth-clamped result
   attributes the production miss to latent inference/statistics, while a
   failing result attributes it to the recurrence bound, finite-data design,
   or identifiability and reports which cells fail.
5. Truth-clamped values never enter ordinary fitting, initialization,
   checkpoint resume, prior selection, or any claim-bearing result.
6. Exact-zero support, improper statistics, non-finite inputs, float32, and a
   source/gate batch at least ten times the smallest fixture follow the CFFG
   numerical robustness contract.
7. Dataset, truth, prior, grid, code, numerical mode, result, and independent
   replay identities are manifested. Calibration and final seeds are rejected
   before generation.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Build the oracle as post-generation diagnostic code in the sibling
  experiment. Do not add a truth-aware option to `fit_rslds`.
- Compare both hard generating assignments and their exact one-hot factor
  beliefs to the production pooled recurrence readout.
- If the oracle passes, create a later ticket for a factor-local soft-statistic
  comparison against an exact small-chain posterior. If it fails, inspect the
  variational PG bound and generator design before changing the engine.

## Future work

`cff-0oqo` owns the production recovery repair after this discriminator.
`cff-95ck` independently repairs the returned-coordinate certificate. A fresh
heterogeneous calibration and freeze remain downstream. This ticket does not
change the generator, recovery ceiling, or final evidence.

## Definitions

- **M-step:** The parameter-update part of an alternating latent-state and
  parameter inference procedure.
- **Truth-clamped:** A diagnostic in which synthetic generating latent values
  replace inferred latent beliefs to isolate parameter learning.
- **Posterior natural parameters:** The additive precision-like coordinates of
  the conjugate recurrence-weight posterior.

## Notes

Architecture handoff: use `docs/ARCHITECTURE.md` section 2d's recurrence
statistic readout and declared-learning boundaries plus the sibling
experiment's truth-clean diagnostic boundary. Compiler, VM, scheduling,
kernels, ledger, serialization, and the public observed-only fit API are N/A
unless the production statistic contract changes.

Duplicated surfaces: every source and gate, hard assignments and one-hot
beliefs, production and independent posterior algebra, common-state scoring,
float32/x64, large batches, and manifested truth separation must stay aligned.

Hostile-input decisions: exact log-zero support stays exact; vacuous or
improper statistics and non-finite values fail loudly; dtype guards derive
from machine precision; the declared large batch exercises at least ten times
the smallest source/gate fixture.

Legend: prerequisite --> dependent

```text
[ ] cff-lqz8 truth-clamped oracle (ready) ----+
                                              +--> [ ] cff-0oqo recovery repair
[ ] cff-95ck terminal coordinate (ready) -----+               |
                                                              +--> [ ] fresh extension run
```

**2026-08-22T14:46:01Z**

Completed passing discriminator. Experiment commits 5c1e8cb and ec2d7bc add and preserve a replay-valid truth-clamped recurrence oracle. On authorized heterogeneous development seed5000, the unchanged PG/JJ recurrence coordinate converged in 131 of 160 maps and passed the 0.150 functional ceilings: worst-source RMSE 0.033226 and worst contrast RMSE 0.038553. Independent NumPy reproduction matched exposure/linear statistics exactly and precision, eta, omega, and posterior means within 4.17e-16 to 3.22e-15. This falsifies finite-data design, identifiability, prior mapping, and recurrence-update algebra as the common failure under exact latent beliefs. The next owner is a separate soft-statistic clamping-ladder ticket; no fit, chain, adapter, control, calibration seed, or final seed ran.

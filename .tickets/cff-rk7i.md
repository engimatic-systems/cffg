---
id: cff-rk7i
status: in_progress
deps: [cff-lqz8]
links: []
created: 2026-08-22T14:46:11Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-0oqo
tags: [rslds, learning, validation]
---
# Attribute recurrence recovery bias with a latent clamping ladder

## Background in Plain English — WHY

The truth-clamped recurrence oracle passed the heterogeneous recovery ceilings
with worst-source error 0.033 and worst source-contrast error 0.039. The same
recurrence equations and prior missed those ceilings when they received the
fitted model's soft beliefs. Independent algebra reproduced the truth-clamped
posterior to numerical precision.

The production miss therefore enters before the recurrence M-step. The main
remaining candidates are regime-transition responsibilities, continuous-state
means and covariances, or the mean-field factorization that combines them. A
clamping ladder can replace one belief family at a time and identify the first
stage that restores recovery.

## Outcomes / Acceptance Criteria — WHAT

1. A development-only diagnostic compares the fitted recurrence sufficient
   statistics with the truth-clamped authority by source, gate, and natural-
   parameter block on the same authorized dataset and prior.
2. A declared clamping ladder separately evaluates at least these coordinates:
   fully inferred beliefs; generating regime transitions with inferred
   continuous moments; inferred regime transitions with generating continuous
   states; and fully generating latent values.
3. Each coordinate reports exposure, linear, quadratic, posterior-mean, full
   transition, conditional-gate, worst-source, and source-contrast differences.
4. The artifact names the earliest clamping step that meets both unchanged
   0.150 functional ceilings, or reports that none does. It does not average a
   failed source or gate into a passing conclusion.
5. If retained checkpoints do not contain the factor-local beliefs required by
   the ladder, a prospective callback records only truth-free per-factor
   categorical and Gaussian moments. Generating truth is joined later after
   dataset-digest verification.
6. The diagnostic distinguishes marginal regime probabilities from the joint
   consecutive-regime belief required by the recurrence factor and rejects a
   product-of-marginals substitute as non-authoritative.
7. Exact-zero support, improper Gaussian moments, non-finite statistics,
   float32, and a factor batch at least ten times the smallest fixture follow
   the numerical robustness contract.
8. Development seeds, streams, priors, checkpoint payloads, clamping choices,
   code, numerical mode, results, and an independent replay are manifested.
   Calibration and final seeds are rejected before generation.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Start with the pooled retained statistics from the prior screen to rank which
  natural-parameter block differs most before adding a prospective callback.
- Reuse `gaussian_node_stats` and `recurrence_weight_stats` rather than deriving
  a second production statistic formula in the experiment.
- If regime-transition responsibilities are decisive, compare one compact
  recurrence factor against exact enumeration before changing a message rule.
- If continuous moments are decisive, inspect affine-aligned mean, covariance,
  and cross-time covariance separately.

## Future work

`cff-0oqo` uses the selected stage to define the production inference repair.
If a factor-local rule is wrong, a new core-engine bug ticket owns that rule and
its derivation. If no step passes, the recurrence approximation or recovery
claim design must be reconsidered in a separate ticket. Fresh calibration and
final evidence remain downstream.

## Definitions

- **Clamping ladder:** A diagnostic sequence that replaces inferred belief
  families with generating values one family at a time.
- **Joint consecutive-regime belief:** The probability table over a source
  regime and its next destination regime for one transition factor.
- **Natural-parameter block:** One additive exposure, linear, or quadratic
  contribution used by the recurrence posterior update.

## Notes

Architecture handoff: use `docs/ARCHITECTURE.md` section 2d's
`gaussian_node_stats`, `recurrence_weight_stats`, and declared-learning
boundaries plus the sibling experiment's truth-clean fit-callback boundary.
Compiler, VM, scheduling, kernels, ledger, and serialization are N/A unless a
production readout contract changes.

Duplicated surfaces: shared and native initialization, every source/gate/time
factor, categorical joint beliefs, Gaussian mean/covariance/second moments,
pooled statistics, float32/x64, checkpoint resume, and post-fit truth joining
must stay aligned.

Hostile-input decisions: literal categorical zeros stay zero; vacuous or
improper Gaussian moments and non-finite statistics fail loudly; dtype guards
derive from machine precision; the large batch covers at least ten times the
smallest factor fixture.

Legend: prerequisite --> dependent

```text
[x] cff-lqz8 truth-clamped M-step oracle
      |
      +--> [ ] cff-rk7i latent clamping ladder (ready)
                  |
                  +--> [ ] cff-0oqo production recovery repair
                              |
                              +--> [ ] fresh extension run
```

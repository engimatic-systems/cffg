---
id: cff-i4o0
status: open
deps: [cff-sawy]
links: []
created: 2026-08-24T13:53:16Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-3v7q
tags: [rslds, validation, stability, experiment]
---
# Run one prospective stability-gated classic D10 cohort

## Background in Plain English — WHY

The original D10 classic cohort and its retained block-swap diagnostic showed
that recurrence strongly affects long trajectories but does not universally
own their boundedness.  cff-x3a8 therefore derived a joint recurrence/dynamics
stability certificate, and cff-sawy must first prove that the certificate can
gate real D10 proposals without corrupting learning or calling rejection
convergence.

This ticket is the single prospective scientific discriminator after that
synthetic bridge passes.  It asks whether the unchanged parity architecture
can learn points that are both Foster-certified and robust on the already
defined classic outcomes.  It is not permission for another prior sweep or a
sequence of outcome-driven reruns.

## Outcomes / Acceptance Criteria — WHAT

1. Execution is impossible until cff-sawy closes with a certified
   initialization, accepted nontrivial proposal, exact status/work accounting,
   and authenticated checkpoint contract.
2. Before generation, freeze a new versioned protocol with fresh development
   coordinates, both shared and native initialization tiers, pair-identical
   streams where comparisons require them, the exact CFFG/code/runtime
   identities, and the certificate settings selected without NASCAR outcomes.
   Calibration and final seeds are rejected before generation.
3. Use the unchanged cff-45ia outcome gates: every declared rollout is bounded
   and exceeds its tier periodicity floor; held-out transition RMSE is at most
   0.15; predictive log-likelihood delta is at least -0.1; and absolute
   label-aligned regime-accuracy delta is at most 0.02.  No threshold, rollout
   length, or score definition changes after registration.
4. Retain per fit the initial/current/proposed certificate evidence, accepted
   and rejected proposal counts, D10 inner/outer statuses, phase work,
   certified/returned coordinate identities, all four family payloads, and
   compact 10,000-step boundedness/periodicity evidence.  A rejection is not a
   convergence observation or an outer-map time measurement.
5. Use a staged funnel on one fresh declared development coordinate across
   both tiers.  Expand exactly once to the remaining predeclared development
   coordinates only if every funnel fit and every gate passes.  Stop on the
   first infrastructure failure, certificate failure, censoring, non-finite
   outcome, or scientific miss; no alternative settings or replacement seeds
   run under this registration.
6. Checkpoint, resume, and aggregate validation are append-only and
   manifest-backed.  A partial family write, settings mismatch, stale D8/D9
   semantic identity, extra file, coordinate swap, or tampered rollout rejects
   before fitting or aggregation.
7. Passing development evidence may authorize a separately reviewed fresh
   calibration/freeze ticket; it is not final evidence.  Any miss closes this
   intervention route and records robust classic long-rollout adequacy as an
   unresolved limitation of the parity architecture.

## Methodological Suggestions — HOW

- Reuse the cff-45ia paired runner and score authority, but introduce a new
  protocol/output identity rather than migrating its completed artifacts.
- Reuse the cff-sawy checkpoint and work-accounting contract verbatim.  Keep
  certificate settings in the protocol, not command-line tuning knobs.
- Funnel the cheapest complete paired coordinate first, then use append-only
  expansion so an interruption cannot silently select a new fit.

## Future Work

If every declared development cell passes, create a new calibration/freeze
ticket with untouched registered final seeds.  If any cell misses, close
cff-3v7q as a completed unresolved classic result and narrow the parity claim
to factor-type architecture plus measured development behavior.

## Definitions

- **Fresh development coordinate:** a preregistered non-calibration,
  non-final dataset/fit/rollout stream tuple not selected from prior outcomes.
- **One-shot stop rule:** the registration permits one funnel and, only after a
  complete pass, one declared expansion; scientific misses are terminal.
- **Stability-gated fit:** every installed outer proposal has passed the exact
  cff-sawy certificate contract.

## Change-Path and Review Handoff

This is an experiment-layer consumer of the observed-only rSLDS fit and
checkpoint contract.  CFFG message rules, D10 objective, model IR/compiler,
adapters, SLDS controls, calibration, and final execution are N/A.  Duplicated
surfaces that must agree are both initialization tiers, funnel/expansion,
fresh/resumed execution, fit/checkpoint/aggregate evidence, and each unchanged
classic metric gate.

Legend: prerequisite --> dependent

```text
[x] cff-x3a8 certificate authority
      |
      +--> [ ] cff-sawy fail-closed proposal gate
                  |
                  +--> [ ] cff-i4o0 prospective cohort (this ticket)
                              |
                              +--> [ ] cff-3v7q classic decision
```

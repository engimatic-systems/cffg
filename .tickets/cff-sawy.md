---
id: cff-sawy
status: open
deps: [cff-x3a8]
links: []
created: 2026-08-24T13:53:16Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-3v7q
tags: [rslds, validation, stability, learning]
---
# Integrate the Foster certificate as a fail-closed D10 proposal gate

## Background in Plain English — WHY

cff-x3a8 derived and independently audited a sound global Foster-Lyapunov
certificate for the actual recurrent switching law and affine dynamics.  That
authority answers whether one fully specified rSLDS point is certified, but it
does not yet stabilize learning: it chooses no replacement point and is not a
solver or objective term.

The smallest parity-compatible use is a fail-closed outer eligibility gate.
Before an undamped four-family D10 proposal is installed, the host fitter asks
the certificate whether that exact proposal is globally admissible.  A
rejected proposal leaves the evaluated point unchanged and cannot masquerade
as convergence or progress.  This ticket proves that execution contract on
synthetic systems before any NASCAR cohort is allowed.

## Outcomes / Acceptance Criteria — WHAT

1. Prospectively fix or deterministically derive every certificate input,
   including the mode Lyapunov matrices, Foster rate, compact radius, and
   subdivision budget.  The derivation may use only the model point and
   declared numerical settings; no rollout outcome, generating state, label,
   calibration seed, or final seed may influence it.
2. Certify the initializer/current installed point before learning.  A point
   that is `violated`, `infeasible`, or `diverged` cannot start a gated outer
   cadence and produces a distinct truthful status and phase record.
3. Evaluate the complete **undamped** initial/dynamics/recurrence/emission
   proposal before any parameter write.  `satisfied` installs all four
   families atomically.  `violated` records `certificate_rejected` and retains
   byte-identical installed parameters.  `infeasible` and `diverged` fail
   loudly without a partial write.
4. An unverified contraction ceiling sentinel is never safe.  Rejection never
   increments consecutive residual hits, counts as free-energy improvement,
   or certifies convergence.  Budgets distinguish evaluated D10 maps,
   certificate attempts, accepted proposals, and rejected proposals.
5. Accepted points retain the existing D10 inner coordinate, Bethe free-energy
   semantics, returned-coordinate certificate, and four-family statistic
   identities.  The gate contributes no energy and does not alter the rSLDS
   probabilistic family.
6. A versioned checkpoint/evidence contract records certificate settings,
   point identity, status, bounds, accepted/rejected ownership, and dtype.  A
   missing, reordered, tampered, D8/D9, or differently configured resume is
   rejected before mutation.
7. Independent synthetic tests exercise both initialization tiers and every
   parameter-writer surface; one certified accepted-progress trajectory; one
   certified-current/rejected-proposal trajectory; atomic four-family writes;
   float32; exact categorical zeros; non-finite/improper points; near-boundary
   sentinels; K=8 scale; and checkpoint/resume tampering.
8. Stop without creating scientific evidence if the synthetic gates cannot
   demonstrate both a certified initialization and at least one accepted
   nontrivial proposal.  In that case close this route as incompatible with a
   usable outer fitter rather than weakening the certificate.
9. No NASCAR fit or rollout, adapter chain, SLDS control, calibration seed,
   final seed, frozen artifact, or confirmatory conclusion changes or runs in
   this ticket.

## Methodological Suggestions — HOW

- Keep the certificate pure and call it at the rSLDS observed-only fit
  boundary immediately before the existing atomic `Learnables.write`.
- Treat `P`, the Foster rate, radius, and compact cover as a model-owned
  deterministic policy object.  If finding them requires optimization, give
  that solve its own converged/budget-exhausted/diverged status and preserve
  the final verified certificate rather than trusting the optimizer result.
- Retain the existing runtime-map residual as the convergence authority.  The
  gate answers admissibility only.

## Future Work

cff-i4o0 owns the one prospective development cohort and remains blocked on
this ticket.  If this gate works synthetically but that cohort misses any
unchanged gate, cff-3v7q records an unresolved classic limitation instead of
starting another prior or seed search.

## Definitions

- **Proposal eligibility gate:** a pre-installation predicate over a complete
  candidate point; it is not a line search, projection, or objective penalty.
- **Certified initialization:** the exact point used by the first D10 inner
  solve satisfies the registered Foster certificate.
- **Accepted progress:** a non-identical undamped outer proposal both satisfies
  the certificate and is installed for the next D10 solve.

## Change-Path and Review Handoff

Applicable `docs/ARCHITECTURE.md` row: the observed-only rSLDS fit boundary
under the **Learnable or checkpoint feature** row.  The authority remains
`cffg.switching_stability`; `cffg.models.rslds_fit` owns proposal gating and
phase/status evidence; `Learnables` retains atomic validation and writes.

Duplicated surfaces: shared and native initialization tiers; initializer and
outer-map parameter writes; accepted and rejected phases; current and restored
checkpoints.  Engine flood/scheduled, host/JIT message rules, Colony, factor
IR/compiler topology, serialization of model structure, adapter/control
execution, and final protocols are N/A unless implementation discovers a
genuine new owner, which must become a prerequisite ticket rather than scope
drift.

Legend: prerequisite --> dependent

```text
[x] cff-x3a8 certificate authority
      |
      +--> [ ] cff-sawy fail-closed proposal gate (this ticket)
                  |
                  +--> [ ] cff-i4o0 prospective cohort
                              |
                              +--> [ ] cff-3v7q classic decision
```

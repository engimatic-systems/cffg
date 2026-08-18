---
id: cff-266l
status: open
deps: [cff-9o39, cff-x5z2]
links: []
created: 2026-08-18T14:10:56Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, calibration]
---
# Reject diverged calibration arms before metric eligibility

## Background in Plain English

The first source-heterogeneous calibration coordinate returned two CFFG fits
with converged inner inference but a diverged outer learning loop. Calibration
correctly kept their last accepted states for diagnostic scoring. A readiness
shortcut then treated every calibration outer status as eligible, so the tier
and coordinate were mislabeled ready even though the final calibration
validator rejects diverged arms.

A calibration failure may remain available for diagnosis, but it cannot select
margins, budgets, or a final claim. Resume must also know when a checkpoint was
produced by different experiment or CFFG code. At experiment commit `1af3d2f`,
the checkpoint identity hashes only the protocol and registration, so a
numerical repair could silently reuse results produced with CFFG revision
`94458b9`.

This ticket makes arm readiness, checkpoint identity, and append-only retry
agree before cff-s6rs regenerates its calibration artifact.

## Outcomes / Acceptance Criteria

1. Calibration metrics are eligible only when inner inference converged and the
   outer fit status is `converged` or `budget_exhausted`. Final metrics remain
   eligible only when both inner and outer statuses are `converged`.
2. A top-level ready tier or coordinate cannot contain a diverged, failed, or
   otherwise ineligible nested primary arm. Raw diagnostic fields may remain
   available without entering primary effects.
3. An outer-diverged, inner-converged regression produces a not-ready tier and
   coordinate, no primary effect for that arm, and an explicit failure reason.
4. Coordinate checkpoint identity includes the committed experiment execution
   revision and declared CFFG revision in addition to protocol and registration
   identity. Resume rejects or append-only supersedes a checkpoint produced by
   different code instead of silently reusing it.
5. The false-ready source-heterogeneous checkpoint and every result produced by
   the superseded CFFG revision remain byte-for-byte available as manifested
   invalid calibration evidence. A corrected run writes new attempts or a new
   artifact root; it never overwrites the old payloads.
6. Focused tests cover calibration and final status matrices, stale code
   identity, interrupted resume, ready-checkpoint supersession, raw aggregate
   recomputation, and retention of diagnostic evidence after one family fails.
7. The experiment README explains why last-accepted-state metrics are
   diagnostic only after divergence and how code-qualified resume works.
8. No final seed is generated, and no immutable dual-family preregistration is
   written by this ticket.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound method
that satisfies the outcomes.

- Centralize the phase-specific outer-status predicate instead of repeating it
  in the CFFG arm, tier, coordinate, and validator layers.
- Preserve the entire invalid run under a separately named manifested artifact
  root before starting the corrected calibration. This is simpler than mutating
  a false-ready checkpoint in place.
- Include code identity in the checkpoint preimage rather than relying on a
  human note or file modification time.

## Future work

cff-s6rs reruns the selected calibration under one coherent experiment and CFFG
revision, validates all eight coordinates, and derives the runtime and
statistical summary. cff-huxt cannot freeze a final protocol until that artifact
is complete. cff-oyie remains the only final-seed owner.

## Definitions

- **Last accepted state:** The most recent parameter state committed before a
  later candidate update failed.
- **Code-qualified checkpoint:** A checkpoint whose identity includes the
  revisions of the code that generated and interpreted it.
- **Superseded artifact:** An immutable historical artifact retained for audit
  but excluded from the replacement calibration summary.

## Notes

Pre-implementation handoff: `docs/ARCHITECTURE.md` has no engine change-path row
for this sibling experiment harness. The duplicated surfaces are calibration
versus final status rules, both primary tiers, both dataset families, tier versus
coordinate readiness, live versus resumed execution, protocol versus code
checkpoint identity, and raw diagnostics versus primary effects. CFFG compiler,
VM, schedule, kernel, ledger, serialization, factor equations, and canonical
documents are N/A here because cff-x5z2 owns the engine repair.

Hostile-input decisions:

- Missing or unknown statuses are ineligible, never implicitly successful.
- Non-finite diagnostic payloads remain rejected by
  `dual_execution._sanitize_coordinate_result`.
- Float32 versus float64 does not change status semantics.
- A full 1,279-transition-factor coordinate is exercised through the
  calibration-only regression or manifested rerun; no final seed is used.

Research finding — Question: why was a diverged coordinate marked ready?
Answer: `_run_tier` lets `phase == "calibration"` bypass the outer-status check,
then `run_family_coordinate` derives readiness from tier status and finite
effects. The later `validate_selected_execution` correctly accepts only
`converged` or `budget_exhausted`, so the generated summary would fail. Resume
also trusts the false top-level ready value and its checkpoint identity omits
the CFFG code revision. Source:
`src/cffg_nascar_validation/confirmatory.py`, `dual_execution.py`,
`dual_calibration.py`, `execution_checkpoint.py`, and calibration-only
source-heterogeneous seed 3101 replicate 0. Confidence: high.

Roadmap after the calibration finding:

```text
Legend: prerequisite --> dependent

[x] cff-11jj declared rSLDS learning
      |
      +--> [ ] cff-x5z2 numerical recurrence fix (ready)
               |
               +--> [ ] cff-266l honest calibration readiness
                        |
                        +--> [~] cff-s6rs replacement calibration
                                 |
                                 +--> [ ] cff-huxt immutable freeze
                                          |
                                          +--> [ ] cff-oyie final run
                                                   |
                                                   +--> [~] cff-nkbj evidence epic
```

`cff-x5z2` is the ready root. `cff-s6rs` remains in progress but is immediately
blocked by the two repairs. The freeze, final run, and evidence epic remain
downstream; the ticket dependency records are authoritative.

---
id: cff-huxt
status: open
deps: [cff-s6rs]
links: []
created: 2026-08-17T20:52:04Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, preregistration]
---
# Freeze the dual-family confirmatory protocol

## Background in Plain English

cff-s6rs produces the complete calibration evidence for both dataset families. The scientific choices must become immutable before cff-oyie generates a final seed. The first cff-0l31 freeze remains useful provenance, but it covered only canonical NASCAR and omitted later report and dual-family requirements.

This ticket validates the complete calibration artifact and writes one replacement preregistration and manifest. The classic family uses the exact canonical generator and fits 10,000 training steps, matching the paper's training scale. Its separate 2,500-step held-out extension means it is not an exact replay of the Figure 3 workflow. This ticket does not fit or generate final data.

## Outcomes / Acceptance Criteria

1. The replacement preregistration fixes both family generators, versions, scales, splits, final seeds, family-qualified data identities, streams, arms, controls, budgets, diagnostics, estimands, margins, multiplicity, claim gatekeeping, failure rules, restart layout, and reporting surfaces.
2. Validation reconstructs the registered protocol from the exact cff-s6rs calibration artifact and rejects any changed, missing, incomplete, non-finite, or unmanifested input.
3. The freeze records and verifies CFFG, experiment, canonical generator, heterogeneous generator, upstream paper adapter, legacy environment, mapping, locks, calibration, evaluation-grid, and source commit hashes.
4. The cff-0l31 preregistration remains byte-for-byte available and is recorded as superseded without execution, with the coverage and report reasons stated.
5. Final commands name cff-oyie as the only authorized owner, use append-only family-qualified resume semantics, and permit no seed, threshold, budget, arm, or claim changes after the new content hash.
6. README explains the exact canonical generator at the paper's 10,000-step training scale, the registered 2,500-step held-out extension, why this is not an exact Figure 3 replay, the heterogeneous capability evidence, fixed-sequence claim wording, and exclusions in plain language.
7. The new preregistration and manifest validate from a clean checkout, and their content hash is recorded in the ticket and PR body.
8. Before the content hash is written, the family-aware final command validates the whole dual preregistration and is complete for per-final heterogeneous eligibility checks, the narrow unmodified baseline, family-separated claims and failures, append-only resume, retained posterior samples, raw aggregate recomputation, and explicit not-ready records.
9. Focused tests reject a tampered registration, changed calibration input, missing or non-finite payload, incomplete coordinate, mismatched family stream, and corrupted checkpoint. They also prove interrupted resume and one-family failure retention without generating a final seed.
10. No registered final seed is generated or fit by this ticket.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound method that satisfies the outcomes.

- Write to a new artifact directory and hash the old freeze rather than editing it.
- Make freeze output deterministic except for an explicitly recorded registration date.
- Verify each nested family/seed stream and data preimage independently.

## Future work

cff-oyie executes the frozen final protocol without changing code or scientific choices. cff-nkbj audits the final result against the registered claim ladder. A failed final layer receives a new repair ticket rather than a changed freeze.

## Definitions

- **Immutable preregistration:** A content-hashed document whose scientific choices cannot change after final execution starts.
- **Replacement freeze:** The dual-family preregistration that supersedes but does not delete cff-0l31.

## Notes

Pre-implementation handoff: `docs/ARCHITECTURE.md` has no engine change-path row for this sibling validation harness. The freeze consumes the unchanged public CFFG fitting result. It updates the experiment README in this ticket; the CFFG README, architecture boundary, and limitations update with the final evidence in cff-oyie.

Duplicated execution surfaces checked before the freeze:

- both family generators, descriptors, source-design evidence, evaluation grids, scales, splits, data identities, and final streams;
- both primary tiers, four Gibbs chains, two learned controls, the narrow unmodified baseline, family-specific absolute outcomes, and classic-first gatekeeping;
- per-family checkpoints, append-only attempts, interruption and resume, retained posterior paths, raw aggregates, final failure isolation, and artifact validation;
- CFFG compiler, VM, schedule, kernels, ledger, serialization, learning declarations, and factor implementations, which are N/A because this ticket consumes the already declared fitting path.

Hostile-input decisions:

- A changed, missing, incomplete, non-finite, or unmanifested calibration or registration payload makes the freeze invalid and produces no preregistration.
- A source-heterogeneous final dataset that misses the frozen exposure, success, failure, or design-rank gate receives a not-ready record before its effects can support a capability claim.
- Final execution uses the calibrated float64 protocol. Existing float32 generator, functional-score, and CFFG regressions remain required survival evidence and cannot be removed by this ticket.
- Registered scales and long-rollout bounds are fixed by calibration. The freeze cannot extrapolate them to untested horizons or model families.

Legend: prerequisite --> dependent

```text
[x] cff-9o39 runner
      |
      +--> [ ] cff-s6rs calibration (in progress)
                  |
                  +--> [ ] cff-huxt freeze
                              |
                              +--> [ ] cff-oyie final execution
                                          |
                                          +--> [ ] cff-nkbj evidence audit
```

cff-s6rs is the active root. No final seed may be generated until cff-huxt closes with the committed replacement hash.

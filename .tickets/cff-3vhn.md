---
id: cff-3vhn
status: in_progress
deps: [cff-0l31]
links: []
created: 2026-08-17T18:23:46Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, statistics]
---
# Complete confirmatory secondary outcomes before final execution

## Background in Plain English

The cff-0l31 freeze correctly fixes the primary comparison, but a pre-run audit against cff-oyie found that the runner does not emit every secondary outcome that cff-oyie requires. It records primary predictive and recovery metrics, control metrics, outer statuses, and chain diagnostics. It does not yet report aligned parameter-family errors, bounded long-rollout periodicity and rate, complete inner-work and phase-timing summaries, or the pinned unmodified recurrence-only workflow as a separately scoped secondary result.

No final seed has run. The frozen phase boundary says any code change after its hash requires a new prerequisite and a new preregistration. This ticket completes those report surfaces and refreezes before cff-oyie starts.

## Outcomes / Acceptance Criteria

1. Every primary rSLDS tier records aligned parameter-family errors, bounded long-rollout periodicity, outer iterations, inner-work counts, phase timings, and peak resident memory as secondary outcomes.
2. Both learned SLDS controls record the applicable secondary outcomes on the identical observations, split, and seed. A field that cannot apply names the structural reason instead of disappearing.
3. The pinned unmodified recurrence-only workflow is reported from its hashed artifact or a newly executed compatible artifact as a narrower, separately identified baseline. The report states that it is not a same-model or same-update comparator.
4. Aggregate code derives bounded-rollout rates, secondary summaries, every primary interval, and the final conclusion from raw per-seed rows. Recalculation tests detect changed or omitted raw rows.
5. Artifact provenance includes the CFFG, experiment, generator, unmodified upstream, paper adapter, both locked environments, preregistration, command, hardware, and complete manifest identities.
6. Compact tests exercise secondary aggregation, unavailable-field accounting, truth-boundary enforcement, and manifest recomputation without using a final seed.
7. The replacement preregistration preserves the already calibrated seeds, mappings, primary metrics, interval family, failure policy, and margins unless a separate prerequisite explicitly owns a scale or runtime correction.
8. No final seed is generated or fit by this ticket.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound method that satisfies the outcomes.

- Reuse the alignment, parameter-error, rollout, and periodicity functions already exercised by the historical observed-only comparison.
- Compute secondary values immediately after each fit and store the raw inputs needed for independent aggregate recomputation.
- Treat the existing hashed reference-smoke artifact as the unmodified comparator when its registered narrow scope is sufficient.

## Future work

A separate prerequisite owns any reference-driven scale or runtime correction. cff-oyie owns the immutable final execution after both prerequisites are complete. cff-nkbj owns the final evidence bundle and scoped conclusion.

## Definitions

- **Bounded-rollout rate:** The fraction of registered seeds whose fitted point model produces a finite, registered-length rollout within the stated amplitude and variance limits.
- **Inner work:** The number of hidden-state inference iterations and retries performed inside parameter-learning coordinates.
- **Parameter-family error:** An alignment-aware distance between a fitted parameter family and the generating family, computed only after fitting.
- **Secondary outcome:** A diagnostic measurement that cannot replace or rescue a failed primary metric.
- **Unmodified comparator:** The pinned authors-code workflow executed without changing its statistical implementation.


## Notes

**2026-08-17T18:24:26Z**

Pre-implementation handoff: docs/ARCHITECTURE.md change-path index is N/A because this implementation remains in the sibling experiment harness and consumes unchanged CFFG public fitting results. Duplicated execution surfaces: both primary rSLDS tiers; same-K and closest-budget SLDS controls; unmodified recurrence-only baseline; raw per-seed secondary rows; aggregate recomputation; manifest validation. CFFG compiler, VM, schedule, kernel, ledger, serialization, production learning, and canonical docs are N/A. Legend: prerequisite --> dependent. [x] cff-0l31 --> [ ] cff-3vhn (ready/in progress) --> [ ] cff-ah4t --> [ ] cff-oyie --> [ ] cff-nkbj. cff-3vhn owns reporting completeness only; cff-ah4t owns scale selection and replacement freeze.

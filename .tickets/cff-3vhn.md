---
id: cff-3vhn
status: closed
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

**2026-08-17T18:35:59Z**

Completed in sibling experiment repository branch cff-3vhn-secondary-outcomes at fc84a28. New secondary.py evaluates alignment-aware parameter-family errors, transition-probability RMSE, generating-truth recovery after fitting, 10,000-step rollout periodicity, boundedness, exact rSLDS inner work from retained phases, and explicit matched-SLDS inner-trace unavailability. Paper-adapter secondary outcomes are computed within each retained chain before aggregation so label-switched parameters are never averaged across chains. artifact_reporting.py validates raw-row aggregate recomputation and hashes every final output payload. The final report now includes the hash-verified pinned unmodified recurrence-only smoke as a narrow baseline with same_model_or_update_parity_claim=false, full registration provenance, seed peak RSS, CFFG phase timings, reference chain timings, primary/secondary bounded-rollout rates, and aggregate parameter errors.

The preregistration schema now fixes 10,000 rollout steps, variance-growth and amplitude-ratio bounds of 100, and disjoint rollout streams for each primary/control arm. The original cff-0l31 freeze is deliberately invalid under the recorded phase rule and remains committed only as superseded provenance; no final seed ran.

Validation: uv run pytest -q reported 30 passed in 24.00 seconds. A calibration-seed T=80 integration exercised both primary tiers and both SLDS controls; all four arms were ready with secondary outputs and no metric error. Exact generating-parameter alignment recovered every family below 1e-10 in compact tests. Raw aggregate mutation, manifest mutation, unmodified baseline hashes, truth-free fitter signatures, float32 fitting, and inner-work availability are covered. git diff --check passed.

Next: cff-ah4t benchmarks and freezes the reference-driven execution scale, incorporating this code. It must not execute a final seed.

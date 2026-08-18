---
id: cff-9cx5
status: closed
deps: []
links: []
created: 2026-08-18T17:08:12Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, preregistration, execution]
---
# Prepare schema-v5 fixed-dataset final execution

## Background in Plain English

The calibration now running is permanently tied to its committed schema-v4
code. That is useful provenance and must remain reproducible. The final evidence
scope changed before any final seed existed: each family now has three named
datasets, and a family passes only when every dataset meets every registered
rule. A confidence interval over three values would imply a population claim
that Cooper did not ask for.

The current final runner also lacks four checks required before an immutable
registration can be trusted. It does not audit heterogeneous source/gate
coverage on each final training prefix, compare runtime data identities with
frozen identities, reject code changed after registration, or include the
pinned unmodified secondary baseline in the dual report.

This ticket prepares and tests the versioned final protocol and runner. It does
not consume the unfinished calibration result, choose any threshold, write the
immutable registration hash, or generate a final dataset.

## Outcomes / Acceptance Criteria

1. Schema v4 remains a supported, immutable calibration contract whose stored
   raw aggregate and selected summary recompute without semantic drift.
2. A separate schema-v5 final protocol is derived from a complete registered-v4
   fixture without changing its family generators, scales, splits, final seeds,
   thresholds, budgets, arms, controls, grids, or stochastic coordinates.
3. The schema-v5 decision policy requires exactly three finite values per
   registered effect in frozen final-seed order and records each dataset's
   value, threshold, direction, and pass status.
4. A schema-v5 family passes only when all three datasets are ready and every
   individual registered effect passes. It computes no confidence interval,
   alpha-level decision, family unlock, or joint pass.
5. The schema-v5 protocol contains and validates one family-qualified data
   identity and stream manifest for each of the six final seeds, with no stream
   collision or overlap with calibration seeds.
6. Before fitting a source-heterogeneous final coordinate, the runner records
   and validates its training-prefix exposure, success, failure, and augmented
   design-rank gates. An ineligible coordinate becomes explicitly not ready and
   starts no fitting arm.
7. Final execution rejects a registration with a changed content hash, an
   unauthorized experiment commit, a changed declared CFFG commit, a dirty
   tracked worktree, or a runtime dataset identity that differs from the frozen
   identity.
8. The final dual report verifies and includes the pinned unmodified
   recurrence-only baseline as secondary evidence without relabeling it as the
   same-model reference.
9. Family-qualified append-only checkpoints retain completed work across an
   interruption, reject another registration or code revision, and keep one
   family's failure from erasing the other family's raw or secondary evidence.
10. Synthetic tests cover passing and failing per-dataset rules, missing and
    non-finite effects, heterogeneous eligibility failure before fit, identity
    mismatch, code mismatch, tampered registration, baseline provenance, and
    raw aggregate recomputation.
11. The experiment README explains the v4 calibration/v5 final boundary and
    fixed-dataset decision in plain language.
12. No registered final seed is generated or fit by this ticket.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound method
that satisfies the outcomes.

- Keep the schema-v4 functions unchanged and add explicit schema-v5 derivation
  and validation functions rather than silently changing old meanings.
- Dispatch claim evaluation by schema. Preserve the old not-ready shape only
  where byte-for-byte calibration recomputation needs it.
- Inject a coordinate runner and repository-state reader in tests so final
  preflights can be exercised without generating data or launching a fit.
- Reuse `source_design.gate_coverage` and its validator instead of maintaining
  another coverage calculation.

## Future work

cff-s6rs owns the measured calibration artifact. cff-huxt consumes that exact
artifact with this prepared code, writes and validates the immutable schema-v5
registration, and records its content hash. cff-oyie alone may then generate
the six final datasets.

## Definitions

- **Fixed-dataset decision:** A pass only when each named final dataset meets
  each registered rule; it makes no statement about an unobserved population.
- **Schema v4:** The committed calibration protocol and aggregation semantics
  used by cff-s6rs at experiment commit `248f9d3`.
- **Schema v5:** The final-only protocol that freezes the lean 3+3 decision and
  runtime identity checks.

## Notes

Pre-implementation handoff: `docs/ARCHITECTURE.md` change-path row is N/A. This
ticket changes the sibling validation harness and consumes the unchanged
serialized `LearnableSpec` path. It updates the experiment README; the final
CFFG documentation remains cff-nkbj work.

Duplicated execution surfaces checked before implementation:

- schema-v4 calibration validation and schema-v5 final validation;
- classic and heterogeneous families, all six final identities, both primary
  tiers, four reference chains, both controls, and the narrow baseline;
- per-dataset decisions, family independence, raw aggregation, append-only
  attempts, registration provenance, and code-qualified resume;
- compiler, VM, schedule, kernels, ledger, factor equations, and CFFG
  serialization, which are N/A because the consumer fitting API is unchanged.

Hostile-input decisions:

- A zero-probability categorical support value remains literal; this ticket
  does not add a floor or pseudocount.
- Improper, vacuous, diverged, budget-exhausted, or non-finite final fits are
  ineligible under cff-266l and cannot export primary effects.
- Final execution is registered float64. Existing factor-level float32 tests
  remain the dtype survival evidence and cannot be weakened here.
- Registered scales are the uncomfortable-size cases. Tests use injected small
  coordinates to exercise orchestration without claiming small-scale evidence.

Legend: prerequisite --> dependent

```text
[x] v4 dual runner + readiness + architecture mapping
                         |
                         +--> [ ] cff-9cx5 (ready)
                                      |
[ ] cff-s6rs calibration -------------+--> [ ] cff-huxt immutable freeze
                                                   |
                                                   +--> [ ] cff-oyie final run
```

**2026-08-18T17:34:03Z**

Implementation complete in the sibling experiment repository on branch cff-9cx5-final-execution.

Commits:
- 260bb55 cff-9cx5: prepare fixed-dataset final execution
- 47dd1a7 cff-9cx5: document fixed-dataset final boundary

Evidence:
- Schema v4 calibration output remains free of final-only report fields.
- Schema v5 is an exact derivation of the embedded validated v4 contract, freezes six family-qualified identities and manifests, and rejects collisions with calibration streams.
- Final decisions record every effect on each of three named datasets and compute no interval, alpha, unlock, or joint pass.
- Final preflight checks the registration hash, authorized experiment and locked CFFG commits, tracked worktree state, runtime dataset and stream identities, heterogeneous training coverage, and unmodified baseline before claim reporting.
- Append-only final interruption/resume and independent family failure are covered by synthetic tests.
- Focused validation: 55 passed in 2.38 seconds.
- Final modern suite: 126 passed in 55.00 seconds.
- git diff --check passed; both implementation commits are clean.
- No final seed was generated or fit.

Next: cff-huxt must consume the completed cff-s6rs schema-v4 artifact, derive and hash the immutable schema-v5 registration at an authorized code commit, and still stop before every final seed.

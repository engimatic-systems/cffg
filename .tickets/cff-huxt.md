---
id: cff-huxt
status: closed
deps: [cff-s6rs, cff-9cx5, cff-7zq1]
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

This ticket validates the complete calibration artifact and writes one replacement preregistration and manifest for a lean, fixed demonstration. It freezes exactly three untouched final datasets for the classic family and three for the source-heterogeneous extension. The registered scope is a bounded demonstration over those six datasets, not a population-wide statistical claim.

The classic family uses the local generator that reproduces the pinned NASCAR
parameter construction and legacy random stream. It fits 10,000 training steps,
matching the paper's training scale. Its separate 2,500-step held-out extension
means it is not an exact replay of the Figure 3 workflow or an execution of the
unmodified authors' generator. The classic run owns the task-performance half
of the parity claim. cff-nkbj audits the factor semantics, parameter mappings,
and numerical oracles that own the probabilistic-architecture half. The
heterogeneous run is an independent extension result, not a prerequisite for
classic parity. This ticket does not fit or generate final data.

## Outcomes / Acceptance Criteria

1. The replacement preregistration fixes both family generators, versions, scales, splits, exactly three final seeds per family, family-qualified data identities, streams, arms, controls, budgets, diagnostics, performance tolerances, decision rules, failure rules, restart layout, and reporting surfaces.
2. Validation reconstructs the registered protocol from the exact cff-s6rs calibration artifact and rejects any changed, missing, incomplete, non-finite, or unmanifested input.
3. The freeze records and verifies CFFG, experiment, canonical generator, heterogeneous generator, upstream paper adapter, legacy environment, mapping, locks, calibration, evaluation-grid, and source commit hashes.
4. The cff-0l31 preregistration remains byte-for-byte available and is recorded as superseded without execution, with the coverage and report reasons stated.
5. Final commands name cff-oyie as the only authorized owner, use append-only family-qualified resume semantics, and permit no seed, threshold, budget, arm, decision rule, or claim changes after the new content hash.
6. README explains the local canonical generator's correspondence to the pinned
   NASCAR construction, the paper's 10,000-step training scale, the registered
   2,500-step held-out extension, why this is not an exact Figure 3 replay or
   authors-code execution, the two-part parity definition, the separate
   heterogeneous capability result, and exclusions in plain language.
7. The new preregistration and manifest validate from a clean checkout, and their content hash is recorded in the ticket and PR body.
8. Before the content hash is written, the family-aware final command validates the whole dual preregistration and is complete for per-final heterogeneous eligibility checks, the narrow unmodified baseline, family-separated claims and failures, append-only resume, retained posterior samples, raw aggregate recomputation, and explicit not-ready records.
9. Focused tests reject a tampered registration, changed calibration input, missing or non-finite payload, incomplete coordinate, mismatched family stream, and corrupted checkpoint. They also prove interrupted resume and one-family failure retention without generating a final seed.
10. No registered final seed is generated or fit by this ticket.
11. The frozen classic decision requires all three classic datasets to be ready, CFFG to meet the absolute NASCAR task-adequacy rules, and its registered outcomes to remain within calibrated practical tolerances of the independent paper-equation adapter. The freeze makes no population-level non-inferiority claim from three datasets.
12. The frozen heterogeneous decision requires all three heterogeneous datasets to be ready and to meet the registered source-dependent functional-recovery rules. Its pass or failure does not change the classic parity decision.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound method that satisfies the outcomes.

- Write to a new artifact directory and hash the old freeze rather than editing it.
- Make freeze output deterministic except for an explicitly recorded registration date.
- Verify each nested family/seed stream and data preimage independently.

## Future work

cff-oyie executes the frozen final protocol without changing code or scientific choices. cff-nkbj combines the classic task result with the already-audited architecture evidence and reports the heterogeneous extension separately. A failed final layer receives a new repair ticket rather than a changed freeze.

## Definitions

- **Immutable preregistration:** A content-hashed document whose scientific choices cannot change after final execution starts.
- **Fixed-dataset decision:** A pass only when each named final dataset meets
  each registered rule; it makes no claim about an unobserved population.
- **Probabilistic-architecture parity:** The CFFG factor types and their composition represent the same probability distributions and dependencies as the classic rSLDS formulation under the documented mapping.
- **Replacement freeze:** The dual-family preregistration that supersedes but does not delete cff-0l31.
- **Task parity:** On all three fixed classic final datasets, CFFG meets the registered NASCAR adequacy rules and performs within the frozen practical tolerances of the independent paper-equation adapter.

## Notes

Pre-implementation handoff: `docs/ARCHITECTURE.md` has no engine change-path row for this sibling validation harness. The freeze consumes the unchanged serialized `LearnableSpec` fitting interface. It updates the experiment README in this ticket; cff-nkbj updates the CFFG README, architecture boundary, and limitations with the final evidence.

Duplicated execution surfaces checked before the freeze:

- both family generators, descriptors, source-design evidence, evaluation grids, scales, splits, data identities, and final streams;
- both primary tiers, four Gibbs chains, two learned controls, the narrow unmodified baseline, family-specific absolute outcomes, and independent family decisions;
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
      +--> [x] cff-x5z2 recurrence statistics
                  |
                  +--> [x] cff-266l readiness semantics
                              +--> [x] cff-s6rs calibration --------+
                                                                    |
[x] cff-9cx5 v5 runner --> [x] cff-7zq1 freeze builder -------------+--> [x] cff-huxt freeze
                                                                               |
                                                                               +--> [ ] cff-oyie final run (ready)
                                                                                         |
                                                                                         +--> [ ] cff-nkbj
```

The immutable freeze is complete. `cff-oyie` is the only ready owner permitted
to generate a registered final seed.

**2026-08-18T17:04:27Z**

Freeze audit — Question → How can the lean fixed-dataset decision change without invalidating the calibration now running at experiment commit 248f9d3? Answer → Preserve schema v4 as the immutable calibration contract and add a separately versioned schema-v5 final protocol. The v4 validator and interval-shaped not-ready aggregate remain only to recompute the calibration artifact byte-for-byte. Freeze validates that complete v4 artifact, copies its generator, scale, thresholds, budgets, and 3+3 seed map without tuning, then replaces the final decision policy with an explicit fixed-dataset rule. For each family, every registered effect must have exactly three finite values in final-seed order, and every individual value must meet its direction and threshold; no confidence interval, alpha level, population inference, family unlock, or joint decision enters the final pass. `claim_evaluation` dispatches by schema so v4 calibration recomputation cannot drift and v5 final execution cannot silently fall back to Bonferroni intervals. The rewritten dual freeze also records the old one-family freeze hash, both family identities and streams, all calibration and source-design manifests, mapping version, exact authorized experiment and CFFG commits, and a final-command preflight that rejects code changed after the registration hash. Source → read-only audit of `freeze.py`, `dual_protocol.py`, `dual_calibration.py`, `claim_evaluation.py`, `dual_execution.py`, and checkpoint identity at experiment commit 61cec3f. Confidence → high for the version-boundary and decision semantics; exact content hashes remain pending the completed cff-s6rs artifact.

**2026-08-18T17:05:50Z**

Final-run audit — Question → Is the family-aware command already complete enough to freeze? Answer → No; cff-huxt owns four concrete pre-freeze gaps. First, `run_family_coordinate` generates a heterogeneous final dataset and immediately fits it; it does not run or record the frozen exposure, success, failure, and augmented-rank gates on that final training prefix. An ineligible dataset must become an explicit not-ready coordinate before fitting. Second, the dual report does not include or verify `secondary.unmodified_baseline`, although the old one-family report does. Third, final execution hashes the registration into checkpoints but does not reject an experiment or CFFG commit that differs from the commits authorized by the registration. Fourth, each coordinate computes a data identity from the runtime Git commit, but no validator compares it with a frozen per-family/per-seed identity. The v5 protocol and runner must close all four gaps before the registration content hash is written. Source → `dual_execution.py` family runner, final preflight, and report assembly; `confirmatory.py` unmodified baseline; `source_design.py` gate coverage; `execution_checkpoint.py` code-qualified identities. Confidence → high; no final seed was inspected or generated.

**2026-08-18T17:06:15Z**

Superseded-freeze identity captured before replacement work: `artifacts/rslds-preregistration/preregistration.json` SHA-256 `308772928c21e7958cd1b7e47d5989d7bc434e93aa55596a246d5bcf989511f7`; canonical JSON content SHA-256 `0ca693cc7eb7821da161a763e0262e1495ff19d6fe92cf6445b7f01a3cf97e4c`; manifest SHA-256 `f7ca80e1dc22637097661aa10368cf8ba5a0e4345137617d4f13137e02c18e50`. Its phase boundary still says `final_seed_executed: false`. The replacement must record these bytes as superseded and must not rewrite them.

**2026-08-19T15:30:55Z**

Measured replacement freeze completed on experiment code commit c3be541 and committed as experiment artifact commit 99d81b5. The schema-v5 registration SHA-256 is 24a72073e98ed14b70d403211c9ac346c6da7c1c9e7e94142944a83ca11be477; its canonical content SHA-256 is e41fa0796995a6c4558c49eaf5207ee6e4862acde71a4d8250fe958a105ec690; the manifest SHA-256 is 487bb0831295293fe2c8bd39f92f44dedf8accdb81fe86afaafef0099386fafa; and final-command.txt SHA-256 is a529e1b436d67f57574b06a08ddfa3db0b9bb8c3f4d70d9b1db01cfc6abe023. Artifact validation reconstructs the registration, manifest, command hash, complete calibration/scale/source-design provenance, both lockfiles, mapping and adapter identities, the superseded unexecuted freeze, all 3+3 final identities and streams, and the independent fixed-dataset decisions. The freeze records experiment commit c3be541 and CFFG commit a686faa as the only authorized execution revisions. No final seed directory exists and phase_boundary.final_seed_executed is false. cff-oyie must run the exact manifested command from the authorized experiment code commit with these byte-identical registration files present.

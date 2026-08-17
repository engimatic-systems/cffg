---
id: cff-v8a1
status: closed
deps: [cff-64ri]
links: []
created: 2026-08-17T17:14:46Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, reference, correctness]
---
# Align reference dynamics with the published destination regime

## Background in Plain English

The cff-64ri recurrence adapter inherits legacy SLDS state code that selects
the linear dynamics for `x_(t+1)` with the source regime `z_t`. Linderman et
al. Eq. (2), the canonical NASCAR generator, and CFFG instead select those
dynamics with the destination regime `z_(t+1)`. An asymmetric model therefore
gives different likelihoods and samples even when every parameter array is
mapped exactly.

The primary same-model mapping in cff-0l31 cannot be frozen until the
independent reference adapter follows the published convention on every
dynamics surface. The pinned unmodified learner remains unchanged as a
secondary historical comparator.

## Outcomes / Acceptance Criteria

1. Adapter trajectory generation selects the destination row for each
   `x_t -> x_(t+1)` conditional.
2. Discrete-state resampling assigns the transition likelihood to `z_(t+1)`,
   not `z_t`.
3. Continuous-state information parameters select the destination dynamics
   row for each adjacent pair.
4. Dynamics-parameter resampling groups sufficient statistics by
   `z_(t+1)`.
5. Direct NumPy oracles cover all four surfaces without importing the adapter
   or the legacy state-update helper under test.
6. An asymmetric fixture makes source and destination choices numerically
   different and checks the published destination result.
7. The compact observed-only adapter smoke fit remains finite and records all
   chain statuses.
8. Source-specific recurrence behavior and the pinned unmodified secondary
   comparator remain unchanged.
9. The provenance ledger and smoke artifact record the correction, Eq. (2),
   the adapter commit, test output, and content hashes.
10. Sparse categorical support is not changed by this indexing correction.
    Non-finite dynamics inputs or outputs fail loudly. The inherited reference
    remains float64-only, and the existing `K=32` recurrence case remains the
    uncomfortable-cardinality check; cff-oyie owns `T=10,000` execution.

## Methodological Suggestions

- Override the adapter state and model classes instead of patching the pinned
  upstream checkout.
- Keep the four destination-index surfaces adjacent or named from one helper
  so a future convention change cannot update only part of the Gibbs cycle.
- Use different matrices, offsets, and covariances for every regime in the
  regression fixture; tied parameters cannot expose this defect.

## Future Work

cff-0l31 owns the complete cross-implementation mapping and preregistration.
cff-oyie owns final-seed execution. Broad correction of the pinned upstream
project is outside this chain.

## Definitions

- **Destination regime:** `z_(t+1)`, the regime selecting the conditional
  distribution of `x_(t+1)` in main-paper Eq. (2).
- **Dynamics surface:** One of generation, discrete-state likelihood,
  continuous-state information parameters, or parameter resampling.
- **Source regime:** `z_t`, which still selects the Eq. (4) recurrence logits.

## Notes

Architecture change-path index: N/A because this ticket changes only the
external reference adapter. Duplicated external execution surfaces audited:
generation, discrete likelihood, Gaussian information form, and conjugate
parameter grouping. CFFG compiler, VM, schedule, learning, serialization, and
documentation surfaces are N/A.

**2026-08-17T17:22:55Z**

Research question: which regime owns the affine Gaussian conditional for x_(t+1), and which adapter paths must share that convention? Answer: Linderman et al. main Eq. (2) selects A[z_(t+1)], B[z_(t+1)], and Q[z_(t+1)]; generation, discrete likelihood, Gaussian information parameters, and conjugate dynamics resampling must therefore all use the destination row. The audit also found that the inherited discrete-likelihood cache overwrote the initial-state contribution, so the adapter now restores that term explicitly. Source: https://proceedings.mlr.press/v54/linderman17a/linderman17a.pdf#page=2. Confidence: high; NumPy-only asymmetric oracles separately verify all four surfaces and distinguish them from source indexing. Implementation completed in sibling repo /Users/ravd/projects/experiments/cffg-nascar-validation on branch cff-v8a1-destination-dynamics: semantic commit 6df6305, proposed head with hashed evidence ab5c27b. Validation: 30 tests passed; observed-only adapter smoke completed; artifact records Eq. (2), implementation commit 6df6305, full-suite JUnit, and five verified payload hashes. The pinned unmodified comparator was not changed. Next: cff-0l31 may freeze the same-model mapping against this corrected adapter.

**2026-08-17T17:23:12Z**

Wiggle-audit review: canonical means the repository's established NASCAR generator; N/A means structurally irrelevant to the external-only change; same-model means identical published equations and mapped parameters; expose means reveal the indexing defect; implement refers to adapter code or the named future comparison harness; support means probability support; complete means the enumerated mapping; remaining/remain means preserved state or explicitly deferred work; closed means ticket acceptance completion; sufficient statistics is the conjugate-regression mathematical term; check/test/validate/verify/oracle/evidence/pass name the listed automated or direct-NumPy authorities; correct means defect repaired; exact means literal equation/index/hash identity; fit/learn mean the observed-only Gibbs parameter workflow; may is permission for the now-ready handoff; explicit owners and named handoffs are cff-v8a1 for the four corrected external surfaces, cff-0l31 for mapping/preregistration, and cff-oyie for final execution; all/every/only are bounded to the four named dynamics surfaces, four regimes, recorded statuses, float64 boundary, or external adapter; existing means the K=32 test at dependency head. Capability ownership is therefore explicit and no CFFG production surface is claimed.

---
id: cff-l5ag
status: open
deps: [cff-8w4c, cff-upih, cff-lo4d]
links: [cff-oyie]
created: 2026-08-20T15:41:17Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, statistics]
---
# Restore eligible mixing in the rSLDS paper-equation adapter

## Background in Plain English

The frozen cff-oyie run completed every paper-adapter chain that reached
fitting, but none of its ten fitted adapter arms passed the registered mixing
check. Their worst split R-hat values ranged from 1.666 to 9.623, above the
1.05 ceiling. Their smallest effective sample sizes ranged from 4.055 to
7.110, below the required 50. The runner therefore withheld every paired
CFFG-versus-adapter result.

This is a reference-inference failure, not evidence that the two probability
models differ. Four completed chains and their retained draws are preserved in
the final artifact. They can show whether the problem comes from regime-label
symmetry, a small set of slow parameters, or the Gibbs updates themselves.

## Outcomes / Acceptance Criteria

1. A deterministic diagnostic names every parameter family responsible for the
   failed R-hat or effective-sample-size check and reports its per-chain trace.
2. The adapter resolves equivalent regime and decision-list representations
   before mixing diagnostics are pooled, while preserving genuinely different
   posterior modes.
3. Four independent chains meet split R-hat at most 1.05 and effective sample
   size at least 50 on both registered calibration families and scales without
   weakening either threshold.
4. The paper-equation density, transition, dynamics, emission, and destination-
   regime oracles remain unchanged and passing.
5. The isolated Python 3.11 environment, pinned source identities, adapter
   provenance, raw draws, and diagnostics remain independently reproducible.
6. Any new confirmatory comparison uses a newly calibrated and frozen protocol
   with fresh final seeds. It does not relabel the cff-oyie result.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Start with the retained cff-oyie chains and rank the 43 monitored variables
  by R-hat and effective sample size.
- Check label and decision-list canonicalization before increasing chain length
  or changing the sampler.
- Use calibration or new development seeds for changes. Keep final seeds 2001
  through 2003 and 2101 through 2103 out of tuning decisions.

## Future work

A later ticket may freeze and run a fresh comparison after this ticket and the
other cff-oyie repair tickets close. This ticket does not rerun final data or
change CFFG learning.

## Definitions

- **Effective sample size:** An estimate of how many independent posterior
  samples a correlated Markov chain represents.
- **R-hat:** A diagnostic that compares variation within and between independent
  chains; values near one indicate agreement.
- **Paper-equation adapter:** The independent implementation of the published
  rSLDS equations used as the same-model comparison arm.

## Notes

Architecture handoff: the applicable boundary is the sibling experiment's
isolated reference adapter. CFFG compiler, VM, schedule, kernels, ledger,
serialization, learnable declarations, and factor equations are N/A.

Duplicated surfaces: both dataset families, both initialization tiers, all four
chains, canonicalization, diagnostic aggregation, retained draws, adapter
provenance, and the nested runtime must stay aligned.

Hostile-input decisions: exact-zero transition support, nearly deterministic
gates, label-switched chains, non-finite draws, and one stuck chain must fail
loudly. The adapter is a locked float64 runtime, so float32 is N/A. The classic
12,500-step dataset supplies the uncomfortable-scale case.

Legend: prerequisite --> dependent

```text
[x] cff-oyie frozen evidence
      |
      +--> [ ] cff-l5ag adapter mixing repair (ready)
                  |
                  +--> [ ] fresh comparison freeze and run
```

**2026-08-21T01:23:37Z**

Independent-review synthesis: make raw recurrence coefficients secondary and require mixing of label/decision-list-invariant quantities that actually support the parity claim. A moderate shared recurrence prior is the leading development pilot, not a registered fix; it must apply identically to CFFG and the adapter, preserve classic NASCAR adequacy, and still needs explicit heterogeneous mode handling. Deterministic label initialization is hygiene, not a substitute for sampling genuine modes. Unified plan: notes/rslds-debug/synthesis/plan.md.

**2026-08-21T03:21:43Z**

Roadmap refinement split. Legend: prerequisite --> dependent

[ ] cff-jq9p claim-bearing diagnostics (ready) --+
                                                  +--> [ ] cff-8w4c shared-prior study
[ ] cff-888k component-collapse evidence (ready) -+                 |
                                                                    +--> [ ] cff-l5ag adapter decision
                                                                                 |
                                                                                 +--> [ ] cff-nkbj parity epic

The diagnostic and collapse tickets can proceed in parallel. Their shared study is the immediate blocker for the adapter decision. Ticket dependency records remain the source of truth.

**2026-08-21T14:37:31Z**

Classic-scale cff-2wb6 pilot stopped after the declared seed-5000 same-K and
shared-initialization arms. Experiment artifact commit 40f992e (runner code
8307549; CFFG 7405d60) validates as `pilot_complete_interpretable`. Same-K
converged in 2 outer maps with maximum completed inner attempt 183. Shared
rSLDS reached the unchanged outer certificate in 13 maps; its runtime residual
tail was 0.228316, 0.141905, 0.097744, 0.072032. The first terminal-inference
attempt exhausted the 8,000-inner ceiling at damping 0.9, then a damping-0.8
retry converged in 4,488 iterations and returned coherent state. Under the
declared attempt-ownership rule, shared inner work is right-censored, so no
classic family budget was derived and no expansion or adapter ran. Non-finite
telemetry paths were empty. Shared used 7,324.17 seconds and peaked at
5,779,226,624 bytes RSS. Its periodicity fit checkpoint is retained. This keeps
cff-l5ag blocked on cff-2wb6. That ticket must acquire uncensored classic inner
convergence measurements under a preregistered diagnostic bound before
cff-8w4c resumes. This result does not reopen cff-qulu.

**2026-08-21T14:54:11Z**

Correction to the 2026-08-21T14:37:31Z pilot note: cff-2wb6 schema v1
classified individual damping attempts instead of each complete registered
retry sequence. Under the corrected schema v2 classifier, the shared rSLDS
terminal hidden-state solve succeeded because its damping-0.8 retry converged
in 4,488 iterations after the damping-0.9 attempt exhausted 8,000. The shared
arm is therefore inner-converged, not right-censored; all 14 hidden-state solves
succeeded, the maximum successful attempt was 6,315 iterations, and the
per-attempt cap remains 8,000 because one rejected retry consumed that cap.
Outer convergence remains 13 maps. The immutable original artifact still
validates. Experiment commits c55ba69 and c937c2b add the classifier and a
pinned no-fit reclassification artifact with zero inner or outer censor rows.
No family budget is derived from the two-arm pilot, and no expansion, adapter,
data generation, or fit ran. The pilot now permits the separately authorized
cff-2wb6 expansion; cff-l5ag remains downstream of that declared cohort rather
than blocked on an inner-classifier decision.

**2026-08-21T18:58:37Z**

Classic-scale convergence blocker resolved by the completed cff-2wb6 schema-v2 cohort at experiment commit 6e3edc3. All 12 declared development arms converged with zero inner/outer censor rows and zero non-finite paths. Maximum completed outer convergence time was 13 maps; maximum successful inner attempt was 6315 iterations. The registered rule selects outer budget 17 and retains inner cap 8000 because seed-5000 shared had one rejected capped attempt followed by a successful retry. The full artifact validates and retains six CFFG periodicity checkpoints. This clears cff-8w4c to conduct its staged classic prior screen under family-valid budgets. It does not run an adapter, repair mixing, or change the cff-l5ag claim; cff-l5ag remains downstream of cff-8w4c.

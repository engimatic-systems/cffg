---
id: cff-l5ag
status: open
deps: []
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

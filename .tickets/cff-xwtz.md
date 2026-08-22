---
id: cff-xwtz
status: in_progress
deps: []
links: []
created: 2026-08-22T19:11:04Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-0oqo
tags: [rslds, inference, diagnostics]
---
# Locate the remaining D9 categorical responsibility bias

## Background in Plain English — WHY

The D9 structured-pair engine now uses one globally consistent consecutive-
regime belief for Gaussian messages and recurrence learning. Its first
source-heterogeneous development fit converged, and the retained factor pairs
reproduced every pooled recurrence statistic to numerical precision. The fit
still missed the unchanged 0.150 recovery ceilings.

A post-fit clamping ladder located the remaining error in categorical regime-
transition responsibilities. Replacing only the inferred regime-transition
pairs with the generating pairs reduced worst-source error from 0.204 to 0.033
and worst source-contrast error from 0.313 to 0.038. Replacing inferred
continuous moments with generating states did not help while the inferred
pairs remained fixed. The next diagnostic must determine whether the inferred
pairs are wrong because CFFG combines categorical-chain evidence incorrectly
or because the learned continuous/dynamics model supplies biased evidence to
an otherwise correct chain smoother.

## Outcomes / Acceptance Criteria — WHAT

1. An independent exact categorical-chain smoother reproduces the retained D9
   singleton and consecutive-state beliefs from the same learned parameters,
   continuous moments, and local evidence at a named dtype-scaled tolerance,
   or preserves a localized mismatch that names its factor and time step.
2. A development-only continuous-state clamping ladder reruns categorical
   smoothing rather than merely swapping states into the recurrence M-step.
   It compares inferred continuous beliefs, generating continuous states, and
   a full generating authority without exposing generating values to fitting.
3. Each ladder stage reports regime accuracy, source/destination pair-count
   error, source/gate exposure and signed-success error, and the unchanged
   worst-source and source-contrast recovery scores after a prior-rooted
   recurrence update.
4. The result chooses one falsifiable owner: categorical message combination,
   learned continuous/dynamics evidence, or an internally correct but biased
   variational approximation. It states the observation that would falsify
   that choice.
5. Exact-zero support remains exact. Improper or non-finite evidence fails
   loudly. Float32 and a K=8, T>=100 case remain finite and avoid exhaustive
   path enumeration.
6. The committed cff-h0hx checkpoint, observations, semantic identity, pair
   orientation, stream identities, and truth boundary authenticate before the
   diagnostic regenerates development truth.
7. No fit, adapter, SLDS control, calibration seed, or final seed runs. The D9
   production behavior remains unchanged until the diagnostic identifies a
   production owner.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method.

- Start from the cff-h0hx seed-5000 native checkpoint and its retained D9 pair
  beliefs. Do not rerun the 12-minute fit.
- Reuse the cff-nbo9 exact-chain oracle, but supply the learned large-coordinate
  potentials and all non-recurrence categorical evidence independently.
- Recompute categorical pairs after clamping continuous states. The existing
  cff-rk7i ladder held the inferred pair fixed and therefore cannot answer this
  question.
- Keep D9 in place during diagnosis. Its pair algebra, marginal consistency,
  pooled learning readout, and small exact-chain oracles pass; reverting would
  restore the known singleton-product defect without explaining this miss.

## Future work

If the independent smoother and CFFG disagree, create a bounded production
repair at the named message or evidence-combination boundary. If they agree,
create a bounded model/inference ticket for the continuous/dynamics evidence
or variational approximation identified by the clamping ladder. A successful
repair unblocks a fresh cff-h0hx development artifact; it does not authorize
calibration or final seeds.

## Definitions

- **Categorical responsibility:** Posterior probability assigned to a regime
  or to one consecutive-regime transition.
- **Continuous-state clamping:** Replacing inferred continuous-state beliefs
  with known development generating states only inside a post-fit diagnostic.
- **D9 structured pair:** The source-row, destination-column consecutive-state
  belief used by the current recurrence rule, Gaussian message, and learner.
- **Local evidence:** The expected-log contribution that a factor supplies to
  one regime variable or consecutive-regime pair.

## Notes

Architecture handoff: inspect recurrence structured smoothing, categorical
messages, Gaussian/dynamics mixture evidence, factor-local pair readout, and
the diagnostic consumer. Learning replacement, persistence, adapters,
controls, calibration, and final execution are N/A unless the diagnostic
localizes the defect there.

Hostile-input decisions are acceptance criterion 5. The independent smoother
must preserve literal support and scale linearly in T up to the K-squared pair
cost.

Legend: prerequisite --> dependent

```text
[x] D9 structured-pair engine + pilot falsifier
      |
      +--> [ ] cff-xwtz categorical responsibility diagnosis (ready)
                  |
                  +--> [ ] cff-h0hx fresh development requalification
                              |
                              +--> [ ] cff-0oqo bounded recovery
```

**2026-08-22T19:13:52Z**

Implementation start from authenticated cff-h0hx seed-5000/native D9 checkpoint. The diagnostic reuses the converged fit and must not rerun fitting. It independently reconstructs categorical-chain evidence and reruns categorical smoothing under inferred versus generating continuous-state diagnostic inputs, while keeping learned parameters fixed. The decision must name the first owner that disagrees and preserve the no-fit truth boundary.

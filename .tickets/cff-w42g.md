---
id: cff-w42g
status: in_progress
deps: []
links: []
created: 2026-08-22T14:26:39Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-l5ag
tags: [rslds, validation, statistics]
---
# Qualify adapter mixing diagnostics by dataset family

## Background in Plain English — WHY

The completed prior screen used six adapter mixing variables for both NASCAR
families. That is too broad for the classic family. Its trajectories identify
transition behavior on each source regime's occupied part of the track, but
they do not identify every source at every counterfactual grid state.

At recurrence strengths 0.3 and 1.0, the classic adapter chains mixed on
likelihood, aligned latent recovery, aligned regime accuracy, and held-out
on-support transition recovery. Only the two balanced-grid variables failed.
The largest chain disagreements occurred at grid states with no nearby
same-source training observation. Those grid variables are required for the
source-heterogeneous extension, but they do not support the bounded classic
parity claim.

The diagnostic schema must name its authority by dataset family. This is a
prospective protocol correction. It must not relabel the completed screen or
the cff-oyie final artifact as passing.

## Outcomes / Acceptance Criteria — WHAT

1. A versioned schema declares exactly four classic eligibility variables:
   likelihood per step, affine-aligned latent error, label-aligned regime
   accuracy, and held-out on-support transition-probability error.
2. The same schema declares all six variables for heterogeneous eligibility,
   including balanced-grid worst-source probability and source-contrast error.
3. Classic balanced-grid diagnostics and all raw-parameter diagnostics remain
   retained as secondary evidence and cannot change classic eligibility.
4. Split R-hat at most 1.05 and effective sample size at least 50 remain the
   eligibility thresholds for every family-authoritative variable.
5. Historical cff-oyie and shared-prior artifacts reproduce their original
   schemas and decisions byte-for-byte; the new authority applies only to a
   newly registered development or final protocol.
6. Compact tests prove that an off-support classic grid disagreement is
   descriptive, an on-support classic disagreement is disqualifying, and
   either heterogeneous grid disagreement is disqualifying.
7. Missing, non-finite, one-chain, and exactly stuck inputs fail loudly before
   eligibility is reported.
8. The schema, family map, thresholds, estimand definitions, implementation
   digest, and recomputation path are pinned in checkpoints and manifests.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Put the family-to-authoritative-variable map in one schema object used by
  execution, checkpoint validation, aggregation, and freeze validation.
- Recompute the completed screen in a development-only comparison report that
  displays both old and proposed decisions without rewriting either artifact.
- Keep the balanced common-state grid unchanged so heterogeneous evidence and
  descriptive classic evidence remain comparable.

## Future work

`cff-upih` uses this authority in its heterogeneous native-basin study.
`cff-l5ag` remains responsible for demonstrating eligible adapter inference and
for requiring a fresh calibration and freeze. This ticket does not change the
sampler, priors, CFFG learning, or final evidence.

## Definitions

- **Balanced grid:** A fixed set of latent states at which every source row is
  evaluated, including source/state pairs not observed on one trajectory.
- **Eligibility authority:** The named diagnostics whose pass or failure
  controls whether a comparison arm may support a claim.
- **On-support:** Evaluated where the fitted source regime has training data,
  rather than at a counterfactual source/state pair.

## Notes

Architecture handoff: use `docs/ARCHITECTURE.md`'s canonical rSLDS consumer
validation boundary and the sibling experiment's protocol, checkpoint,
aggregation, and freeze boundaries. CFFG factor equations, IR, compiler, VM,
schedules, kernels, ledger, and serialization are N/A.

Duplicated surfaces: classic and heterogeneous family roles, both initialization
tiers, all chains, execution records, checkpoints, aggregate recomputation,
registration, freeze, and historical schema compatibility must stay aligned.

Hostile-input decisions: exact-zero probabilities remain exact; non-finite,
missing, single-chain, and stuck-chain diagnostics fail loudly. The isolated
adapter is float64, so float32 is N/A. The 12,500-step classic coordinate is
the uncomfortable-scale case.

Legend: prerequisite --> dependent

```text
[ ] cff-w42g family-qualified authority (ready)
      |
      +--> [ ] cff-upih native-basin study
                  |
                  +--> [ ] cff-l5ag adapter repair
                              |
                              +--> [ ] cff-nkbj parity epic
```

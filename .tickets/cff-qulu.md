---
id: cff-qulu
status: in_progress
deps: [cff-0xjq, cff-tlh3]
links: [cff-oyie]
created: 2026-08-20T15:41:17Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, learning, numerics]
---
# Certify learned switching-model outer convergence at heterogeneous scale

## Background in Plain English — WHY

The frozen heterogeneous run stopped several learned switching models at its
outer-iteration limit. Every inner inference solve converged, so the result
could not show whether the outer map was unstable or merely slow.

A bounded development replay now separates those cases. On calibration seed
3101, replicate 1, the shared-initialization rSLDS and same-regime-count SLDS
control both converged at outer iteration 15. The selected calibration cap had
been 12. Installed-parameter and transition-function update vectors contracted
without a two-cycle signal. This evidence does not justify a globalization
algorithm. It does justify a clean installed-parameter certificate and a
budget policy that treats exhausted runs as censored rather than as measured
convergence times.

## Outcomes / Acceptance Criteria — WHAT

1. The canonical runtime-map maximum becomes the rSLDS outer convergence
   authority; the legacy mixed snapshot and truth-free functional transition
   residual remain named diagnostics.
2. Learned SLDS controls use the same installed-parameter residual semantics
   and coherent terminal-state contract.
3. Converged, budget-exhausted, and diverged remain distinct terminal statuses,
   and a non-finite residual or free energy is always divergence.
4. The outer budget rule rejects exhausted coordinates as convergence-time
   measurements and derives headroom only from completed development fits.
5. Both rSLDS initialization tiers and both learned SLDS controls converge on a
   declared set of heterogeneous development datasets at the selected
   5,120-step training scale without weakening the 0.1 threshold or the
   two-consecutive-update rule.
6. Exact-zero support, improper incoming beliefs, float32, and a training chain
   at least ten times the compact fixture remain covered by executable tests.
7. The cff-oyie budget-exhausted records remain unchanged; any confirmatory
   rerun uses a new calibration and freeze.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Reuse `cffg.models.coordinate_fit` for both model families and let each model
  define its own named installed-parameter blocks.
- Count complete outer-map evaluations. With the current full replacement map,
  this equals accepted outer updates.
- Funnel validation through compact tests, one 5,120-step development arm, then
  the remaining declared development coordinates.
- Create a separate globalization ticket only if new development evidence shows
  opposing update vectors, persistent noncontraction, or no finite headroom.

## Future work

`cff-0oqo` owns source-function recovery once the learner can reach a certified
terminal state. A later ticket may freeze fresh final evidence. This ticket
does not change the recurrence probability model.

## Definitions

- **Canonical runtime map:** The complete undamped replacement expressed in the
  normalized parameters and potentials installed into the next inner solve.
- **Censored exhaustion:** A fit known only not to have converged before its
  limit; its true convergence time is unknown.
- **Outer loop:** The alternating cycle that infers hidden beliefs, replaces
  learned parameters, and checks whole-model convergence.
- **Residual:** The scale-normalized installed-parameter change used to certify outer
  convergence.

## Notes

Architecture handoff: use the `docs/ARCHITECTURE.md` observed-only learning
change path and inspect declared learnables, engine inference, replacement,
and final readout. Compiler, VM, schedule, kernels, ledger, and serialization
are N/A unless the chosen fix changes their contract.

Duplicated surfaces: rSLDS shared and native initialization, both learned SLDS
controls, outer status propagation, free-energy and residual traces, secondary
reporting, and experiment eligibility must stay aligned.

Hostile-input decisions are acceptance criterion 6. Sparse or improper beliefs
must fail loudly, dtype guards must scale with machine precision, and the
paper-scale classic path remains a separate uncomfortable-scale regression.

Legend: prerequisite --> dependent

```text
[x] cff-0xjq certificate decision
      |
      +--> [ ] cff-qulu runtime certificate + budget policy (ready)
                  |
                  +--> [ ] cff-0oqo source recovery
                              |
                              +--> [ ] fresh comparison run
```

**2026-08-21T01:23:37Z**

Independent-review synthesis: instrument and justify the convergence certificate before using it as a globalization merit. The initial-regime Dirichlet can dominate the raw maximum despite other families being stable, but it must not simply be excluded without showing belief and claim-bearing functional stability. Repair the SLDS stale-terminal-state contract, observe full-step tails on new development seeds, then implement a shared safeguarded natural/statistic-coordinate outer coordinator. Preserve 1,280 as a deterministic failure regression; validate the future protocol at the chosen heterogeneous scale. Unified plan: notes/rslds-debug/synthesis/plan.md.

**2026-08-21T03:56:00Z**

The cff-0xjq replay supersedes the proposed globalization step. All three rSLDS
residual views and the matched-control residual reached the unchanged two-hit
gate at iteration 15, three updates after the old calibration cap. Direction
evidence showed contraction rather than a cycle. This ticket now owns the
installed-parameter certificate, censored-budget policy, and replication at the
selected scale. Globalization is conditional future work, not an acceptance
requirement.

**2026-08-21T04:33:51Z**

Selected-scale validation split. Legend: prerequisite --> dependent

[x] cff-0xjq 1,280-step certificate decision
      |
      +--> [x] runtime certificate + censored-budget code
                  |
                  +--> [ ] cff-tlh3 5,120-step development cohort (ready/in progress)
                              |
                              +--> [ ] cff-qulu budget decision
                                          |
                                          +--> [ ] cff-0oqo source recovery

The focused cohort replaces an obsolete three-horizon/reference-chain rerun. Dependency records remain the source of truth.

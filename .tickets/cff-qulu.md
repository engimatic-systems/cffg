---
id: cff-qulu
status: open
deps: [cff-0xjq]
links: [cff-oyie]
created: 2026-08-20T15:41:17Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, learning, numerics]
---
# Globalize learned switching-model outer convergence at heterogeneous scale

## Background in Plain English

The final heterogeneous run exposed scale-sensitive outer-loop convergence.
On seed 2103, both CFFG rSLDS tiers reached the frozen limit of sixteen outer
updates with residuals 0.225 and 0.484, above the 0.1 convergence threshold.
The two learned SLDS controls also exhausted the same limit on seeds 2101 and
2103. Their inner inference solves converged, so the failure is in the
coordinate-replacement loop rather than the message solver.

Simply treating budget exhaustion as success would hide the defect. The outer
learner needs a stable continuation or globalization rule that can make
progress at heterogeneous scale and still reports an honest failure when it
cannot converge.

## Outcomes / Acceptance Criteria

1. A deterministic development regression reproduces oscillating or stalled
   outer residuals for both the rSLDS primary model and learned SLDS controls.
2. The outer learner uses a scale-safe globalization rule instead of relying on
   a fixed damping value tuned to a compact fixture.
3. Converged, budget-exhausted, and diverged remain distinct terminal statuses,
   and a non-finite residual or free energy is always divergence.
4. Both initialization tiers converge on declared heterogeneous development
   seeds at the registered 1,280-step training scale without weakening the
   residual threshold or consecutive-iteration rule.
5. Same-regime-count and closest-free-scalar-budget SLDS controls receive the
   same convergence contract as the rSLDS path.
6. Exact-zero support, improper incoming beliefs, float32, and a training chain
   at least ten times the compact fixture remain covered by executable tests.
7. The cff-oyie budget-exhausted records remain unchanged; any confirmatory
   rerun uses a new calibration and freeze.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Diagnose whether free energy, message residual, and parameter displacement
  agree before choosing backtracking, damping adaptation, or trust-region
  continuation.
- Keep inner inference and outer replacement traces separate in diagnostics.
- Use calibration or new development seeds, not cff-oyie's final seeds, to tune
  the rule.

## Future work

cff-0oqo owns source-function recovery once the learner can reach a certified
terminal state. A later ticket may freeze fresh final evidence. This ticket
does not change the recurrence probability model.

## Definitions

- **Globalization:** A rule such as backtracking or adaptive damping that keeps
  a local update method stable when a full step is unsafe.
- **Outer loop:** The alternating cycle that infers hidden beliefs, replaces
  learned parameters, and checks whole-model convergence.
- **Residual:** The registered scale-normalized change used to certify outer
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
[x] cff-oyie convergence evidence
      |
      +--> [ ] cff-qulu outer globalization (ready)
                  |
                  +--> [ ] cff-0oqo source recovery
                              |
                              +--> [ ] fresh comparison run
```

**2026-08-21T01:23:37Z**

Independent-review synthesis: instrument and justify the convergence certificate before using it as a globalization merit. The initial-regime Dirichlet can dominate the raw maximum despite other families being stable, but it must not simply be excluded without showing belief and claim-bearing functional stability. Repair the SLDS stale-terminal-state contract, observe full-step tails on new development seeds, then implement a shared safeguarded natural/statistic-coordinate outer coordinator. Preserve 1,280 as a deterministic failure regression; validate the future protocol at the chosen heterogeneous scale. Unified plan: notes/rslds-debug/synthesis/plan.md.

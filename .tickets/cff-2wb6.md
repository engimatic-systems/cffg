---
id: cff-2wb6
status: in_progress
deps: []
links: []
created: 2026-08-21T12:11:58Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-8w4c
tags: [rslds, validation, numerics]
---
# Measure classic-scale convergence under the repaired certificate

# Background in Plain English — WHY

The staged shared-prior funnel reached the canonical 10,000-step training scale before any paper-adapter chain ran. Both CFFG initialization tiers exhausted the heterogeneous-derived 19-map outer budget. Both matched SLDS controls exhausted the 4,000-iteration inner budget on their first outer evaluation. The historical classic final seeds converged under the former certificate, but that does not measure convergence time under the repaired installed-parameter certificate on new development data.

The funnel also exposed an evidence-retention defect: a non-finite value in fit-summary telemetry prevented periodicity checkpoint serialization. Fitted point parameters had already passed finite-parameter validation. The exact telemetry field was not retained by the old callback path, so a focused no-adapter runner must preserve it before selecting any larger budget.

# Outcomes / Acceptance Criteria — WHAT

1. A development-only classic runner executes CFFG shared/native tiers and same-K/K=5 SLDS controls without paper-adapter chains, calibration seeds, or final seeds.
2. Every tier and control durably records inner and outer status, canonical/legacy residual histories, per-phase work, non-finite telemetry paths, terminal coherence, wall time, and peak RSS before aggregate validation.
3. The runner uses declared classic development seeds and the exact 12,500-total/10,000-training generator protocol, with append-only per-arm checkpoints and explicit resume/tamper validation.
4. Exhausted inner or outer runs are right-censored and cannot select a budget. If every declared arm converges, family-specific inner and outer budgets are derived from completed measurements with registered headroom.
5. The selected budget leaves the 0.1 outer threshold and two-consecutive-map rule unchanged and does not reinterpret the completed cff-oyie artifact.
6. The artifact provides cff-3v7q-compatible fit checkpoints for converged classic CFFG arms so periodicity work reuses these fits.
7. Compact tests cover finite, non-finite, exhausted, divergent, float32, exact-zero support, interrupted resume, and historical-artifact validation before any long run.

# Methodological Suggestions — HOW

- Adapt the selected-scale outer runner to a family-specific classic plan rather than sending controls and adapters through the full prior-study coordinate.
- Funnel one development seed through same-K SLDS and shared CFFG first. Expand only after both return durable interpretable statuses.
- Record exact non-finite JSON paths using the existing coordinate sanitizer vocabulary; do not replace scientific infinities with unlabeled nulls.
- Separate inner and outer right-censoring. A larger outer cap cannot repair a first inner solve that never converges.

# Future work

After this ticket, cff-8w4c can screen recurrence priors with a family-valid classic budget, and cff-3v7q can replay retained fits across declared rollout streams. Algorithmic globalization or inner-solver changes require a new defect ticket only if the measured development traces justify them.

# Definitions

- **Family-specific budget:** An inference limit measured for one declared dataset family and scale rather than borrowed from another family.
- **Right-censored:** The run only establishes that convergence took longer than its limit; its convergence time is unknown.
- **Telemetry path:** The exact nested result field containing a non-finite value or serialization failure.

# Notes

Architecture handoff: the sibling experiment fitting/status/checkpoint boundary applies. CFFG Engine, compiler, VM, factor equations, and serialization are N/A unless a focused trace proves a core defect.

Duplicated surfaces: both CFFG tiers, both SLDS controls, inner and outer statuses, terminal inference, runtime certificate views, fit-summary sanitization, per-arm checkpoints, manifests, and classic periodicity checkpoint export must stay aligned.

Hostile-input decisions: non-finite residual or free energy is divergence, exhausted status is never convergence evidence, float32 guards use dtype precision, exact-zero support stays exact, and the 10,000-step training scale is the uncomfortable-size case.

Legend: prerequisite --> dependent

```text
[x] heterogeneous selected-scale budget (cff-tlh3)
      |
      +--> [ ] classic family convergence measurement (this ticket)
                  |
                  +--> [ ] cff-8w4c staged prior screen
                  |
                  +--> [ ] cff-3v7q periodicity replay
```



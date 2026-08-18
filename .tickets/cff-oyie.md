---
id: cff-oyie
status: open
deps: [cff-huxt]
links: []
created: 2026-08-10T02:58:08Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, statistics]
---
# Run the dual-family confirmatory rSLDS comparison

## Background in Plain English

cff-huxt freezes one final protocol before any final dataset exists. This ticket is the only owner allowed to generate and fit those final datasets. It executes the frozen protocol without changing its scientific choices.

The protocol asks two separate questions. The classic family uses Linderman's source-homogeneous NASCAR generator, fits the paper's 10,000-step training scale, and adds a registered 2,500-step held-out extension. It tests empirical non-inferiority and absolute adequacy on that declared workflow, not an exact replay of Figure 3. The extension family uses distinct source-specific gates and tests whether CFFG also recovers source-dependent transition functions on a shared state grid.

The classic claim is tested first. A heterogeneous claim is unlocked only if the classic claim passes. A failure in either family remains visible and never erases the other family's measurements.

## Outcomes / Acceptance Criteria

1. Final execution starts from the committed cff-huxt preregistration content hash. The experiment code, CFFG code, environments, generators, adapter, seeds, arms, budgets, stopping rules, metrics, alignment rules, thresholds, multiplicity rule, and failure policy match that freeze.
2. Every registered family, final seed, replicate, primary tier, Gibbs chain, and learned SLDS control runs or receives an explicit failed or not-ready record. No seed is dropped, replaced, or pooled away.
3. Each source-heterogeneous final dataset records source-by-gate exposures, successes, failures, and augmented-design rank before its fitted effects can support a capability claim. Every source/gate cell must have at least 20 exposures, 5 successes, 5 failures, and rank 3; a miss produces an explicit not-ready record under the frozen policy.
4. Neither CFFG nor a reference fitter receives generating states, regimes, parameters, held-out observations, or truth-derived alignment information before fitting finishes.
5. Both primary tiers report paired per-seed held-out predictive log likelihood, affine-aligned latent RMSE, label-aligned regime accuracy, runtime status, and every registered secondary outcome for both families.
6. The classic result applies its registered simultaneous uncertainty procedure to every paired and absolute primary effect. It passes only when all registered classic effects pass and every required classic coordinate is ready.
7. The heterogeneous result applies its registered simultaneous uncertainty procedure to every paired effect, shared-grid worst-source transition error, and source-contrast recovery effect. Its confirmatory claim is unlocked only after the classic result passes and only when every required heterogeneous effect and coordinate passes.
8. A failed classic result locks the joint and heterogeneous confirmatory claims but preserves descriptive heterogeneous results. A failed heterogeneous result does not erase a passed classic result. The report states each family decision and the joint decision separately.
9. The same-`K` and closest-free-scalar-budget learned SLDS controls use the identical final observations, splits, and family-qualified seeds as their corresponding primary arms.
10. The pinned unmodified upstream learner is reported once as a narrower recurrence-only workflow baseline. It is not used to claim same-model, parameter, authors-code, or algorithm parity.
11. Secondary reporting includes aligned parameter-family errors where identifiable, bounded long-rollout periodicity and bounded-rollout rate, functional transition recovery, outer iterations, inner work, phase timings, wall time, and peak resident memory. Raw recurrence coefficients remain excluded from primary claims because affine coordinates, regime labels, and decision-list gate order do not define a unique coefficient correspondence.
12. Aggregate tables retain every failed and not-ready coordinate. They include raw paired effects, family-wise simultaneous intervals, absolute-effect decisions, the classic-first gatekeeping decision, and available secondary evidence rather than only means or successful fits.
13. Artifacts record per-family generator and data identities; experiment, CFFG, upstream, adapter, legacy-environment, lock, mapping, and evaluation-grid hashes; exact commands; hardware; seeds; configurations; observation hashes; raw metrics; statuses; timing; memory; preregistration identity; append-only attempt ledgers; retained posterior samples; and a complete manifest hash.
14. Artifact validation recomputes every registered family aggregate, interval, threshold decision, gatekeeping result, runtime summary, and available secondary summary from raw coordinate records. It rejects missing, changed, non-finite, or unmanifested payloads.
15. README.md states the supported family and joint conclusions in plain language. docs/ARCHITECTURE.md records the generator, reference, fitting, checkpoint, and artifact boundaries. docs/LIMITATIONS.md records the tied-source classic limitation, local extension boundary, affine and label identifiability limits, registered scales, every miss or non-convergence, and the exclusion of authors-code and universal algorithm parity.
16. If a registered effect, eligibility gate, or execution arm fails, the report says so and creates or links a ticket for each evidence-backed failing owner. This ticket does not change the frozen protocol or expand into repair work.
17. Exact-zero support, float32 survival, loud non-convergence, source-gate coverage, long-horizon boundedness, and paper-scale training retain the numerical and experimental contracts frozen by cff-huxt.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound method that satisfies the outcomes.

- Treat one family, seed, and replicate as the minimum durable coordinate, then validate and manifest it immediately.
- Keep every attempt append-only and derive all reports from immutable raw records.
- Do not inspect aggregate final effects until the registered seed set is complete unless the frozen failure policy requires an earlier stop.
- Reuse the dashboard-backed short-suite result when repository code is unchanged. Keep the marked slow confirmatory experiment outside the default pytest gate.

## Future work

cff-nkbj audits the completed result against the registered claim ladder and closes only the evidence layers that pass. Real observations, other switching-system families, balanced-tree transitions, and destination-softmax transitions remain separate roadmap work. Failed layers receive their own tickets under criterion 16.

## Definitions

- **Classic family:** The source-homogeneous canonical NASCAR generator fitted for 10,000 training steps with a separate 2,500-step held-out extension.
- **Confirmatory run:** Evidence collected after the protocol, thresholds, and failure rules have been frozen.
- **Fixed-sequence gatekeeping:** The classic claim is tested first, and only a passing classic claim unlocks the heterogeneous confirmatory claim.
- **Heterogeneous family:** The local NASCAR extension whose transition gates differ by source regime.
- **Not-ready record:** An explicit result stating that missing, invalid, or failed required evidence prevents the registered claim.
- **Primary tier:** One of the two same-model comparisons against the independent full paper-equation adapter.
- **Source contrast:** The difference between two source regimes' transition-probability functions on the same frozen state grid.

## Notes

Pre-implementation handoff: `docs/ARCHITECTURE.md` change-path row is N/A for engine ownership because final execution stays in the sibling validation repository and consumes the frozen CFFG public fitting interface. The final evidence commit still updates the project README, architecture boundary description, and limitations in place.

Duplicated execution surfaces checked before final execution:

- classic and heterogeneous generators, data identities, scales, splits, eligibility rules, and evaluation grids;
- shared-initialization and native-end-to-end CFFG and paper-adapter tiers, four isolated Gibbs chains per tier, and both learned SLDS controls;
- paired non-inferiority outcomes, family-specific absolute outcomes, fixed-sequence claim gatekeeping, secondary outcomes, and the narrow unmodified baseline;
- family-qualified append-only attempts, interruption and resume, retained posterior samples, raw aggregates, manifests, commands, environment records, and family-specific failure retention;
- CFFG compiler, VM, schedule, kernels, ledger, serialization, learning declarations, and factor implementations, which are N/A unless a separate prerequisite repairs a discovered engine defect.

Hostile-input decisions:

- Sparse or one-sided source/gate data and rank-deficient heterogeneous designs produce a registered not-ready record before they can support a capability claim. The tied-source classic family uses its registered held-out on-support outcome instead of an unidentified counterfactual source-row outcome.
- Improper, vacuous, non-finite, divergent, budget-exhausted, or otherwise failed fitter results remain explicit failed or not-ready evidence. They never become a passing coordinate.
- Final execution uses the frozen float64 protocol. Existing float32 generator, functional-score, and engine regressions remain required consumer-survival evidence; this ticket does not weaken them.
- The 12,500-step classic dataset and registered 10,000-step rollouts exercise the declared long-horizon case. Results remain bounded to the registered families and scales and do not establish universal asymptotic behavior.

Legend: prerequisite --> dependent

```text
[x] cff-9o39 restartable dual execution
      |
      +--> [ ] cff-s6rs measured calibration (in progress)
                  |
                  +--> [ ] cff-huxt immutable dual freeze
                              |
                              +--> [ ] cff-oyie final execution
                                          |
                                          +--> [ ] cff-nkbj evidence audit
```

cff-s6rs is the active root. cff-oyie remains blocked until calibration and the replacement freeze are complete.

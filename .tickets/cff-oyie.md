---
id: cff-oyie
status: closed
deps: [cff-huxt]
links: [cff-wea9, cff-l5ag, cff-q1au, cff-qulu, cff-0oqo, cff-3v7q]
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

The protocol asks two separate questions. The classic family uses Linderman's source-homogeneous NASCAR generator, fits the paper's 10,000-step training scale, and adds a registered 2,500-step held-out extension. Across exactly three untouched final datasets, it tests whether CFFG performs roughly as well as the independent paper-equation adapter while meeting absolute NASCAR task-adequacy rules. This is the task-performance half of the requested parity demonstration, not an exact replay of Figure 3 and not a population-wide statistical claim.

The extension family uses distinct source-specific gates and tests, on exactly three different untouched final datasets, whether CFFG also recovers source-dependent transition functions on a shared state grid. It is reported independently: its pass or failure does not alter the classic parity result.

## Outcomes / Acceptance Criteria

1. Final execution starts from the committed cff-huxt preregistration content hash. The experiment code, CFFG code, environments, generators, adapter, seeds, arms, budgets, stopping rules, metrics, alignment rules, thresholds, decision rules, and failure policy match that freeze.
2. All three registered final datasets in each family, and every registered replicate, primary tier, Gibbs chain, and learned SLDS control, run or receive an explicit failed or not-ready record. No seed is dropped, replaced, or pooled away.
3. Each source-heterogeneous final dataset records source-by-gate exposures, successes, failures, and augmented-design rank before its fitted effects can support a capability claim. Every source/gate cell must have at least 20 exposures, 5 successes, 5 failures, and rank 3; a miss produces an explicit not-ready record under the frozen policy.
4. Neither CFFG nor a reference fitter receives generating states, regimes, parameters, held-out observations, or truth-derived alignment information before fitting finishes.
5. Both primary tiers report paired per-seed held-out predictive log likelihood, affine-aligned latent RMSE, label-aligned regime accuracy, runtime status, and every registered secondary outcome for both families.
6. The classic task result passes only when all three classic datasets are ready, every registered absolute NASCAR adequacy rule passes, and the registered CFFG outcomes remain within the frozen practical tolerances of the independent paper-equation adapter. The report includes all raw per-dataset results and summary variation without describing three datasets as population-level non-inferiority.
7. The heterogeneous result passes only when all three heterogeneous datasets are ready and every registered source-design, shared-grid worst-source transition, and source-contrast recovery rule passes.
8. The two family decisions are independent. A failure in either family does not erase, lock, or relabel the other family's result. The report does not turn both passing results into a stronger joint statistical claim.
9. The same-`K` and closest-free-scalar-budget learned SLDS controls use the identical final observations, splits, and family-qualified seeds as their corresponding primary arms.
10. The pinned unmodified upstream learner is reported once as a narrower recurrence-only workflow baseline. It is not used to claim same-model, parameter, authors-code, or algorithm parity.
11. Secondary reporting includes aligned parameter-family errors where identifiable, bounded long-rollout periodicity and bounded-rollout rate, functional transition recovery, outer iterations, inner work, phase timings, wall time, and peak resident memory. Raw recurrence coefficients remain excluded from primary claims because affine coordinates, regime labels, and decision-list gate order do not define a unique coefficient correspondence.
12. Aggregate tables retain every failed and not-ready coordinate. They include raw paired effects, per-family summaries, every absolute decision, and available secondary evidence rather than only means or successful fits.
13. Artifacts record per-family generator and data identities; experiment, CFFG, upstream, adapter, legacy-environment, lock, mapping, and evaluation-grid hashes; exact commands; hardware; seeds; configurations; observation hashes; raw metrics; statuses; timing; memory; preregistration identity; append-only attempt ledgers; retained posterior samples; and a complete manifest hash.
14. Artifact validation recomputes every registered family aggregate, tolerance and threshold decision, runtime summary, and available secondary summary from raw coordinate records. It rejects missing, changed, non-finite, or unmanifested payloads.
15. The experiment README states both family conclusions in plain language. cff-nkbj owns the corresponding final CFFG README, architecture, and limitations audit.
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
- **Heterogeneous family:** The local NASCAR extension whose transition gates differ by source regime.
- **Not-ready record:** An explicit result stating that missing, invalid, or failed required evidence prevents the registered claim.
- **Primary tier:** One of the two same-model comparisons against the independent full paper-equation adapter.
- **Source contrast:** The difference between two source regimes' transition-probability functions on the same frozen state grid.
- **Task parity:** On all three fixed classic final datasets, CFFG meets the registered NASCAR adequacy rules and performs within the frozen practical tolerances of the independent paper-equation adapter.

## Notes

Pre-implementation handoff: `docs/ARCHITECTURE.md` change-path row is N/A for engine ownership because final execution stays in the sibling validation repository and consumes the frozen CFFG public fitting interface. The final evidence commit still updates the project README, architecture boundary description, and limitations in place.

Duplicated execution surfaces checked before final execution:

- classic and heterogeneous generators, data identities, scales, splits, eligibility rules, and evaluation grids;
- shared-initialization and native-end-to-end CFFG and paper-adapter tiers, four isolated Gibbs chains per tier, and both learned SLDS controls;
- paired comparison outcomes, family-specific absolute outcomes, independent family decisions, secondary outcomes, and the narrow unmodified baseline;
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
      +--> [ ] cff-x5z2 recurrence statistics (in progress)
                  |
                  +--> [ ] cff-266l readiness semantics
                              |
                              +--> [ ] cff-s6rs measured calibration
                                          |
                                          +--> [ ] cff-huxt immutable 3+3 freeze
                                                      |
                                                      +--> [ ] cff-oyie 3+3 execution
                                                                  |
                                                                  +--> [ ] cff-nkbj evidence audit
```

cff-x5z2 is the active root. cff-oyie remains blocked until calibration and the replacement freeze are complete.

**2026-08-20T15:48:53Z**

Final handoff: the only authorized schema-v5 command completed all six fixed coordinates without seed replacement or scientific changes. The local experiment evidence commit is 6c0437a. results.json SHA-256 is eaf4bdccb6b2d9ecf35f59a91fa1012873bba697c8fe499781b28c1f787697c4; manifest SHA-256 is b71c41ad46cd2e210a9e204743d8fd58f1a76a69305afae8601c7e307b64a56a. The manifest, raw dual aggregate recomputation, and final provenance validator all pass. Execution wall time was 51,254.918 seconds with 13,230,833,664-byte peak RSS. The interrupted seed-2001 attempt remains append-only evidence of the missing legacy-source preflight defect tracked by cff-wea9.

Both family conclusions are not_ready and remain independent. All six classic CFFG tiers and controls converged, but all six classic adapter arms failed the frozen mixing eligibility rule, so no paired effects or formal absolute decisions were issued. Heterogeneous seed 2101 had converged CFFG tiers but ineligible adapter mixing and budget-exhausted controls; seed 2102 stopped before fitting because source 3 gate 2 had four successes instead of five; seed 2103 had budget-exhausted CFFG tiers and controls plus ineligible adapter mixing. Descriptively, classic native seed 2001 periodicity was 0.4058 against its 0.4110 floor. Heterogeneous seed 2101 transition and contrast errors were 0.1946/0.2498 shared and 0.3761/0.3736 native against 0.150 ceilings. The experiment README reports these boundaries and does not claim task parity or source recovery.

Evidence-backed repairs: cff-l5ag owns adapter mixing; cff-q1au owns final-path source/gate coverage; cff-qulu owns outer convergence; cff-0oqo owns source-function recovery after q1au and qulu; cff-3v7q owns classic periodicity robustness; cff-wea9 owns early legacy-source preflight.

Legend: prerequisite --> dependent

[x] cff-huxt immutable freeze --> [x] cff-oyie frozen execution --> [ ] cff-nkbj evidence audit
                                              |
                                              +--> [ ] cff-l5ag adapter mixing
                                              +--> [ ] cff-q1au coverage --+
                                              +--> [ ] cff-qulu convergence+--> [ ] cff-0oqo recovery
                                              +--> [ ] cff-3v7q periodicity
                                              +--> [ ] cff-wea9 preflight

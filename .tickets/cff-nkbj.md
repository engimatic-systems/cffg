---
id: cff-nkbj
status: closed
deps: [cff-oyie, cff-ora6]
links: [cff-r6bl, cff-0f3g, cff-dlrj, cff-11jj]
created: 2026-08-09T13:09:26Z
type: epic
priority: 1
assignee: coopwilliams
tags: [rslds, validation, learning, statistics]
---
# Epic: Demonstrate classic rSLDS parity and source-heterogeneous recovery

## Background in Plain English

cff-l7y4 showed that CFFG could fit one synthetic NASCAR rSLDS from observations. It did not provide a multi-seed comparison with an independent implementation, and the classic NASCAR generator ties its recurrence gates across source regimes. That classic dataset therefore cannot show that a learner recovers distinct source-specific gates.

For this epic, parity has two concrete parts. First, CFFG must express the same probabilistic architecture as the classic rSLDS: its factor types and their composition must represent the same distributions and dependencies under a documented parameter mapping. This does not require identical source code or an identical inference algorithm. Second, CFFG must perform roughly as well as the classic paper-equation method on the NASCAR task.

The empirical demonstration uses exactly three untouched final datasets from each of two families, with no seed dropping. The classic family preserves Linderman's source-homogeneous NASCAR generator and fits 10,000 training steps, the scale reported in the paper, with a separate 2,500-step held-out extension. It owns the task-parity result. The heterogeneous family keeps the NASCAR dynamics but gives each source regime distinct transition gates and checks recovery on the same frozen state grid for every source. It is an independent extension showing a capability the classic dataset cannot test.

This is a bounded engineering demonstration, not a population-wide statistical result. It does not claim identical code, identity between inference algorithms, or universal performance across every rSLDS problem.

## Outcomes / Acceptance Criteria

1. The declared-learning foundation through cff-dqk2 and cff-11jj is closed, and the comparison uses only the serialized `LearnableSpec` fitting path rather than a manual parameter-update compatibility path.
2. The independent paper-equation adapter, destination-regime dynamics correction, locked legacy environment, numerical oracles, restartable dual runner, family generators, source-design audit, secondary outcomes, and scale benchmark are closed and recorded in their owning tickets.
3. The classic generator's parameter arrays, legacy `RandomState` stream, and golden trajectory hash match the pinned canonical fixtures. The heterogeneous generator has distinct source rows, common-state source dependence, full-rank source/gate designs, registered exposure and binary-outcome gates, deterministic streams, and bounded long rollouts.
4. The architecture evidence identifies each classic rSLDS probability distribution and dependency, its owning CFFG factor type or graph composition, the parameter mapping, and an executable oracle or fixture that would fail if their probabilistic semantics differed.
5. cff-s6rs closes with complete calibration-only evidence for both families, family-specific scales and splits, fixed optimization and Gibbs budgets, fixed practical comparison tolerances and absolute thresholds, whole-run projections, complete manifests, and no generated or fitted final seed.
6. cff-huxt closes with one immutable dual-family preregistration for exactly three final seeds per family. It reconstructs from cff-s6rs, records the superseded one-family freeze without executing it, authorizes only cff-oyie to run final seeds, and validates from a clean checkout.
7. cff-oyie closes with all six registered final datasets and every registered tier, Gibbs chain, learned control, failure, aggregate, decision, retained posterior, attempt ledger, provenance record, and manifest reported and independently recomputable. No final seed is dropped or replaced.
8. The classic conclusion, if all three classic datasets are ready and pass their registered rules, states that CFFG instantiates the classic rSLDS probabilistic architecture and, on three fixed untouched paper-scale classic NASCAR datasets, performed roughly as well as the independent paper-equation adapter while meeting the registered absolute task-adequacy rules.
9. The classic conclusion does not call the registered workflow an exact Figure 3 replay because the paper did not use the held-out extension, and it does not generalize from three datasets to a population-wide non-inferiority claim.
10. The heterogeneous conclusion, if all three heterogeneous datasets are ready and pass their registered rules, states separately that CFFG recovered source-dependent transition behavior on the local extension. It is not required for the classic parity conclusion.
11. A failure in one family does not erase, lock, or relabel the other family's result. The epic closes with an honest complete report and creates repair tickets for failed evidence layers rather than changing the frozen protocol.
12. The pinned unmodified authors' learner remains a separately labeled recurrence-only secondary baseline. It is not described as a same-model reference for source-specific learning.
13. The final wording explicitly excludes authors-code identity, inference-algorithm identity, and universal rSLDS-family performance.
14. README.md, docs/ARCHITECTURE.md, and docs/LIMITATIONS.md agree with the final artifact about the two-part parity definition, both family boundaries, scales, reference provenance, runtime statuses, results, missed gates, identifiability limits, and claim exclusions.
15. Every branch in the stacked roadmap has atomic commits, an updated ticket handoff, the validation required by AGENTS.md for its change type, a clean worktree, and a PR body that follows the structure used by PR #63.
16. Balanced-tree, explicit gate-sharing, destination-softmax, real-observation, and other switching-system work remains linked future work and does not block this decision-list evidence result.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound method that satisfies the outcomes.

- Treat calibration, freeze, and final execution as irreversible phase barriers. Never inspect a final seed before the replacement freeze.
- Keep classic and heterogeneous results and failures separate in raw records and reports.
- Keep the independent same-model adapter separate from the narrower unmodified-upstream baseline.
- Put any evidence-backed repair in a new prerequisite ticket rather than changing the frozen protocol after final execution begins.

## Future work

cff-0f3g owns explicit tree topology, balanced-tree inference, gate sharing, learning, and scale validation. cff-dlrj owns the destination-softmax recurrence derivation. Real observations and other switching-system datasets remain future validation after this synthetic two-family comparison. None of that work is a prerequisite for closing this epic.

## Definitions

- **Absolute adequacy rule:** A registered ceiling or floor that CFFG must meet against generating truth or bounded behavior, independent of agreement with the reference adapter.
- **Probabilistic-architecture parity:** The CFFG factor types and their composition represent the same probability distributions and dependencies as the classic rSLDS formulation under the documented mapping.
- **Classic family:** The source-homogeneous canonical NASCAR generator fitted for 10,000 training steps with a separate 2,500-step held-out extension.
- **Heterogeneous family:** The local NASCAR extension whose transition gates differ by source regime.
- **Paper-equation adapter:** A separately versioned implementation of the published equations based on pinned legacy code, not the authors' unmodified executable learner.
- **Source contrast:** The difference between two source regimes' transition-probability functions evaluated on the same frozen state grid.
- **Task parity:** On all three fixed classic final datasets, CFFG meets the registered NASCAR adequacy rules and performs within the frozen practical tolerances of the independent paper-equation adapter.

## Notes

Pre-implementation handoff: this epic coordinates child work and owns the final claim audit. `docs/ARCHITECTURE.md` has no engine change-path row for the sibling validation harness. The final evidence landing updates the existing rSLDS consumer-validation section plus README.md and docs/LIMITATIONS.md; docs/DERIVATIONS.md is N/A unless a new mathematical message or free-energy derivation is introduced.

Duplicated surfaces checked across the epic:

- both dataset families, primary tiers, four reference chains, learned controls, and the narrow unmodified baseline;
- calibration, scale selection, immutable freeze, final generation, append-only resume, aggregation, manifests, and final documentation;
- paired outcomes, family-specific absolute outcomes, independent family decisions, failed-family retention, and explicit excluded claims;
- compiler, VM, schedule, kernels, ledger, serialization, and core factor equations, which are N/A after cff-11jj unless a separately owned engine defect is discovered.

Legend: prerequisite --> dependent

```text
[x] dual-family harness --> [x] cff-x5z2 --> [x] cff-266l
                                                   |
                                                   +--> [ ] cff-s6rs (in progress)
                                                                 |
                                                                 +--> [ ] cff-huxt
                                                                           |
                                                                           +--> [ ] cff-oyie
                                                                                     |
[x] factor-semantics audit --> [x] cff-ora6 --------------------------------+--> [ ] cff-nkbj

Linked future work: cff-0f3g balanced tree; cff-dlrj destination softmax
```

Research question → Does classic NASCAR test distinct source gates?

Answer → No. Its recurrence weights and offsets are broadcast across all four source regimes, and source regime is confounded with naturally occupied state support. The classic family therefore uses held-out on-support transition adequacy and makes no source-row recovery claim.

Source → `src/cffg_nascar_validation/nascar.py`, the pinned `recurrent-slds` NASCAR example, and the calibration-only seed-3101 counterfactual-grid audit recorded in cff-s6rs.

Confidence → High.

Research question → What may both passing families support?

Answer → The architecture audit may establish that the CFFG graph has the classic rSLDS probabilistic semantics. If all three fixed classic datasets pass, the classic run may add a bounded task-parity statement at the paper training scale plus held-out extension. The heterogeneous family may separately add broader evidence for source-dependent transition-function recovery. Together they do not establish identical source code, identical inference algorithms, or universal rSLDS performance.

Source → Linderman et al. 2017 section 5.1; the pinned Figure 3 example; cff-i00o's registered claim schema; and cff-s6rs's calibration boundary.

Confidence → High for the claim separation and exclusions; final support depends on the preregistered results.

### 2026-08-18 — probabilistic-architecture audit

Question → Does the implemented graph represent the classic NASCAR rSLDS
probability model, and what evidence is still missing?

Answer → Yes for the source-homogeneous, shared-emission NASCAR subclass.
The graph has a categorical initial regime, a tied Gaussian initial latent
state, a normalized source-indexed decision-list recurrence, destination-indexed
linear-Gaussian dynamics, and a shared linear-Gaussian emission. Equality nodes
only split graph degree and add no probability term. CFFG and the independent
adapter use different inference algorithms: variational Pólya–Gamma/JJ bounds
and Gaussian moment collapse versus Pólya–Gamma Gibbs sampling. Existing direct
oracles strongly cover recurrence and destination dynamics. The audit found
that initial-regime, initial-Gaussian, and emission conversions needed direct
cross-mapping assertions; cff-ora6 added them and a versioned five-term
architecture table. Both fit interfaces are truth-clean: generating states,
regimes, and parameters first enter after fitting, during evaluation.

Source → Linderman et al. 2017 equations and Section 5.1; pinned
`recurrent-slds` NASCAR source; CFFG `models/rslds.py`, `recurrence.py`,
`mixg.py`, and `gauss.py`; sibling experiment `paper_adapter.py`,
`comparison_adapter.py`, `rslds_mapping.py`, and their executable tests.

Confidence → High for the local NASCAR factorization, all five mapped joint
terms, the transition and dynamics equations, and the truth boundary after
cff-ora6 experiment commit `61cec3f` passed both modern and isolated legacy
test suites.

**2026-08-20T15:56:39Z**

Final evidence audit: the epic reaches only the first half of the requested parity target. cff-ora6 establishes probabilistic-architecture parity for the source-homogeneous, shared-emission classic NASCAR subclass through a five-term mapping and independent oracles. cff-oyie does not establish classic task parity: all six classic CFFG tiers converged, but all six adapter tiers failed the frozen mixing gate, so no paired or formal absolute decision was eligible. The independent heterogeneous claim is also not_ready because seed 2102 failed prefit coverage, seed 2103 exhausted both CFFG outer budgets, every fitted adapter tier failed mixing, and the one converged seed had descriptive source-function errors above 0.150.

README.md, docs/ARCHITECTURE.md, and docs/LIMITATIONS.md now agree on both family boundaries, 10,000+2,500 and 1,280+320 scales, reference provenance, runtime statuses, descriptive misses, affine/label identifiability, and excluded claims. The documentation commit is fd04c98. The complete local experiment commit is 6c0437a; results SHA-256 is eaf4bdccb6b2d9ecf35f59a91fa1012873bba697c8fe499781b28c1f787697c4 and manifest SHA-256 is b71c41ad46cd2e210a9e204743d8fd58f1a76a69305afae8601c7e307b64a56a. Manifest, raw aggregate, final provenance, and canonical-document consistency checks pass.

Repair owners are cff-l5ag adapter mixing, cff-q1au heterogeneous coverage, cff-qulu outer convergence, cff-0oqo source-function recovery, cff-3v7q classic periodicity, and cff-wea9 source preflight. Balanced trees and destination-softmax remain cff-0f3g and cff-dlrj.

Stack audit: all 19 prerequisite and child tickets through cff-oyie and cff-ora6 are closed; all 20 existing branch heads equal their origin revisions; every draft PR uses the four PR #63 headings; checked worktrees are clean; tk dep cycle and git diff --check pass. The final cff-nkbj PR body is staged in the agent-share directory with the same heading contract. No short suite is required for this documentation-and-ticket-only convergence branch; owning code branches retain their recorded final tests.

Legend: prerequisite --> dependent

[x] cff-dqk2 --> [x] cff-11jj --> [x] dual harness --> [x] cff-oyie --+
[x] cff-ora6 architecture mapping ----------------------------------------+--> [x] cff-nkbj
                                                                                |
                                                                                +--> [ ] repair tickets above

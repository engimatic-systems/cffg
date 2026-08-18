---
id: cff-nkbj
status: in_progress
deps: [cff-oyie]
links: [cff-r6bl, cff-0f3g, cff-dlrj, cff-11jj]
created: 2026-08-09T13:09:26Z
type: epic
priority: 1
assignee: coopwilliams
tags: [rslds, validation, learning, statistics]
---
# Epic: Establish classic rSLDS evidence and source-heterogeneous recovery

## Background in Plain English

cff-l7y4 showed that CFFG could fit one synthetic NASCAR rSLDS from observations. It did not provide a multi-seed comparison with an independent implementation, and the classic NASCAR generator ties its recurrence gates across source regimes. That classic dataset therefore cannot show that a learner recovers distinct source-specific gates.

This epic uses two separate dataset families. The classic family preserves Linderman's source-homogeneous NASCAR generator and fits 10,000 training steps, the scale reported in the paper, with a separate 2,500-step held-out extension. The heterogeneous family keeps the NASCAR dynamics but gives each source regime distinct transition gates and checks recovery on the same frozen state grid for every source.

The two families answer different questions. A passing classic result can support empirical non-inferiority and absolute adequacy on the exact canonical generator at the declared training scale and held-out extension. A passing heterogeneous result can add evidence that CFFG recovers source-dependent transition functions. The classic claim is tested first, and only a passing classic claim unlocks the heterogeneous confirmatory claim.

Neither result establishes parity with the authors' executable code, identity between inference algorithms, validation of CFFG message or free-energy equations, or universal rSLDS performance.

## Outcomes / Acceptance Criteria

1. The declared-learning foundation through cff-dqk2 and cff-11jj is closed, and the comparison uses only the serialized `LearnableSpec` fitting path rather than a manual parameter-update compatibility path.
2. The independent paper-equation adapter, destination-regime dynamics correction, locked legacy environment, numerical oracles, restartable dual runner, family generators, source-design audit, secondary outcomes, and scale benchmark are closed and recorded in their owning tickets.
3. The classic generator's parameter arrays, legacy `RandomState` stream, and golden trajectory hash match the pinned canonical fixtures. The heterogeneous generator has distinct source rows, common-state source dependence, full-rank source/gate designs, registered exposure and binary-outcome gates, deterministic streams, and bounded long rollouts.
4. cff-s6rs closes with complete calibration-only evidence for both families, family-specific scales and splits, fixed optimization and Gibbs budgets, fixed paired and absolute thresholds, whole-run projections, complete manifests, and no generated or fitted final seed.
5. cff-huxt closes with one immutable dual-family preregistration that reconstructs from cff-s6rs, records the superseded one-family freeze without executing it, authorizes only cff-oyie to run final seeds, and validates from a clean checkout.
6. cff-oyie closes with every registered final family, seed, tier, Gibbs chain, learned control, failure, aggregate, interval, decision, retained posterior, attempt ledger, provenance record, and manifest reported and independently recomputable.
7. The classic conclusion, if every registered classic effect and coordinate passes, states that CFFG was empirically non-inferior to the independent paper-equation adapter and met its registered absolute adequacy gates on the exact source-homogeneous canonical generator with 10,000 fitted steps and a separate 2,500-step held-out extension.
8. The classic conclusion does not call the registered workflow an exact Figure 3 replay because the paper did not use the held-out extension.
9. The heterogeneous conclusion, if the classic claim is unlocked and every registered heterogeneous effect and coordinate passes, states that CFFG was empirically non-inferior to the independent adapter and met the registered shared-grid worst-source transition and source-contrast recovery gates on the local source-heterogeneous extension.
10. A failed classic result locks the joint and heterogeneous confirmatory claims but preserves descriptive heterogeneous results. A failed heterogeneous result does not erase a passed classic result. The epic closes with an honest complete report and creates repair tickets for failed evidence layers rather than changing the frozen protocol.
11. The pinned unmodified authors' learner remains a separately labeled recurrence-only secondary baseline. It is not described as a same-model reference for source-specific learning.
12. The final wording explicitly excludes authors-code parity, inference-algorithm identity, message-equation parity, free-energy validation, and universal rSLDS-family parity.
13. README.md, docs/ARCHITECTURE.md, and docs/LIMITATIONS.md agree with the final artifact about both family boundaries, scales, reference provenance, runtime statuses, results, missed gates, identifiability limits, and claim exclusions.
14. Every branch in the stacked roadmap has atomic commits, an updated ticket handoff, the validation required by AGENTS.md for its change type, a clean worktree, and a PR body that follows the structure used by PR #63.
15. Balanced-tree, explicit gate-sharing, destination-softmax, real-observation, and other switching-system work remains linked future work and does not block this decision-list evidence result.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound method that satisfies the outcomes.

- Treat calibration, freeze, and final execution as irreversible phase barriers. Never inspect a final seed before the replacement freeze.
- Keep classic and heterogeneous effects, failures, and claims separate in raw records and reports.
- Keep the independent same-model adapter separate from the narrower unmodified-upstream baseline.
- Put any evidence-backed repair in a new prerequisite ticket rather than changing the frozen protocol after final execution begins.

## Future work

cff-0f3g owns explicit tree topology, balanced-tree inference, gate sharing, learning, and scale validation. cff-dlrj owns the destination-softmax recurrence derivation. Real observations and other switching-system datasets remain future validation after this synthetic two-family comparison. None of that work is a prerequisite for closing this epic.

## Definitions

- **Absolute adequacy gate:** A registered ceiling or floor that CFFG must meet against generating truth or bounded behavior, independent of agreement with the reference adapter.
- **Classic family:** The source-homogeneous canonical NASCAR generator fitted for 10,000 training steps with a separate 2,500-step held-out extension.
- **Empirical non-inferiority:** Every registered paired effect's simultaneous lower confidence bound meets its frozen margin against the independent adapter.
- **Fixed-sequence gatekeeping:** The classic claim is tested first, and only a passing classic claim unlocks the heterogeneous confirmatory claim.
- **Heterogeneous family:** The local NASCAR extension whose transition gates differ by source regime.
- **Paper-equation adapter:** A separately versioned implementation of the published equations based on pinned legacy code, not the authors' unmodified executable learner.
- **Source contrast:** The difference between two source regimes' transition-probability functions evaluated on the same frozen state grid.

## Notes

Pre-implementation handoff: this epic coordinates child work and owns the final claim audit. `docs/ARCHITECTURE.md` has no engine change-path row for the sibling validation harness. The final evidence landing updates the existing rSLDS consumer-validation section plus README.md and docs/LIMITATIONS.md; docs/DERIVATIONS.md is N/A unless a new mathematical message or free-energy derivation is introduced.

Duplicated surfaces checked across the epic:

- both dataset families, primary tiers, four reference chains, learned controls, and the narrow unmodified baseline;
- calibration, scale selection, immutable freeze, final generation, append-only resume, aggregation, manifests, and final documentation;
- paired outcomes, family-specific absolute outcomes, fixed-sequence gatekeeping, failed-family retention, and explicit excluded claims;
- compiler, VM, schedule, kernels, ledger, serialization, and core factor equations, which are N/A after cff-11jj unless a separately owned engine defect is discovered.

Legend: prerequisite --> dependent

```text
[x] cff-8xve + cff-i00o + cff-9o39
                    |
                    +--> [ ] cff-s6rs calibration (in progress)
                                  |
                                  +--> [ ] cff-huxt freeze
                                                |
                                                +--> [ ] cff-oyie final execution
                                                              |
                                                              +--> [ ] cff-nkbj audit

Linked future work: cff-0f3g balanced tree; cff-dlrj destination softmax
```

Research question → Does classic NASCAR test distinct source gates?

Answer → No. Its recurrence weights and offsets are broadcast across all four source regimes, and source regime is confounded with naturally occupied state support. The classic family therefore uses held-out on-support transition adequacy and makes no source-row recovery claim.

Source → `src/cffg_nascar_validation/nascar.py`, the pinned `recurrent-slds` NASCAR example, and the calibration-only seed-3101 counterfactual-grid audit recorded in cff-s6rs.

Confidence → High.

Research question → What may both passing families support?

Answer → The classic family may support bounded empirical non-inferiority and adequacy on the exact canonical generator at the declared paper training scale plus held-out extension. The heterogeneous family may add broader evidence for source-dependent transition-function recovery. Together they do not establish authors-code parity, algorithm identity, CFFG-paper fidelity, or universal rSLDS performance.

Source → Linderman et al. 2017 section 5.1; the pinned Figure 3 example; cff-i00o's registered claim schema; and cff-s6rs's calibration boundary.

Confidence → High for the claim separation and exclusions; final support depends on the preregistered results.

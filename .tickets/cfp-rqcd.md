---
id: cfp-rqcd
status: closed
deps: []
links: [cfp-okq2, cfp-9806]
created: 2026-11-22T01:00:18Z
type: chore
priority: 1
assignee: coopwilliams
---
# Scope parallel reactor and automatic schedule work after PR 185

## Background in Plain English — WHY

Cooper approved continuing the PR chain. The cold-start study supplied a fast reference and left two independent questions: how the reactor behaves in closed loop, and how Engine.run can safely choose that calculation.

## Outcomes / Acceptance Criteria — WHAT

1. Both workstreams have ready entry tickets with the reference and PR tip identified.
2. Open tickets have been screened for overlap, with evidence and dispositions for candidates.
3. The dependency records and a compact tree agree about the two starts, their dependents and the existing PCGym roadmap.

## Methodological Suggestions — HOW

Use tk ready, blocked and dep tree --full. Ticket-only change. Runtime execution surfaces are N/A.

## Future work

Execution stays in the tickets linked in the handoff below.

## Definitions

A closed loop feeds each action to the simulator and plans again from the resulting observation.


## Notes

**2026-11-22T01:02:26Z**

PR #185 at f9df9d6 is the shared upstream. The cold-start reference at 265ff14 supplies the completed mathematical prerequisite. Both starts are P1 and ready. The experiment uses the external PCGym checkout; capture its imported CFFG path and commit. The implementation uses a separate CFFG branch.

Legend: prerequisite --> dependent; [x] completed; [R] ready; [ ] blocked.

[x] cfp-rh8k (checked reference)
  +--> [R] cfp-9806 reactor pilot --> [ ] cfp-lsxw paired episodes
  |                                       --> [ ] cff-wt1c synthesis --> cff-xa6z
  +--> [R] cfp-hlqd chain recognition --> [ ] cfp-ru3t executor
                                          --> [ ] cfp-okq2 auto dispatch --> cff-xa6z

cff-wt1c additionally needs the matched feature-specific comparison arms. cff-xa6z also awaits cff-3wsw and cff-fj4v. Neither is a prerequisite for either ready root. Verified against tk dep tree --full, tk ready, tk blocked and tk dep cycle.

**2026-11-22T01:02:26Z**

Backlog disposition at f8b851b: screened all 56 previously open tickets (32 ready, 24 blocked), including the existing production scheduling ticket. No open ticket met all its closure criteria or lost its entire use case. Keep closed fixes opt-ewtf and cfp-rh8k as evidence; the pilot replays new real windows and the paired study supplies the fixed-model arm of cff-wt1c. No repeat of the nine-window startup investigation is required to begin.

Pruned 13 direct open prerequisites from cff-xa6z where a retained dependency still requires the same outcome. The rolling facade cff-wdhj can proceed from its serialized contract without waiting for publication of the entire recipe collection; cff-fj4v remains linked and both deliverables remain in the epic. Reclassified existing action status gating (cff-5edo), arena byte accounting (cff-j7mm), benign candidate/cache coverage (cff-g7rc), shared binding delegation (cff-4yr6) and exact Colony parity (cff-wbhu) as regression evidence. Their outstanding outcomes remain measurable. Updated cff-ahps to credit the completed extension, nonlinear and neural paths.

Scope checks: the discrete HMM fusion target in cff-yl0a concerns a different shape and timing boundary. Parameter-only spawn reuse, public readout batching and compact plates retain separate deliverables. Parameter uncertainty, bounds, hybrid partial-goal behavior and float32 accuracy all require evidence outside the fixed-parameter float64 reactor reference. Retained the tree-recurrence and softmax contracts, the nonstationarity experiment, structure selection and reduction derivations, visualization and glucose work: none tests the reactor shortcut. In particular, no-penalty goal directions in one Gaussian chain do not fix mixture collapse in cff-c5qf. Research confidence is high for the recorded ticket graph and cited source/test coverage; open performance targets and controller outcomes remain unmeasured here.

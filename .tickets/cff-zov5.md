---
id: cff-zov5
status: open
deps: [cff-x0if]
links: []
created: 2026-08-31T03:47:26Z
type: epic
priority: 2
assignee: coopwilliams
parent: cff-ahps
tags: [rslds, learning, hybrid, roadmap]
---
# Epic: Recover source-specific recurrent transition laws from observations

## Background in Plain English — WHY

In an rSLDS, the hidden discrete state is a regime. Ordinary hidden-regime
inference asks which regime was active. This epic asks a harder question:
after inferring those hidden regimes and the continuous trajectory from
observations, can the learner recover a different transition law for each
current regime?

The production `recurrence_link` already stores separate rows for separate
source regimes, and its direct learning algebra passes factor-local oracles.
The unresolved failure appears when the regimes and continuous trajectory are
both hidden. D9 represents a consecutive-regime pair but separates the whole
regime path from the continuous path. Historical development fits, including
the later D10 experiment, missed the fixed `0.150` source-function recovery
ceilings even when coverage, local statistics, and outer convergence were
healthy.

This epic follows a bounded sequence. cff-iur0 derives the smallest posterior
coordinate that preserves the missing regime/trajectory dependence. cff-ocxl
implements the accepted coordinate through a generic component boundary.
cff-x0if then runs one three-dataset, observation-only development gate. The
sequence does not port the historical whole-template D10 dispatcher: selecting
an inference method by recognizing one complete rSLDS layout would compromise
the repository's mix-and-match factor-graph vision.

## Outcomes / Acceptance Criteria — WHAT

1. cff-iur0 closes with a reviewed derivation and an explicit decision about
   the smallest posterior dependence required beyond D9.
2. cff-ocxl closes with that decision implemented as a reusable component
   coordinate whose semantics come from the serialized graph and declared
   constraints, not from an rSLDS model name or complete-template pattern.
3. The component exchanges ordinary boundary messages with surrounding
   factors and passes the three composition fixtures named by cff-ocxl.
4. Unsupported compositions fail during validation or compilation. Adding one
   factor or constraint cannot silently switch the region back to D9 or to a
   different free-energy objective.
5. Component inference, recurrence learning statistics, and Bethe free-energy
   ownership agree with independent small oracles and the repository's
   exact-support, float32, scale, and status contracts.
6. cff-x0if runs exactly the three declared development datasets and passes
   the unchanged `0.150` worst-source transition and source-contrast ceilings
   using observations as the only fitting data.
7. README.md, docs/ARCHITECTURE.md, docs/DERIVATIONS.md, and
   docs/LIMITATIONS.md state the demonstrated capability, approximation,
   composition boundary, and any remaining failure without claiming
   Linderman-code parity or general switching-system consistency.
8. No reference-adapter cohort, confirmatory seed set, long-rollout
   certification, balanced-tree transition, or destination-softmax transition
   is required to close this epic.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Treat the three children as phase barriers. Do not write production
  component code until cff-iur0 fixes the mathematical and objective contract.
- Reuse D9 factor rules, pair readouts, and learning APIs wherever the accepted
  derivation leaves their meaning unchanged.
- Use the old D10 mathematics and tiny oracles as research inputs only. Do not
  replay its whole-template compiler or engine dispatch.
- If a qualification gate fails, diagnose the existing three artifacts once
  and hand one evidence-backed owner to a new ticket. Do not add seeds,
  comparators, thresholds, or an unrelated repair to cff-x0if.

## Future work

The older cff-nkbj chain owns an external empirical comparison against a
paper-equation adapter. It is neither a prerequisite nor a substitute for this
epic and should not resume until this bounded observation-only capability is
established. cff-0f3g and cff-dlrj continue to own balanced-tree and
destination-softmax recurrence families. Real data and confirmatory
multi-seed comparisons remain later validation.

## Definitions

- **Component coordinate:** One declared inference update over a connected
  graph region, with explicit boundary messages and local objective ownership.
- **Current regime / source regime:** The discrete state that selects which
  recurrence row governs the next-regime transition.
- **D9:** The shipped structured recurrence approximation that preserves each
  consecutive-regime pair while separating the complete categorical and
  Gaussian paths.
- **Source-contrast error:** Error in the difference between two source rows'
  transition probabilities at the same continuous state.
- **Source-specific transition law:** A transition-probability function with a
  distinct parameter row for each current regime.

## Notes

Roadmap authoring handoff. Arrow direction is prerequisite --> dependent.

```text
[x] cff-s6yy structured recurrence baseline
      |
      +--> [ ] cff-iur0 derivation (ready)
                  |
                  +--> [ ] cff-ocxl generic component integration
                              |
                              +--> [ ] cff-x0if bounded qualification
                                          |
                                          +--> [ ] cff-zov5 capability epic

Separate later comparison: [ ] cff-nkbj
Parallel recurrence families: [ ] cff-0f3g, [ ] cff-dlrj
```

Intelligible-ticket review: complete. Capability ownership is split by phase:
cff-iur0 owns the mathematical decision, cff-ocxl owns the model-independent
component protocol and regime/trajectory binding, cff-x0if owns the bounded
development result, and this epic owns only their combined capability claim.

Wiggle-audit review retained clear contextual uses. `fixed` is the pinned
`0.150` ceiling; `generic` and `reusable` mean the model-independent component
protocol exercised by the named composition fixtures; `integrate` means its
compiler/runtime wiring; `preserve` means retaining a named posterior
dependence or D9 pair; `same` means one continuous-state value; `implement`
means cff-ocxl's planned internal integration; `support` and `exact` refer to
literal probability support; `complete` names the whole graph layout or path;
`ready` means dependency-ready; `remaining` and `later` name deliberately
excluded work; `approximation` names D9 or cff-iur0's selected mathematical
family; `close` means ticket acceptance; `certification` is the excluded
long-rollout gate; `convergence` is the historical outer fixed-point status;
`evidence` is a failure diagnostic; `exactly` fixes the three-dataset count;
`match` occurs only in the project phrase mix-and-match; `oracle` means an
independent numeric reference; `parity` means the excluded published-reference
claim; `pass` means a named oracle, fixture, or threshold succeeds; `sound`
means another method satisfying the acceptance criteria; `validation` names
input rejection or later empirical work; `fit` means fixed-dataset parameter
estimation; `infer` means hidden-state posterior calculation or the declared
component update; `learn` means local posterior algebra or the integrated
fitting workflow as stated; `recover` means estimating withheld transition
functions; `scale` names the hostile-size contract; `small` means the lowest
order accepted dependence or a bounded oracle; `can` asks the epic's explicit
capability question; `may` grants methodological choice; `should` gates the
separate comparison; `any` is limited to documented failures; `only` marks the
named information and ticket boundaries; `current` means the mathematical
source regime, not repository recency; and `existing` means the three sealed
qualification artifacts after they have run.

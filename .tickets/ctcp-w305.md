---
id: ctcp-w305
status: in_progress
deps: [ctcp-6msb, ctcl-p70e, cctr-1vvy]
links: []
created: 2026-11-28T04:50:36Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-2gev
---
# Implement the serialized update for changing transition columns

Apply the chosen generative transition mechanism only to explicitly declared transition learnables. The stationary path serves as the control and must keep its current behavior.

## Design

New IR field and learnable/checkpoint feature paths. Support arbitrary named factored transition blocks. Evaluate host/JIT, single/batch, both schedules and schema/legacy round trips. First measure the missing/inferred-categorical update against the derivation and bound runtime storage.

## Acceptance Criteria

Meet cff-2gev sections 2, 4, 5 and 6. Preserve tied ownership, immutable priors, structural supports, executed-data cadence, checkpoint identity and engine status. Document exact and projected moments and show stationary equivalence for posteriors, messages, local free energy and MAP on the frozen validation corpus.


## Notes

**2026-11-28T06:39:11Z**

Starting from the published stationary feature at 294994d (#230) above the D1a derivation (#225). The New IR field and Learnable or checkpoint feature rows apply to declaration/hash, scope, compilation of plan metadata and both raw-count and expected-log target layouts, real-event folding, reset prior, and current/legacy restore. The Runtime numerical or status invariant row applies to finite projected concentrations, masked column validity, host/JIT/vmap and atomic updates. GFE-TM uses the existing D1 node formula on its allowed simplex face. New factor family, stochastic key, and a new schedule N/A. Test stationary default and explicit h=0 against the frozen prior implementation before modeling changes. An explicit support mask must distinguish excluded coordinates from numerical underflow or a sentinel collision; never infer a new structural face from an updated posterior.

**2026-11-28T06:39:21Z**

Dependency direction is prerequisite --> dependent. [closed] ctcp-6msb (D1a) + [closed] ctcl-p70e (native stationary control) + [ready] cctr-1vvy (pure projected update) --> ctcp-w305 (declaration, learning and restore) --> ctcp-qodi (frozen behavioral comparison). Structural-face encoding belongs to the runtime integration; changing allowed topology over time is outside the D1a model.

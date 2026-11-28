---
id: ctcp-w305
status: open
deps: [ctcp-6msb, ctcl-p70e]
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


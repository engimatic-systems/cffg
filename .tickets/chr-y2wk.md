---
id: chr-y2wk
status: in_progress
deps: []
links: [chr-odhx]
created: 2026-11-25T06:05:45Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-wbhu
---
# Compile and qualify mixed-GFE fixed-support controller phases

Mixed generalized-free-energy runs retain a host decision per numerical iteration. Move fixed-support work behind a reusable device program while keeping the current undamped root, stationary free energy, inner status and exact support as the certification authority. Compare single and batched outcomes and record costs.

## Design

Architecture change-path: runtime numerical/status change. Compare Engine, Colony and the host trust reference. State and checkpoint format are unchanged. Preserve dynamic parameters, support discovery, phase boundaries, failure facts and each member’s counters. A fixed support is the set of feasible labels on each message. A trust step limits its movement in reduced message coordinates.

## Acceptance Criteria

Use targeted boundary, inner-failure, scale and float32 tests. Measure warm/cold, representative budget and batch footprint. Document any controller stage whose qualification remains open. Larger runtime integration and final suite belong to cff-wbhu.


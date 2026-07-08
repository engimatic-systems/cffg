---
id: cff-dmfi
status: in_progress
deps: [cff-2wse]
links: []
created: 2026-07-08T02:57:59Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-ql2n
---
# viz: deterministic rectilinear grid layout, edit-stable

Integer-grid placement: columns from t{i}/ slice prefixes + glue-chain topology (time left-to-right); lanes per paper POMDP convention (param terminals above, state chain middle, obs/goal composites below); orthogonal wire routing. Stability: placement is an insertion-ordered function of stable names so adding a box/wire inserts a row/col and preserves all other relative positions.


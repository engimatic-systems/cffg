---
id: cff-50j0
status: closed
deps: [cff-t6ec]
links: []
created: 2026-07-08T02:57:59Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-ql2n
---
# viz: golden SVG tests + layout-stability regression

Golden files for tmaze, gridworld, hand-built HMM (conftest) at levels 0 and 4. Stability regression: add a factor+wire to a template, assert positions of pre-existing elements change only by the inserted row/col shift (relative order preserved). Add new test file to dashboard EPIC_FILES.


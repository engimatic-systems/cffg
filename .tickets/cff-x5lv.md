---
id: cff-x5lv
status: open
deps: [cff-9bex]
links: [cff-bxk4]
created: 2026-07-09T02:46:53Z
type: feature
priority: 2
assignee: coopwilliams
---
# viz: label placement — terminal-end parameter labels + staggered overhang labels

Labels collide on high-degree factors (LGSSM gallery: parameter label 'zn' printed on top of variable label 'x0c'; 'zm' on 'x2'; terminal 'y0' on wire label 'y_0'). DESIGN (Cooper, 2026-07-09, two rules): (1) PARAMETER wires are labeled at the TERMINAL end only — the blob name sits beside the pointmass square, the papers' own convention (Fig 22's d/A/c) — never mid-wire; VARIABLE (structural) wires keep mid-wire labels and get priority for label space. (2) When parallel wires share a side, stagger their lengths like signal bars — each wire at least one label-width longer than the previous — and place each label on the overhang segment that is uniquely its own, so every label is unambiguously adjacent to exactly one wire. Elastic terminal wires make the stretching free; structural wires should rarely need it after the routing fix. Depends on the routing ticket (side spreading reduces how often staggering is needed at all). Code pointers: layout.py _label_pos (wire labels), Terminal label placement (~line 374), svg.py label rendering. Acceptance direction: a label-collision property test (no two label bounding boxes overlap on the canonical + LGSSM models at all compression levels).

## Acceptance Criteria

Zero overlapping label boxes on canonical + LGSSM diagrams at all levels (property test); parameter names appear at terminals only; goldens refreshed once.


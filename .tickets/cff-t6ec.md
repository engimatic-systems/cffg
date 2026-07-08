---
id: cff-t6ec
status: in_progress
deps: [cff-dmfi, cff-fcg2]
links: []
created: 2026-07-08T02:57:59Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-ql2n
---
# viz: SVG backend (paper style) + save_svg/save_png via optional extra

Scene graph -> SVG string, dependency-free: monochrome paper style, math-italic labels with subscripts (z_t), factor-type glyphs (Cat, T, TM, =, Dir), ellipsis for rollout continuation. save_png via cairosvg behind uv optional extra 'viz' with a clear ImportError message. Default compression level 4.


---
id: cff-bxk4
status: open
deps: [cff-4e5z]
links: []
created: 2026-07-08T16:50:10Z
type: task
priority: 3
assignee: coopwilliams
parent: cff-ahps
---
# viz: Gaussian/continuous domain rendering

Extend the diagram pipeline for continuous variables: dimension labels on Gaussian edges (papers annotate domains), distinct terminal glyphs for Gaussian priors/observed edges, GFE-composite boxes unchanged. Small; mostly diagram.derive + golden SVG refresh. DELEGATION: Opus can do this whole ticket against the golden-SVG test pattern; top-level review only.

## Acceptance Criteria

LGSSM template renders correctly at all compression levels; golden SVGs added; layout stability tests green.


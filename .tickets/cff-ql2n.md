---
id: cff-ql2n
status: open
deps: []
links: []
created: 2026-07-08T02:57:39Z
type: epic
priority: 1
assignee: coopwilliams
---
# M8: Plotting facility — paper-faithful CFFG diagrams

Render Template as CFFG diagrams in the paper's visual algebra (Part I §5): squares/wires/beads, form-constraint symbols, P-substitution squares, data terminals, GFE composite boxes, rollout ellipsis. Pipeline: Template -> diagram IR (uncompressed annotation) -> compression passes 1-4 (p.19 recipe) -> deterministic rectilinear grid layout (time left-to-right, edit-stable) -> scene graph -> SVG (PNG via optional cairosvg extra). Scene graph carries source-element IDs for future GUI hit-testing. Default render level: full compression. MP-schedule overlay is a separate backlog ticket.


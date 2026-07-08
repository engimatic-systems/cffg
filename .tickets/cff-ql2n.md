---
id: cff-ql2n
status: closed
deps: []
links: []
created: 2026-07-08T02:57:39Z
type: epic
priority: 1
assignee: coopwilliams
---
# M8: Plotting facility — paper-faithful CFFG diagrams

Render Template as CFFG diagrams in the paper's visual algebra (Part I §5): squares/wires/beads, form-constraint symbols, P-substitution squares, data terminals, GFE composite boxes, rollout ellipsis. Pipeline: Template -> diagram IR (uncompressed annotation) -> compression passes 1-4 (p.19 recipe) -> deterministic rectilinear grid layout (time left-to-right, edit-stable) -> scene graph -> SVG (PNG via optional cairosvg extra). Scene graph carries source-element IDs for future GUI hit-testing. Default render level: full compression. MP-schedule overlay is a separate backlog ticket.


## Notes

**2026-07-08T03:26:08Z**

Delivered: cffg.viz pipeline (derive -> compress -> layout -> scene -> SVG/PNG). All five build tickets closed; 25 new tests incl. goldens + edit-stability regression; T-maze render matches Part I Fig 26, level 0 matches Fig 14 conventions. Verified against paper figures visually (PNGs via cairosvg; on this Mac needs DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib for brew cairo). Remaining thread: cff-jude (MP schedule overlay, Figs 23/24/27) stays as backlog — scene graph already anchors overlays to wires by ref. Closing epic; jude is independently actionable.

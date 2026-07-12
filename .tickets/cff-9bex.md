---
id: cff-9bex
status: open
deps: []
links: [cff-bxk4]
created: 2026-07-09T02:46:53Z
type: bug
priority: 2
assignee: coopwilliams
---
# viz: wire routing at high-degree factors — no interior crossings, spread to free sides

Five-connection factors (gaussian_linear: out, in, + 3 parameter terminals; visible in the LGSSM gallery diagrams, test_kalman rendering tests) produce wires that enter a box on one side and jog diagonally THROUGH the box interior to their attachment point, and bunched parallel entries whose geometry collides with the in-box decorations (domes/beads). DESIGN (Cooper, 2026-07-09): terminal/parameter wires are ELASTIC — they may lengthen and bend freely — so before stacking a second wire onto an occupied side, fan parameter wires out to unoccupied sides (many boxes have a free top/bottom); only when all four sides are genuinely occupied does a side take multiple attachment slots, evenly spaced, with the wire routed entirely OUTSIDE the box. Structural (factor-to-factor) wires keep routing priority: straightest path, claimed first. Code pointers: cffg/viz/layout.py — _route_sides claims sides (structural first, ports, then blobs via first_free/_TERMINAL_PREFERENCE), _anchor_points distributes same-side wires; the interior-crossing defect appears where a claimed side's anchor and the polyline's approach direction disagree, and where internal bead/dome leader lines meet crowded borders. Acceptance direction: golden SVGs for a 5- and 6-degree factor; property: no wire segment intersects any box interior it does not terminate on; layout-stability tests stay green.

## Acceptance Criteria

No wire crosses a box interior on the LGSSM diagrams at every compression level; 5/6-degree golden SVGs added; existing golden + layout-stability suites green.


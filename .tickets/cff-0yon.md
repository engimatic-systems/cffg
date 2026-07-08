---
id: cff-0yon
status: in_progress
deps: []
links: []
created: 2026-07-08T03:38:18Z
type: task
priority: 2
assignee: coopwilliams
tags: [viz, dashboard]
---
# viz: per-test diagram gallery in the dashboard

Tests that render diagrams record SVGs (plus JSON sidecar with the test id) into gitignored tests/.artifacts/ via a viz_gallery fixture. tools/dashboard.py copies them to ~/agent-share/cffg-viz/ and renders rows that have images as light-tinted <details> twirl-downs; images lazy-load and click through to the raw SVG. Also fix run_suite classname parsing: class-based tests currently file under Unassigned.


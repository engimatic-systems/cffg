---
id: cff-0yon
status: closed
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


## Notes

**2026-07-08T03:42:27Z**

Landed. 14 diagrams across 9 test galleries; dashboard copies artifacts to ~/agent-share/cffg-viz so the page works over the share. tests/.artifacts is gitignored and wiped per pytest session — repo stays clean. Classname fix means class-based tests now group correctly (one-time NEW badge churn this run).

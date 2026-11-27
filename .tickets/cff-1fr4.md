---
id: cff-1fr4
status: closed
deps: []
links: [cff-xikb]
created: 2026-11-27T03:37:38Z
type: feature
priority: 1
assignee: coopwilliams
---
# Guide Studio users through models and make graph connections unambiguous

## Background in Plain English — WHY

Cooper found that the first edit in a blank model is blocked and some wires appear to connect to the wrong factors. New users and graph authors need a way to explain the interface and walk through a saved model.

## Outcomes / Acceptance Criteria — WHAT

1. A user can preview and commit the first variable and factor in a blank model.
2. A remembered Open-model filter can hide files with known loading errors before pagination.
3. The browser tab shows the workstation logo. Hovering a variable reveals its repeated uses while preserving the selection.
4. Time-indexed recurrences read from left to right. Factor positions and wire routes keep unrelated edges outside boxes. Factor glyphs fit their boxes.
5. An accessible introductory tour guides users through the workstation. Agents can publish, revise and share an ordered tour of a particular model through the model transaction interface. Project exchange and history preserve it.
6. The updated private deployment works through the existing agent-share link.

## Methodological Suggestions — HOW

Use the Studio model editing and experiments change path. Check browser canvas and outline, shared scene and exported SVG, host index and browser results, revision-based agent commands and Studio project exchange. Solver, JIT, batching, checkpoint and native numerical bundle semantics are N/A. Add numerical geometry checks and exercise keyboard and pointer paths in a real browser.

## Future work

No downstream ticket is required.

## Definitions

A factor is a model component attached to variable wires. An anchor is a persistent identifier for a graph element. A guided tour is an ordered sequence of explanations pointing to parts of the interface or model.


## Notes

**2026-11-27T03:51:41Z**

Verified in Chromium on desktop and phone: entrance to the five-step rSLDS tour, intro playback, pointer and keyboard focus, stable tour links, author updates, project round-trip, plate stops, deleted-target recovery during playback, prior creation, schema filter and favicon. The actual rSLDS had manual offsets on z0_prior and switch2. Those offsets now apply to chronological columns when valid. Changing the route never changes the numerical graph. The older grid is used for saved offsets that would otherwise collide.

**2026-11-27T04:20:57Z**

Geometry research: x0_dynamics is the wire from x0_rest_split.b to dynamics1.in, while z0_prior supplies z0 to switch1.in. The old three-step layout routed it through the unrelated prior, and the six-step layout also routed x4 and x4_switch through emission5. Geometry checks on both horizons and all five compression levels found zero crossings through unrelated boxes at 3-pixel clearance, zero detached endpoints, and strictly increasing time slices. Source: template incidence, Document.describe and build_scene, tests/test_viz_layout.py, and /tmp/cffg_geometry_oracle.py. Confidence high. Chromium confirmed the blank form previously rejected the valid empty-valued prior choice, and passing acceptance verifies the picker filters ahead of 120 result rows. SVG group tests measured horizontal NMA clearance, shared and per-copy hover isolation, and reachable tour targets in a virtualized scene. Final deployment reached via the existing private launcher; model tour was the only change to the live rSLDS revision, with identical graph digest, learned state and arrays. Private HTTPS verified by host browser. Dashboard-backed suite at 6a1393b passed 2765 tests, 337 opt-in deselected in 1186.747 seconds wall time. The 45 focused tests, browser acceptance (desktop and 390px viewport), ruff and ticket dependency checks passed. No pending work.

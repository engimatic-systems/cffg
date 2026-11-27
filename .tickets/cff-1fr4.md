---
id: cff-1fr4
status: in_progress
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


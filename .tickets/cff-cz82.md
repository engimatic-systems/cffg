---
id: cff-cz82
status: open
deps: [cff-x6c0, cff-7piw]
links: [cff-9bex]
created: 2026-11-26T19:44:48Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-xikb
tags: [studio]
---
# Edit a Forney graph from a keyboard-first workstation

## Background in Plain English — WHY

The paper notation uses factor boxes and variable wires. Cooper needs to search and edit that notation quickly on a remote screen, with enough context to see what a change affects.

## Outcomes / Acceptance Criteria — WHAT

1. A keyboard-only walkthrough creates, opens, annotates, wires, renames, resizes, initializes, undoes and saves models using a canvas, an outline and an inspector.
2. All editing actions use named commands and can be rebound by context. Conflicts, browser-reserved keys, reset and import/export are covered. Focus returns predictably after commands.
3. Canvas and outline select the same scoped entities, including repeated plates and parallel or disconnected components. Invalid gestures leave the accepted graph intact.
4. Interaction latency, virtualized large-graph navigation and continued use during compilation meet the proposal budgets on a documented client.

## Methodological Suggestions — HOW

Build from cffg.viz.build_scene and the action catalog. Run browser tests through Python Playwright. Use the New IR field row only if representation changes. Host/JIT, schedule, vmap and checkpoint decode are N/A for UI-only work.

## Future work

Agent review and live runs use the same panels, revision identities and command registry.

## Definitions

Forney notation draws variables as wires. A plate represents repeated structure. An inspector edits the selected item's properties.

---
id: cff-cz82
status: closed
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

## Notes

**2026-11-26T21:13:28Z**

Shipped keyboard commands, editable keymap, outline, scoped plate navigation, grid offsets, command search and accessible diagram semantics. Chrome browser smoke covered creation, F2 rename, note, save/export/reopen, invalid edit rejection, run, stepping, profiling and responsive 1440×900 and 390px layouts. The full private HTTPS launcher also opened the real rSLDS and its measurements. Research: how do dense scientific graphs behave under pan and zoom? On the M2 Max, Chromium 153, 891×530 viewport, 5,000 factors + 5,000 wires (1,012–1,128 in view), the high-resolution local p95 was 16.67ms for both gestures. Default timer read 16.7 and 16.8ms. Keyboard p95: palette 33.2ms, outline 34.2ms, selection 28.3ms. Dense low-zoom views keep selected labels. Evidence: graph.js, browser-perf-high-resolution.json and browser-perf-final-chrome.json in /tmp/cffg-studio-qa. High confidence on this client; transport latency varies.

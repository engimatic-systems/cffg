---
id: ccrp-fuob
status: in_progress
deps: []
links: [cns-t0b9]
created: 2026-11-25T19:36:00Z
type: task
priority: 2
assignee: coopwilliams
---
# Explain the navigation test through a recorded trip

## Background in Plain English

Cooper found the navigation report hard to follow. A reader should be able to explain what the traveler sees, what each map remembers, and how those choices affect the turn at the first junction.

## Outcomes / Acceptance Criteria

1. Start the guide with the task and a worked Amber-to-Blue trip using the recorded actions.
2. Explain the two candidate maps, the tile comparison, and how the prediction test differs from reaching a goal.
3. Restate both layouts and their results in everyday language. Verify desktop and mobile replay, archived scores and served HTML after the rewrite.

## Methodological Suggestions

Work from the recorded steps. This is presentation-only reporting within isolated tooling. The engine, model, measurement, serialization, scheduler and Colony change paths are N/A. Use the existing browser and archive checks.

## Future work

The follow-up on learning movement sequences belongs to ccrp-x8yx.

## Definitions

A junction is a tile where paths meet. A route is the series of moves from a named starting tile to a named destination.


## Notes

**2026-11-25T19:36:07Z**

Use the completed study for the actual junctions, moves and goals. Update the HTML template and served guide and refresh its archive verification.

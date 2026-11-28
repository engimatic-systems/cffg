---
id: cff-47w9
status: closed
deps: []
links: []
created: 2026-11-28T00:45:04Z
type: feature
priority: 1
assignee: coopwilliams
---
# Show the route between adjacent guided tour stops

## Background in Plain English
Cooper wants to see where each stop sits in a large graph. Traveling directly between close-up views hides the distance and direction of the move.

## Outcomes / Acceptance Criteria
1. Playback offers a keyboard-operable Animations toggle that defaults on and remembers an explicit choice. Honor a viewer who already requested reduced motion until they choose otherwise.
2. Moving forward or backward zooms out for context, pans across the graph, and zooms back in. The framing works for groups, overviews and plate boundaries.
3. Turning the toggle off lands immediately. Rapid paging and exit cancel prior motion.
4. Verify the visible route, destinations, keyboard operation and preference in targeted GUI tests and check the deployed private link.

## Methodological Suggestions
The architecture change path is Studio model editing and experiments. Check the browser tour controller, canvas camera, interface tour, agent-authored model tour and tooltip catalog. Agent/API data, native bundles and numerical execution are N/A because the preference belongs to the viewer.

## Future work
No further work is identified.

## Definitions
A stop is one narrated view. A plate is a repeated graph fragment with its own canvas.


## Notes

**2026-11-28T00:51:11Z**

Implemented the viewer-local Animations checkbox, a distance-aware three-stage camera path in both directions, orientation via the enclosing plate box, and an animated outline for the introductory tour. Toggling off during travel settles the current stop. Reviewed Carbon wording, requested a correction distinguishing the two kinds of tour, and approved the corrected explanation. Targeted GUI verification: tools/check_studio_tour_motion.py passed in Chromium and WebKit, including both stop bounds in the mid-pan viewport, reverse paging, keyboard toggle, interruption, reload and reduced motion. Six grouped-tour scenarios and 390×345 long narration checks passed. Verified the agent-share launcher and existing rSLDS tour on the private deployment e5101e8753773f5f: actual zoom moved from 2.2 to 0.848 and back to 2.2, the tooltip appeared, and the toggle persisted. No model data changed.

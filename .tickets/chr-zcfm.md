---
id: chr-zcfm
status: open
deps: [cff-pyy9, cff-4yr6, cff-3kdk]
links: []
created: 2026-11-25T06:04:34Z
type: feature
priority: 2
assignee: coopwilliams
---
# Resume Craftax with a hierarchical survival and navigation model

## Background in Plain English — WHY

The first-descent controller on experiment/craftax-live entered the dungeon in seven of eight lanes. All 16 episodes that entered eventually ended in death. It chose actions through CFFG but stored its map in Python and used hand-built forecasts. Cooper paused game experiments to improve the library first. The last verified experiment and scale investigation are ccl-ydjn and ccl-ybsp on the Craftax worktree. The frozen controller is at ff845a4 and its results are served at http://localhost:8080/craftax-live/first-descent/results.html.

## Outcomes / Acceptance Criteria — WHAT

1. Resume only after the M7 hierarchy, batched observation binding and selective posterior readout dependencies are complete and Cooper resumes the experiment.
2. Compare a slow readiness or regional navigation model driving a fast movement and survival model against a flat controller using matched observations, actions, environments and compute budgets. Freeze the primary outcome before inspecting results. Count every starting lane and handle episode cutoffs explicitly.
3. Identify which location, hazard and resource beliefs the graph actually infers, which action consequences are learned or specified, and which policy decisions still use game-specific code.
4. Scope a weighted local transition for uncertain player position before choosing its implementation. Specify the directed support for each action (legal moves, blocked attempts, staying put, collisions, boundaries and teleports when applicable); how a viewport and terrain evidence update pose; how to represent multiple possible positions and uncertain walkability; and when the map grows or changes floor. Specify probability normalization, zero-support behavior, duplicates, updates to weights or topology, learnable versus fixed rows, and what the factor must return in both directions, to the action selector, and to the free-energy ledger. Determine when an existing factorization suffices and when a neighbor-indexed weighted transition is required.
5. Define a small dense reference including ambiguous motion, obstacles, impossible destinations and sparse support. Measure posterior, action and ledger parity, scaling in allowed moves, peak memory, artifact size and warm decision time at fixed precision before choosing a public sparse-factor API. Report output equivalence to the dense table and the impact of coupling pose to uncertain tiles.
6. Give the resumed experiment a short goal/change description, native graph visualization, frame-level replay and a report that distinguishes dungeon entry, survival and return.

## Methodological Suggestions — HOW

The previous probe measured a connected 860-factor tree on an M2 Max. Bulk loading 99 tile bits and requesting two beliefs reduced median model time from about 37 ms to 3.58 ms in the research prototype. Use the public replacements. One candidate primary endpoint is first dungeon entry plus 200 further actions alive within a fixed lane budget. Keep survival after entry as a secondary outcome. For sparse movement compare a dense action-by-source-by-destination table with a bounded list of destinations and weights per action and source, including reverse lookup or segmented reductions. Benchmark a small patch and a whole floor.

## Future work

The M7 and input/output tickets below own reusable library changes. Decide whether a general weighted sparse transition merits a separate implementation ticket from this scoping evidence. This ticket preserves a paused experiment and does not authorize new Craftax training runs.

## Definitions

A lane is an independently seeded stream of game episodes. Pose is the player location, possibly with direction. Support is the set of outcomes with nonzero probability. A dense transition stores all source/destination pairs, including impossible moves. A cavity message passes evidence back without reusing the recipient’s own prior.


## Notes

**2026-11-25T06:04:34Z**

Legend: prerequisite --> dependent. Implementation can proceed in parallel on the three roots. The Craftax experiment itself is parked.

cff-wbhu (compiled sweep loop) --> cff-9xmq (balanced broadcast) --> cff-mjku (interior branches) --> cff-pyy9 (M7) --+
cff-4yr6 (batched observations, ready) -----------------------------------------------------------------------------+--> resume Craftax
cff-3kdk (selective readouts, ready) --------------------------------------------------------------------------------+

**2026-11-25T06:27:42Z**

Library timing on the frozen 860-factor spatial surrogate (1,116 variables, same 25 warm calls) is a starting point for resumption. Input binding fell from 22.036 ms to 0.219 ms and requested posterior readout from 14.113 ms to 0.080 ms. End-to-end median decision time fell from 39.853 ms to 3.995 ms. Both sets of 52 recorded marginal arrays and 52 energies matched exactly. The inference phase itself took 2.263 and 2.276 ms. Sources ff845a4 and 9da2f7a, raw comparison ~/agent-share/hierarchical-runtime/spatial-proxy/comparison.json. The surrogate is one fixed graph and contains no game episodes. Real map size, transition support and end-to-end controller behavior still need the measurements in outcomes 4 and 5 when Craftax resumes.

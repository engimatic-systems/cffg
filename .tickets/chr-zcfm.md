---
id: chr-zcfm
status: closed
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
4. Give both arms the same rule-based navigation proxy. Read visible interaction geometry from the player-centered symbolic observation. Use exact overworld tile coordinates and exact displacements relative to the first landing on a lit floor. Carry known ladder endpoints across revisits. If a move is unobservable, report the loss of a map coordinate and keep using visible offsets. Keep native simulator coordinates confined to independent verification.
5. Check the positioning proxy against native movement on the frozen Experiment 2 action tapes, including blocked moves, floor changes and episode resets. Report coverage and discrepancies separately for each floor. Keep the agent observation and action spaces identical across arms.
6. Give the resumed experiment a short goal/change description, native graph visualization, frame-level replay and a report that distinguishes dungeon entry, survival and return.

## Methodological Suggestions — HOW

The previous probe measured a connected 860-factor tree on an M2 Max. Bulk loading 99 tile bits and requesting two beliefs reduced median model time from about 37 ms to 3.58 ms in the research prototype. Use the public replacements. One candidate primary endpoint is first dungeon entry plus 200 further actions alive within a fixed lane budget. Keep survival after entry as a secondary outcome. The local navigator may route to observed targets with breadth-first search. The hierarchical and flat controllers can then be compared on readiness, supplies and danger using identical movement assistance.

## Future work

All three library prerequisites are complete. A probabilistic spatial model is optional research for tasks where hidden movement changes the decision. If it becomes useful, first define the directed moves per action, including blocked moves, stay, collisions, boundaries and floor changes. Specify what is uncertain about pose and terrain, how to normalize weights and handle duplicate or zero-probability destinations, and whether connectivity or weights can change or be learned. Check messages to the previous position, next position and action, plus the free-energy term, against a small dense reference. Compare time, peak memory and artifact size as the map grows before proposing a public sparse-factor API. Cooper has authorized positioning work. A new scored survival experiment is a separate step.

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

**2026-11-25T07:59:10Z**

Resume point. All three direct prerequisites are closed and this ticket appears in tk ready. No new Craftax episodes or training were run. The next experiment still requires Cooper to resume it. The 8× Hallway throughput goal chr-d4dl is independent and remains open. The local transition contract and dense parity check in outcomes 4–5 are the first modeling task on resumption.

Prerequisite --> dependent
[x] cff-wbhu --> [x] cff-9xmq --> [x] cff-mjku --> [x] cff-pyy9 --+
[x] cff-4yr6 -------------------------------------------------------+--> [ ] chr-zcfm (ready, parked)
[x] cff-3kdk -------------------------------------------------------+

**2026-11-25T14:05:52Z**

Cooper chose a navigation proxy for the survival comparison. Outcomes 4–5 now supersede the earlier proposal to make sparse transition scoping the first experiment. Full Craftax-Symbolic-v1 exposes visible tile offsets and facing. Default reset fixes the overworld origin at (24,24). A cardinal move checks the destination before mobs move. The first dungeon lands at a generated ladder, so its map uses entry-relative coordinates. Dark destinations can hide collisions on later floors. The unchanged three hard prerequisites are closed.

**2026-11-25T14:07:35Z**

Positioning is ready for the next controller. Experiment branch experiment/craftax-live commit ea90752 (ticket ccl-n8rp) provides tools/craftax_live/position.py and an independent native scorer. Every archived action and reset matched: 45,353 absolute overworld steps plus 116 resets, and 3,799 first-dungeon steps in entry-relative coordinates. All 24 descents were included. A later unobservable move can make the floor position unknown while visible offsets remain exact. Receipt ~/agent-share/craftax-experiment2-verification/position-TkxfoH/positions.json. This replay reused Experiment 2 actions; it does not measure policy improvement.

**2026-11-25T15:17:12Z**

Resolved by Experiment 3 on local branch experiment/craftax-survival at 34189b1. The source-frozen runtime dd4128b used the public M7 hierarchy in Craftax against a matched flat joint filter. Both policies used the same observation-only navigation and action graph. Eight paired lanes of 2,048 real actions per arm yielded 5 vs 6 first dungeon entries and 0 vs 1 lanes surviving another 200 actions. Independent replay checked every action, recorded position, native frame and lane outcome. The report, raw evidence and all episode replays are accessible at http://localhost:8080/craftax-live/dungeon-survival/results.html. Project validation passed 2,751 default tests, with 335 deselected and 1,178.203 s wall time. The experiment branch carries the full qualification and implementation.

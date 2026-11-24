---
id: cao-srin
status: in_progress
deps: [car-ne0a, car-8rgl]
links: []
created: 2026-11-24T14:50:52Z
type: task
priority: 1
assignee: coopwilliams
parent: car-zieb
tags: [axiom, benchmark]
---
# Run the frozen AXIOM 10,000-step primary evaluation

## Background in Plain English — WHY

Cooper approved the corrected default and asked for the full 10,000-step AXIOM evaluation. The 24-step acting test established that a plan reaches the game. Long-running reward and restart behavior are open. The frozen primary comparison is the untuned CFFG agent against the paper's published per-game returns.

## Outcomes / Acceptance Criteria — WHAT

1. Run the predeclared 10,000-action development pilot at the full search budget. Record the real actions, rewards, costs, resets, periodic reductions and any failure.
2. Run all 20 engineering qualification cases and verify step-500 exact continuation against uninterrupted execution before opening scored seeds.
3. Evaluate the 100 frozen game/seed pairs for 10,000 native actions each. Retain failures and their denominators. Only complete samples can receive a primary verdict.
4. Publish plain-English results, source and protocol identities, per-game and macro comparisons, startup and warmed cost, and recoverable run records. Preserve the frozen numerical options and arm.

## Methodological Suggestions — HOW

Use the pinned MIT simulator and an independently implemented CFFG agent, in separate processes. Freeze the candidate before qualification. On an interrupted process replay the recorded simulator actions and verify the observations before restoring the exact agent checkpoint. Architecture change paths are the rolling-agent boundary, numerical/status behavior and serialization. Test fresh versus resumed, actual versus imagined input, pre- versus post-reset, source/seed binding and terminal failure. Alternate graph schedules are N/A to this experiment.

## Future work

car-zieb tracks the ancillary ablations and visual perturbation comparisons. car-ozw0 tracks the provenance of direct original-agent outputs.

## Definitions

One real action advances the game once. Each decision compares 512 candidate action sequences, each 32 imagined steps long with three draws. The score for a seed is its sum of 10,000 real rewards. A seed fixes the starting random streams for one run.

## Notes

**2026-11-24T15:09:43Z**

Full-budget Explode development pilot launched from immutable e741943d820c25fdedfee1059efacc6c5fb52221. Source digest b2dd424ff72de822ac4147bede47f3e733fc04e40d9a577cadc34741902a3673, model seed 88101, game seed 180400, real budget 10000. ROOT is ~/agent-share/axiom-10000. Live report at http://localhost:8080/axiom-10000/index.html. Zero-score or poor early returns do not stop this registered run. The supervisor writes the native action log, exact checkpoint every 500 and a stage receipt on completion. Scored seeds remain behind the published qualification gate.

**2026-11-24T16:21:54Z**

At implementation head 152866c the default suite passed all 2808 selected tests, with 380 slow tests deselected. Dashboard wall time 1333.136 seconds, pytest 1317.22 seconds. The ongoing isolated development run shared the CPU for this validation. The measured selection adjustment is being tracked separately and can reuse this passing result. The scored agent is the separately sealed e741943 snapshot.

**2026-11-24T17:19:15Z**

Cooper chose to report the 10,000-action Explode run first. This session finishes development/Explode-88101 and reports its complete native reward and costs. Both later study stages remain unlaunched. Their original protocol, seeds, and scoring rules stay in the ticket for a future decision.

**2026-11-24T22:35:42Z**

Requested Explode milestone complete. Immutable e741943 on model seed 88101/game seed 180400 scored +157 (199 catches, 42 misses, 9759 neutral actions); first half +64, second half +93. Historical untuned Table 1 mean 180, SD 30 across 10 seeds; one point difference of -23 is descriptive. Worker ran 26726.495 s (7 h 25 min), all 20 scheduled reductions, no restart. A separate fresh simulator reproduced all 10001 initial/action receipts, including every reward and frame hash. Final trace SHA256 c639ca630fc55cd11f8c4fe4d212b9756fd34f0cb31adef21cf5a864b5572eb3. Native replay and finished report: ~/agent-share/axiom-10000/{explode-native-replay.json,explode.html}; live index serves the report, campaign.html preserves the cumulative dashboard. Cooper requested this single run first. No pre-evaluation or primary workers launched. If resumed, use the same sealed snapshot and receipts. The broader ticket stays in progress for that separate decision.

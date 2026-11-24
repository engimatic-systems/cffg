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

---
id: cff-djod
status: closed
deps: [cff-x6c0]
links: []
created: 2026-11-26T19:44:49Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-xikb
tags: [studio]
---
# Run seeded data and stream status-bearing inference steps

## Background in Plain English — WHY

An experimenter wants to watch inference progress and inspect a particular step without freezing the editor. A single flooding sweep has different semantics from an engine run using a chosen schedule.

## Outcomes / Acceptance Criteria — WHAT

1. A pinned model revision runs on schema-valid, seeded random observations and retained observation tapes. Requests record seed, data digest, engine, dtype, device, schedule and budget.
2. The UI and API can start, step, pause, resume, cancel and replay a run. They distinguish simulation ticks from inference iterations and declare the available stepping granularity for each schedule.
3. Events include completed work, defined free energy, residuals, diagnostics and final converged, exhausted or diverged status. Reconnection restores missed events. Traced runs preserve the reference schedule and terminal state.

## Methodological Suggestions — HOW

Use the Runtime numerical or status invariant row. Check host/JIT, flooding/scheduled and exact, and single/vmap as advertised. Checkpoint decode is N/A unless resuming persisted solver state. Exercise sparse support, vacuous evidence, float32 and 10× horizons. Compare trace-on with the existing public run entry point.

## Future work

The resource profiler and generated simulator tickets consume this run protocol.

## Definitions

A schedule orders message updates. Free energy is the model's inference objective. An observation tape stores exactly which observations a run received.

## Notes

**2026-11-26T21:13:28Z**

Shipped isolated revision-pinned jobs, random inputs, observation tapes, durable status events and replay; tested pause/step/resume/cancel with browser and API. Each trace sample replays an increasing solver budget with the same initial state and hardening disabled. The regular run and repeated trace produced the same terminal beliefs. The saved production rSLDS experiment converged in six iterations; seed, port contract, generation source and observations are retained.

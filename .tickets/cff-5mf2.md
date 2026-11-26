---
id: cff-5mf2
status: open
deps: [cff-djod, cff-j7mm]
links: [cff-6s31]
created: 2026-11-26T19:44:49Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-xikb
tags: [studio]
---
# Display structural estimates, cold and warm timings, and measured memory

## Background in Plain English — WHY

Cooper wants to know whether a proposed graph fits and how fast it runs. A count based on declared shapes, the size of initialized state and the peak memory of a real process answer different questions.

## Outcomes / Acceptance Criteria — WHAT

1. Every accepted revision exposes available structural counts or bounds immediately. Persistent state, temporary executable buffers and measured peak memory have separate fields with provenance.
2. The final report separates process startup, host compilation, JAX lowering/compilation, first synchronized execution and repeated warm times. It identifies the device, cache conditions, repetitions and measurement units.
3. A fixture with Gaussian messages and another with a scheduled run compare the report to actual arrays and synchronized timings. Unsupported metrics display their availability reason. Benchmark runs exclude live instrumentation overhead.

## Methodological Suggestions — HOW

Consume cff-j7mm and reuse the ordinary-iteration benchmark. Coordinate pre-grounding plate counts with cff-6s31. Apply the Runtime numerical or status invariant row and its host/JIT, schedule and single/vmap pairs to each exposed estimate. Checkpoint pairs are N/A.

## Future work

Use the same report to compare revisions and feed critique attachments.

## Definitions

JAX compiles numerical functions. RSS is resident process memory. A warm timing reuses an already compiled executable.

---
id: ccpr-m0bu
status: closed
deps: [ccpr-ojza]
links: []
created: 2026-11-28T05:57:47Z
type: feature
priority: 3
assignee: coopwilliams
parent: cff-6s31
tags: [plates, compiler]
---
# Support nested plates across compilation, learning and saved models

## Background in Plain English

A fragment can itself contain repeated rows, such as sensors within each room. An existing validation guard blocks this case because the old expansion would lose the inner model. Lifting it requires agreement about which names and learning histories belong to each copy.

## Outcomes / Acceptance Criteria

1. Compile and resize a two-level model with multidimensional indices, scoped shared parameters, private parameters, chains, broadcasts and exposed ports. Check the parent's criteria 5.1–5.25 against an independently materialized reference.
2. Each copied variable, port, file reference and learned parameter has its complete scope. Reject name collisions and missing or corrupted array files before publishing or restoring a bundle.
3. A saved model reloads with the same current parameter and learner values, compiled ownership, observable beliefs and free energy. A fragment-level agent declaration remains invalid.
4. Preserve existing single-level, plateless and historical source behavior when admitting nested structures. Verify prior valid source hashes and current checkpoint decoding.

## Methodological Suggestions

Use nested loops in the test reference so it cannot share the compact address implementation with the production compiler.

## Future work

cff-6s31 checks nested resource counts and includes this path in its end-to-end release evidence.

## Definitions

A scope is the sequence of plate names and coordinates that identifies an object in a repeated model. A bundle stores the model source and its arrays.

## Notes

**2026-11-28T07:27:17Z**

Implementing against the actual compact compiler source 2feecc2 while its qualification runs. Architecture change paths: template IR validation and placement, public expansion and native slot layout, cached Colony numeric sources, learnable ownership, compact JSON and CFFGZ. An inner tied parameter belongs to each enclosing copy; a matching directly declared enclosing owner follows that declaration. N-D row-major addressing is shared with PlateGrid. Check explicit flat references, scoped file tokens, malformed and corrupt inputs, resize and restore. Solver kernels, schedule, readout and free-energy rules are unchanged; verify parity on the public paths.

**2026-11-28T07:32:45Z**

Implementation preserves row-major coordinates and resolves wiring inside out. An independent flat model matches the full arena, beliefs, evidence and sparse-support outcomes in f32/f64; nested chains and cascaded Broadcast hubs match the corresponding flat connections. Bundle tests exercise per-room and global calibration ownership, per-sensor priors, manifest corruption, extracted paths and exact learned counts. Initial compatibility run: 134 passed (4 opt-in deselected). After the final numeric-source and cyclic-input changes: 54 focused cases passed. Existing one-level model.json bytes were checked against 2feecc2 on two fixtures. Current release dependency is the reviewed one-level compiler and final short suite.

**2026-11-28T10:58:09Z**

A room can contain a grid of sensors. Sensors in that room can share a calibration and keep individual priors. The compact compiler resolves nested wiring and inherited owners before allocating the graph slots. The editable expansion and the bundle reader use the same coordinate and ownership rules. A direct outer declaration can share a calibration across rooms.

The focused checks include a separately constructed flat model, analytic observation likelihoods, multi-axis resize, chains and Broadcast hubs across levels, cached compilation, cyclic input rejection and restoration of learned counts. The canonical writer retained byte-identical model JSON for two existing single-level examples. Prerequisite https://github.com/engimatic-systems/cffg/pull/238.

The required short suite passed 2,897 cases in 430.349 seconds at 07b7c37a1133.

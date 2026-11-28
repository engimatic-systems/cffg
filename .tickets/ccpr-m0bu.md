---
id: ccpr-m0bu
status: open
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

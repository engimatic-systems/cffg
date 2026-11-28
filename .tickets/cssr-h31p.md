---
id: cssr-h31p
status: in_progress
deps: [cff-du6m]
links: []
created: 2026-11-28T05:28:13Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-cynx
tags: [structure-learning]
---
# Replay structure proposals from their source graph and edits

A reviewer needs to reconstruct which graph was actually scored. A structural hash can be shared by models with different numeric parameters. Define one record for built-in and external proposals, with separate identities for structure and numeric inputs.

## Design

Cover epic items 3, 4 and 25. Reuse Template, the serialized edit bundle and typed-port composition. Record parent, proposal source, ordered edits/compositions, and source/result structure hashes. Bind separate parameter or fragment content identities wherever replay depends on them. Architecture row: New IR field for any new source format. The proposal record belongs to the host. Engine schedules, vectorized messages and learning checkpoint formats are outside this change.

## Acceptance Criteria

Reconstruct an edited and a composed candidate from saved inputs, verify source/result identities and identical evaluation arrays. Wrong sources, missing numeric inputs and invalid edits produce a specific failure without mutating the source. Existing models and edit bundles still round-trip. Demonstrate replay after load and at 10 times the smallest edit count.

## Notes

**2026-11-28T05:42:19Z**

Architecture change path: New IR field, host proposal manifest only. Reuse core edit-bundle and compose interpreters and Template save/load; no change to the model IR, its schema, compiler or checkpoint. Preserve ordered operations and scope arrays with their owning models. Separate a source graph ID and parent candidate ID from the proposal producer, structure hashes and explicit numeric content. Wrong source, invalid edit and missing or changed inputs must fail before exposing a candidate. JIT/host, scheduled/flood and Colony/single execution are unchanged. Exact zeros and dtype/shape are recorded as input bytes; replay performs no probability arithmetic. Exercise at least ten edits.
